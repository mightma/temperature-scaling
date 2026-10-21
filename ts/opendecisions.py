"""OpenDecisions Capability with letter readout and per-question-type temperatures.

    uv run python -m ts.opendecisions --model Qwen/Qwen3.5-9B

One forward pass per item (quyet-v1 prompt), a softmax over the item's option letters, then the
official scorer (ts/vendor/opendecisions_score.py) for Intelligence, Calibration, Capability and TVD.

Temperatures, one per question type (choice / yes-no / score), as in the paper and predict_letters.py:
  t1        T = 1 for every type (zero-shot)
  crossfit  each item's T is fitted by NLL on the other folds only; folds are whole `group`s (an
            ungrouped item is its own group), so no item is scored with a T that saw it or one of its
            variants. This is the honest estimate.
  insample  T fitted on all of OpenDecisions. Optimistic (fitted on the test set); reference only.
  fixed     temperatures given with --fixed-T (e.g. fitted on another dataset); nothing is fitted here.
Fitted temperatures are constrained to T >= --min-T (1 by default, as in the paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import fit_temperature, softmax
from .prompts import render_quyet_v1
from .vendor.opendecisions_score import score as official_score

REPO_ID = "jgeuter/OpenDecisions"
REVISION = "ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe"
TYPES = ("choice", "noul", "score")
TYPE_NAMES = {"choice": "choice", "noul": "yes/no", "score": "score"}
SETTINGS = {
    "t1": "T = 1 (zero-shot)",
    "crossfit": "Per-type T, cross-fitted by group",
    "insample": "Per-type T, fitted on all items (optimistic)",
    "fixed": "Fixed T (--fixed-T)",
}


def load_tasks(revision: str = REVISION) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, "opendecisions.jsonl", repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_index(task: dict) -> int:
    """Letter index of the expected answer under quyet-v1's option order."""
    qtype = task["question"]["type"]
    if qtype == "choice":
        return task["labels"].index(task["expected"])  # options keep the criteria order == labels
    if qtype == "noul":
        return 0 if task["expected"] == "yes" else 1  # A = true, B = false
    return int(task["expected"])  # score levels lowest first


def to_probs(task: dict, p: np.ndarray) -> dict[str, float]:
    qtype = task["question"]["type"]
    if qtype == "choice":
        return {lab: float(p[i]) for i, lab in enumerate(task["labels"])}
    if qtype == "noul":
        return {"yes": float(p[0]), "no": float(p[1])}
    return {str(i): float(p[i]) for i in range(len(task["labels"]))}


def fit_type_temperatures(z, y, types, mask, min_T: float, max_T: float) -> dict[str, float]:
    out = {}
    for t in TYPES:
        m = mask & (types == t)
        out[t] = fit_temperature(z[m], y[m], t_min=min_T, t_max=max_T)["T"] if m.any() else 1.0
    return out


def temperatures(z, y, types, groups, min_T: float, max_T: float, n_folds: int, seed: int) -> tuple[dict, dict]:
    """Per-item temperatures for every setting, and the fitted values for the report."""
    n = len(y)
    everything = np.ones(n, dtype=bool)
    insample = fit_type_temperatures(z, y, types, everything, min_T, max_T)

    uniq = np.array(sorted(set(groups.tolist())))
    fold_of_group = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_folds))
    fold = np.array([fold_of_group[g] for g in groups])
    cross = np.ones(n)
    per_fold = []
    for k in range(n_folds):
        fitted = fit_type_temperatures(z, y, types, fold != k, min_T, max_T)
        per_fold.append(fitted)
        for t in TYPES:
            m = (fold == k) & (types == t)
            cross[m] = fitted[t]

    item_T = {
        "t1": np.ones(n),
        "crossfit": cross,
        "insample": np.array([insample[t] for t in types]),
    }
    return item_T, {"insample": insample, "crossfit_folds": per_fold}


def parse_fixed_T(spec: str) -> dict[str, float]:
    """"1.54" (all types) or "choice,yesno,score" as in predict_letters.py's --temperatures."""
    vals = [float(x) for x in spec.split(",")]
    if len(vals) == 1:
        vals *= 3
    if len(vals) != 3 or min(vals) <= 0:
        raise ValueError("--fixed-T takes one positive value or three (choice,yesno,score)")
    return dict(zip(TYPES, vals))


def paired_bootstrap(tasks, preds: dict[str, list[dict]], units: np.ndarray, n_boot: int, seed: int) -> dict:
    """95% intervals for each setting's Capability and its difference to T = 1, resampling whole
    groups with replacement (the paper's paired bootstrap over question groups)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(units.tolist()))
    members = {u: [] for u in uniq}
    for i, u in enumerate(units):
        members[u].append(i)
    draws = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = [i for u in rng.choice(len(uniq), len(uniq)) for i in members[uniq[u]]]
        # the scorer rejects repeated ids, so give every resampled copy its own id
        sample = [{**tasks[i], "id": f"{tasks[i]['id']}#{j}"} for j, i in enumerate(idx)]
        for key, ps in preds.items():
            sp = [{"id": f"{ps[i]['id']}#{j}", "probs": ps[i]["probs"]} for j, i in enumerate(idx)]
            draws[key].append(official_score(sample, sp)["capability"])
    base = np.array(draws["t1"])
    out = {}
    for key, d in draws.items():
        d = np.array(d)
        out[key] = {
            "capability_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "diff_vs_t1_ci": [float(np.percentile(d - base, 2.5)), float(np.percentile(d - base, 97.5))],
        }
    return out


def write_report(path: Path, r: dict):
    c = r["config"]
    L = [f"# OpenDecisions: `{c['model']}` (`{c['prompt_format']}` prompt)\n"]
    L.append(
        f"- {r['n_items']} items, dataset revision `{c['revision']}`; scored with the official `score.py` "
        "(ts/vendor/opendecisions_score.py)\n"
        "- One temperature per question type; fitted temperatures are NLL-optimal subject to "
        f"T ≥ {c['min_T']:g}\n"
        f"- Cross-fitting: {c['n_folds']} folds of whole `group`s; an ungrouped item is its own group "
        f"({r['n_groups']} units)\n"
    )
    L.append("## Capability\n")
    n_tvd = next(iter(r["settings"].values()))["score"]["calibration_items"]
    L.append(f"| Setting | Capability | Intelligence | Calibration | TVD ({n_tvd} items) | T choice / yes-no / score |")
    L.append("|---|---:|---:|---:|---:|---|")
    for key, s in r["settings"].items():
        sc = s["score"]
        L.append(
            f"| {SETTINGS[key]} | **{sc['capability']:.1f}** | {sc['intelligence']:.1f} | {sc['calibration']:.1f} | "
            f"{sc['calibration_items_tvd']:.3f} | {s['T_label']} |"
        )
    L.append("")
    if c.get("fixed_T"):
        L.append(f"Fixed T source: {c.get('fixed_T_note') or 'given on the command line'}.\n")
    if r.get("bootstrap"):
        L.append(
            f"Paired bootstrap over groups ({c['bootstrap']} resamples; an ungrouped item is its own group), 95% intervals:\n"
        )
        L.append("| Setting | Capability CI | Capability − T=1 CI |\n|---|---|---|")
        for key, b in r["bootstrap"].items():
            lo, hi = b["capability_ci"]
            dlo, dhi = b["diff_vs_t1_ci"]
            L.append(f"| {SETTINGS[key]} | [{lo:.1f}, {hi:.1f}] | {'–' if key == 't1' else f'[{dlo:+.1f}, {dhi:+.1f}]'} |")
        L.append("")
    L.append("## Per question type\n")
    L.append("| Setting | Type | Accuracy % | Competence | Calibration |\n|---|---|---:|---:|---:|")
    for key, s in r["settings"].items():
        sc = s["score"]
        for t in TYPES:
            if t not in sc["accuracy_per_type"]:
                continue
            cal = sc["calibration_per_type"].get(t)
            L.append(
                f"| {SETTINGS[key]} | {TYPE_NAMES[t]} | {sc['accuracy_per_type'][t]:.1f} | "
                f"{sc['competence_per_type'][t]:.1f} | {'–' if cal is None else f'{cal:.1f}'} |"
            )
    L.append("")
    lm = r["letter_mass"]
    L.append(
        "## Letter mass\n\nShare of the full-vocabulary next-token probability on the item's option letters "
        f"(both `\" X\"` and `\"X\"`): mean **{lm['mean']:.3f}**, min {lm['min']:.3f}, "
        f"items below 0.5: {lm['below_0.5']}.\n"
    )
    L.append(
        "## Cross-fitted temperatures per fold\n\n| Fold | choice | yes/no | score |\n|---|---:|---:|---:|"
    )
    for k, f in enumerate(r["fit"]["crossfit_folds"]):
        L.append(f"| {k} | {f['choice']:.3f} | {f['noul']:.3f} | {f['score']:.3f} |")
    L.append("")
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--prompt-format", default="quyet-v1", choices=["quyet-v1"])
    ap.add_argument("--letter-tokens", default="both", choices=["both", "space", "nospace"])
    ap.add_argument("--min-T", type=float, default=1.0, help="lower bound on fitted temperatures (paper: 1)")
    ap.add_argument("--max-T", type=float, default=20.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fixed-T", default=None, help='extra setting with given temperatures: "T" or "choice,yesno,score"')
    ap.add_argument("--fixed-T-note", default=None, help="where the --fixed-T values come from, for the report")
    ap.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples over groups (0 = off)")
    ap.add_argument("--max-items", type=int, default=None, help="random subsample of N items, for quick runs")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--force", action="store_true", help="recompute cached logits")
    args = ap.parse_args(argv)

    tasks = load_tasks(args.revision)
    if args.max_items and args.max_items < len(tasks):
        keep = sorted(np.random.default_rng(args.seed).choice(len(tasks), args.max_items, replace=False))
        tasks = [tasks[i] for i in keep]
    tag = f"opendecisions_{args.prompt_format}_{args.letter_tokens}" + (f"_n{args.max_items}" if args.max_items else "")
    out_dir = Path(args.output_dir) / args.model.replace("/", "__") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "logits.npz"
    ids = np.array([t["id"] for t in tasks])
    if cache.exists() and not args.force and np.array_equal(np.load(cache)["id"], ids):
        print(f"Loading cached logits from {cache}")
        d = np.load(cache)
        z, mass = d["z"], d["letter_mass"]
    else:
        from .logits import extract_prompt_logits, load_model

        model, tok = load_model(args.model, dtype=args.dtype, attn_impl=args.attn_impl)
        prompts = [render_quyet_v1(t["state"], t["question"], tok) for t in tasks]
        ks = [len(t["labels"]) for t in tasks]
        z, mass = extract_prompt_logits(
            model, tok, prompts, ks, args.letter_tokens, args.max_batch_tokens, args.max_batch_size,
            desc="opendecisions", letter_mass=True,
        )
        np.savez(cache, id=ids, z=z, letter_mass=mass)

    y = np.array([gold_index(t) for t in tasks])
    types = np.array([t["question"]["type"] for t in tasks])
    # items that share a `group` (≤ 4 variants of one decision) stay in one fold; the 1,726
    # ungrouped items are their own unit
    groups = np.array([t["group"] or t["id"] for t in tasks])
    item_T, fit = temperatures(z, y, types, groups, args.min_T, args.max_T, args.n_folds, args.seed)
    if args.fixed_T:
        fixed = parse_fixed_T(args.fixed_T)
        item_T["fixed"] = np.array([fixed[t] for t in types])

    settings, all_preds = {}, {}
    for key, T in item_T.items():
        p = softmax(z, T)
        preds = [{"id": t["id"], "probs": to_probs(t, p[i])} for i, t in enumerate(tasks)]
        with open(out_dir / f"predictions_{key}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in preds)
        if key == "crossfit":
            label = "per fold, see below"
        else:
            Ts = {t: float(T[types == t][0]) for t in TYPES if (types == t).any()}
            label = " / ".join(f"{Ts[t]:.3f}" if t in Ts else "–" for t in TYPES)
        settings[key] = {"score": official_score(tasks, preds), "T_label": label}
        all_preds[key] = preds
    boot = paired_bootstrap(tasks, all_preds, groups, args.bootstrap, args.seed) if args.bootstrap else None

    result = {
        "config": vars(args),
        "n_items": len(tasks),
        "n_groups": int(len(set(groups.tolist()))),
        "fit": fit,
        "settings": settings,
        "bootstrap": boot,
        "letter_mass": {
            "mean": float(np.nanmean(mass)),
            "min": float(np.nanmin(mass)),
            "below_0.5": int((mass < 0.5).sum()),
            "mean_per_type": {t: float(np.nanmean(mass[types == t])) for t in TYPES if (types == t).any()},
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir / "report.md", result)

    print(f"\n{args.model} on OpenDecisions ({len(tasks)} items)   outputs in {out_dir}")
    print(f"{'setting':<46}{'Capability':>11}{'Intell.':>9}{'Calib.':>8}{'TVD':>7}   Δ vs T=1 (95% CI)")
    for key, s in settings.items():
        sc = s["score"]
        ci = "" if not boot or key == "t1" else "   [{:+.1f}, {:+.1f}]".format(*boot[key]["diff_vs_t1_ci"])
        print(
            f"{SETTINGS[key]:<46}{sc['capability']:>11.1f}{sc['intelligence']:>9.1f}"
            f"{sc['calibration']:>8.1f}{sc['calibration_items_tvd']:>7.3f}{ci}"
        )
    lm = result["letter_mass"]
    print(f"letter mass mean {lm['mean']:.3f} / min {lm['min']:.4f}")


if __name__ == "__main__":
    main()
