"""One-command pipeline: extract letter logits -> fit T on calibration split -> evaluate on test.

    uv run python -m ts.run --model Qwen/Qwen3.5-9B --benchmark mmlu

With ``--benchmark mmlu`` the fitted temperatures are also evaluated (never fitted) on
MMLU-Pro and MMLU-Redux; see ``--extra-test``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .calibration import fit_group_temperatures, fit_temperature, item_temperatures, metrics, select_shrinkage, softmax
from .data import BENCHMARKS, EXTRA_TESTS, load_benchmark, load_extra_test
from .prompts import PROMPT_FORMATS

METRIC_ROWS = [
    ("accuracy", "Accuracy", "↑"),
    ("ece_15", "ECE (15 bins)", "↓"),
    ("ece_10", "ECE (10 bins)", "↓"),
    ("mce_15", "MCE (15 bins)", "↓"),
    ("nll", "NLL", "↓"),
    ("brier", "Brier", "↓"),
    ("mean_confidence", "Mean confidence", ""),
]

METHOD_NAMES = {
    "none": "Uncalibrated (T=1)",
    "global": "Global TS",
    "subject_indep": "Per-subject TS (independent)",
    "subject_shrunk": "Per-subject TS (shrunk)",
}

EXTRA_NAMES = {"mmlu_pro": "MMLU-Pro (test)", "mmlu_redux": "MMLU-Redux 2.0 (error_type = ok)"}


def get_logits(args, splits: dict[str, list[dict]], out_dir: Path) -> dict[str, dict]:
    """Option logits for every split, cached per split in ``logits_<split>.npz``."""
    paths = {s: out_dir / f"logits_{s}.npz" for s in splits}
    out = {s: dict(np.load(p, allow_pickle=False)) for s, p in paths.items() if p.exists() and not args.force}
    if out:
        print(f"Loading cached logits for {sorted(out)} from {out_dir}")
    missing = [s for s in splits if s not in out]
    if not missing:
        return out

    from .logits import extract_logits, load_model

    model, tok = load_model(args.model, dtype=args.dtype, attn_impl=args.attn_impl)
    for s in missing:
        items = splits[s]
        t0 = time.time()
        z = extract_logits(
            model, tok, items, args.prompt_format, args.letter_tokens, args.max_batch_tokens, args.max_batch_size, desc=s
        )
        out[s] = {
            "z": z,
            "y": np.array([it["answer"] for it in items]),
            "subject": np.array([it["subject"] for it in items]),
            "group": np.array([it.get("group", it["subject"]) for it in items]),
        }
        np.savez(paths[s], **out[s])
        print(f"{s}: {len(items)} items in {time.time() - t0:.0f}s")
    return out


def fit_methods(zc, yc, sc, group_ts: bool, seed: int) -> dict:
    """Fit every calibration method on the calibration split only."""
    fit = fit_temperature(zc, yc)
    details = {"global": fit}
    if group_ts and len(np.unique(sc)) > 1:
        sel = select_shrinkage(zc, yc, sc, seed=seed)
        for key, lam in (("subject_indep", 0.0), ("subject_shrunk", sel["lam"])):
            details[key] = {"lam": lam, "T": fit_group_temperatures(zc, yc, sc, fit["T"], lam)}
        details["shrinkage_cv"] = sel
    return details


def apply_methods(subjects: np.ndarray, details: dict) -> dict[str, np.ndarray]:
    """Per-item temperature for each method. Subjects without a fitted T use the global T."""
    T = details["global"]["T"]
    temps = {"none": np.ones(len(subjects)), "global": np.full(len(subjects), T)}
    for key in ("subject_indep", "subject_shrunk"):
        if key in details:
            temps[key] = item_temperatures(subjects, details[key]["T"], T)
    return temps


def subset_table(z, y, groups, subjects, temps, calib_subjects, n_bins):
    rows = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        subj_in_group = set(subjects[m].tolist())
        row = {
            "subject": g,
            "n": int(m.sum()),
            "n_calib": int(np.isin(calib_subjects, list(subj_in_group)).sum()),
        }
        for key, tt in temps.items():
            mt = metrics(z[m], y[m], tt[m], (n_bins,))
            row["accuracy"] = mt["accuracy"]
            row[f"ece_{key}"] = mt[f"ece_{n_bins}"]
            row[f"gap_{key}"] = mt["mean_confidence"] - mt["accuracy"]
            if key.startswith("subject_"):
                row[f"T_{key}"] = float(tt[m].mean())  # mean per-item T when a group mixes subjects
        rows.append(row)
    return rows


def high_conf_check(z, y, temps, thresholds=(0.8, 0.9, 0.95)):
    out = []
    for key, tt in temps.items():
        p = softmax(z, tt)
        conf, correct = p.max(1), p.argmax(1) == y
        for th in thresholds:
            m = conf >= th
            out.append(
                {
                    "method": key,
                    "threshold": th,
                    "share": float(m.mean()),
                    "accuracy": float(correct[m].mean()) if m.any() else None,
                    "mean_conf": float(conf[m].mean()) if m.any() else None,
                }
            )
    return out


def evaluate_set(d: dict, details: dict, calib_subjects: np.ndarray, n_bins: int) -> tuple[dict, dict]:
    z, y, s = d["z"], d["y"], d["subject"]
    g = d.get("group", s)
    temps = apply_methods(s, details)
    pred0 = z.argmax(1)
    ev = {
        "methods": {k: metrics(z, y, tt) for k, tt in temps.items()},
        "predictions_unchanged": bool(all((softmax(z, tt).argmax(1) == pred0).all() for tt in temps.values())),
        "share_with_subject_T": float(np.isin(s, calib_subjects).mean()),
        "high_conf": high_conf_check(z, y, temps),
        "subsets": subset_table(z, y, g, s, temps, calib_subjects, n_bins),
    }
    return ev, temps


# ---------------------------------------------------------------------------- report


def _metric_table(L, by_method):
    keys = list(by_method)
    L.append("| Metric | " + " | ".join(METHOD_NAMES[k] for k in keys) + " |")
    L.append("|---|" + "---:|" * len(keys))
    for key, name, arrow in METRIC_ROWS:
        vals = [by_method[k][key] for k in keys]
        best = min(vals) if arrow == "↓" else None
        cells = [f"**{v:.4f}**" if best is not None and v == best else f"{v:.4f}" for v in vals]
        L.append(f"| {name} {arrow} | " + " | ".join(cells) + " |")
    L.append("")


def _eval_section(L, ev: dict, n_bins: int, level: str, figure: str, subset_label: str = "Subset"):
    methods = ev["methods"]
    _metric_table(L, methods)
    L.append(
        f"Predictions identical to uncalibrated for every method: **{ev['predictions_unchanged']}** "
        f"(T>0 does not change the argmax, so accuracy is unchanged by construction).\n"
    )
    L.append(f"{level}# High-confidence region\n")
    L.append("| Method | conf ≥ | share of items | accuracy | mean conf | conf − acc |\n|---|---:|---:|---:|---:|---:|")
    for h in ev["high_conf"]:
        if h["accuracy"] is None:
            L.append(f"| {METHOD_NAMES[h['method']]} | {h['threshold']} | 0 | – | – | – |")
        else:
            L.append(
                f"| {METHOD_NAMES[h['method']]} | {h['threshold']} | {h['share']:.3f} | {h['accuracy']:.4f} | {h['mean_conf']:.4f} | {h['mean_conf'] - h['accuracy']:+.4f} |"
            )
    L.append("")
    rows = ev["subsets"]
    if len(rows) > 1:
        last = list(methods)[-1]
        L.append(f"{level}# Per-{subset_label.lower()} calibration (sorted by ECE{n_bins} under {METHOD_NAMES[last]})\n")
        L.append(
            "`gap` = mean confidence − accuracy (positive: over-confident). `n_cal` = calibration items available "
            "to fit the T used here. `T` = mean per-item temperature. Per-subset ECE on ~100 items is itself noisy "
            "(small-sample ECE is biased upward).\n"
        )
        has_subj = "subject_shrunk" in methods
        head = f"| {subset_label} | n | n_cal | acc | ECE none | ECE global |"
        sep = "|---|---:|---:|---:|---:|---:|"
        if has_subj:
            head += " ECE indep | ECE shrunk | T indep | T shrunk |"
            sep += "---:|---:|---:|---:|"
        head += " gap none | gap global |" + (" gap shrunk |" if has_subj else "")
        sep += "---:|---:|" + ("---:|" if has_subj else "")
        L += [head, sep]
        for s in sorted(rows, key=lambda x: -x[f"ece_{last}"]):
            line = f"| {s['subject']} | {s['n']} | {s['n_calib']} | {s['accuracy']:.3f} | {s['ece_none']:.4f} | {s['ece_global']:.4f} |"
            if has_subj:
                line += f" {s['ece_subject_indep']:.4f} | {s['ece_subject_shrunk']:.4f} | {s['T_subject_indep']:.3f} | {s['T_subject_shrunk']:.3f} |"
            line += f" {s['gap_none']:+.4f} | {s['gap_global']:+.4f} |" + (f" {s['gap_subject_shrunk']:+.4f} |" if has_subj else "")
            L.append(line)
        L.append("")
    L.append(f"![reliability]({figure})\n")


def write_report(path: Path, r: dict):
    L = []
    c = r["config"]
    nb = c["n_bins"]
    L.append(f"# Temperature scaling report: `{c['model']}` on `{c['benchmark']}`\n")
    L.append(
        f"- Prompt format: `{c['prompt_format']}`, letter tokens: `{c['letter_tokens']}` "
        f"(logits of ' X' / 'X' merged with logsumexp when `both`)\n"
        f"- Softmax over the K option letters only (not the full vocabulary)\n"
        f"- Calibration split (fits every T): **{r['calib']['methods']['none']['n']}** items; "
        f"held-out test split: **{r['test']['methods']['none']['n']}** items"
        + "".join(f"; test-only `{k}`: **{v['methods']['none']['n']}** items" for k, v in r["extra"].items())
        + "\n- ECE/MCE use equal-width confidence bins; both 15 and 10 bins reported\n"
    )
    f = r["fit"]["global"]
    L.append(f"## Fitted temperature\n\n**Global T = {f['T']:.4f}** (grid search 0.05–10: {f['T_grid']:.4f}; LBFGS on log T: {f['T_lbfgs']:.4f})\n")
    L.append("T > 1 means the raw model was over-confident; T < 1 means under-confident.\n")
    if "subject_shrunk" in r["fit"]:
        cv = r["fit"]["shrinkage_cv"]
        L.append(
            "Per-subject TS fits one T per subject on the calibration split, shrunk toward the global T:\n"
            "`T_s = argmin NLL_s(T) + λ·(log T − log T_global)²`. λ=0 is fully independent, λ=∞ is global TS. "
            f"λ chosen by subject-stratified 5-fold CV on the calibration split: **λ = {cv['lam']:g}**.\n"
        )
        L.append("| λ | " + " | ".join(cv["cv_nll"]) + " |\n|---|" + "---:|" * len(cv["cv_nll"]))
        L.append("| CV NLL | " + " | ".join(f"{v:.4f}" for v in cv["cv_nll"].values()) + " |\n")
    L.append("## Test split\n")
    _eval_section(L, r["test"], nb, "##", "reliability_test.png")
    for name, ev in r["extra"].items():
        L.append(f"## Test-only: {EXTRA_NAMES.get(name, name)}\n")
        L.append(
            "Temperatures are the ones fitted on the calibration split above; nothing is fitted on this set. "
            f"{ev['share_with_subject_T'] * 100:.1f}% of items have a subject with a fitted per-subject T; the rest use the global T.\n"
        )
        _eval_section(L, ev, nb, "##", f"reliability_{name}.png", "Category" if name == "mmlu_pro" else "Subset")
    L.append("## Calibration split metrics (in-sample for every fitted method)\n")
    _metric_table(L, r["calib"]["methods"])
    L.append("## NLL vs T on the calibration split\n\n![nll](nll_vs_T.png)\n")
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--benchmark", default="mmlu", choices=sorted(BENCHMARKS))
    ap.add_argument(
        "--extra-test",
        nargs="*",
        default=None,
        choices=sorted(EXTRA_TESTS),
        help="test-only sets evaluated with the fitted temperatures (default: mmlu_pro mmlu_redux when --benchmark mmlu)",
    )
    ap.add_argument("--prompt-format", default="plain", choices=list(PROMPT_FORMATS))
    ap.add_argument("--letter-tokens", default="both", choices=["both", "space", "nospace"])
    ap.add_argument("--calib-size", type=int, default=None, help="subsample calibration split (default: all)")
    ap.add_argument("--max-test", type=int, default=None, help="subsample every test set, for quick runs")
    ap.add_argument("--n-bins", type=int, default=15, help="bins for reliability diagram / per-subset ECE")
    ap.add_argument("--no-subject-ts", action="store_true", help="skip per-subject temperature scaling")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default=None, help="e.g. sdpa, flash_attention_2")
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--force", action="store_true", help="recompute logits even if cached")
    args = ap.parse_args(argv)
    if args.extra_test is None:
        # mmlu_pro as an extra set only makes sense when it isn't the calibration benchmark
        args.extra_test = ["mmlu_pro", "mmlu_redux"] if args.benchmark == "mmlu" else []

    tag = f"{args.benchmark}_{args.prompt_format}_{args.letter_tokens}"
    if args.max_test:
        tag += f"_test{args.max_test}"
    if args.calib_size:
        tag += f"_calib{args.calib_size}"
    out_dir = Path(args.output_dir) / args.model.replace("/", "__") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_benchmark(args.benchmark, seed=args.seed, calib_size=args.calib_size, max_test=args.max_test)
    for name in args.extra_test:
        splits[f"extra_{name}"] = load_extra_test(name, args.max_test, args.seed)
    data = get_logits(args, splits, out_dir)
    zc, yc, sc = data["calib"]["z"], data["calib"]["y"], data["calib"]["subject"]

    # Only the calibration split is used to fit temperatures (and to choose λ).
    details = fit_methods(zc, yc, sc, not args.no_subject_ts, args.seed)
    calib_temps = apply_methods(sc, details)
    test_ev, test_temps = evaluate_set(data["test"], details, sc, args.n_bins)
    extra, extra_temps = {}, {}
    for name in args.extra_test:
        extra[name], extra_temps[name] = evaluate_set(data[f"extra_{name}"], details, sc, args.n_bins)
    result = {
        "config": {**vars(args), "n_bins": args.n_bins},
        "fit": details,
        "calib": {"methods": {k: metrics(zc, yc, tt) for k, tt in calib_temps.items()}},
        "test": test_ev,
        "extra": extra,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    from .plots import nll_curve, reliability_diagram

    def plot(d, temps, path, label):
        panels = [(METHOD_NAMES[k], temps[k]) for k in ("none", "global", "subject_shrunk") if k in temps]
        reliability_diagram(d["z"], d["y"], panels, path, args.n_bins, f"{args.model} · {label} (n={len(d['y'])})")

    plot(data["test"], test_temps, out_dir / "reliability_test.png", f"{args.benchmark} test")
    for name in args.extra_test:
        plot(data[f"extra_{name}"], extra_temps[name], out_dir / f"reliability_{name}.png", EXTRA_NAMES.get(name, name))
    nll_curve(details["global"], out_dir / "nll_vs_T.png", f"{args.model} · {args.benchmark} calibration-set NLL")
    write_report(out_dir / "report.md", result)

    print(f"\nGlobal T = {details['global']['T']:.4f}   (outputs in {out_dir})")
    if "shrinkage_cv" in details:
        print(f"Per-subject shrinkage λ = {details['shrinkage_cv']['lam']:g}")
    for label, ev in [(f"{args.benchmark} test", test_ev)] + [(EXTRA_NAMES.get(k, k), v) for k, v in extra.items()]:
        ms = ev["methods"]
        print(f"\n[{label}]  n = {ms['none']['n']}")
        print(f"{'metric':<18}" + "".join(f"{k:>16}" for k in ms))
        for key, nm, _ in METRIC_ROWS:
            print(f"{nm:<18}" + "".join(f"{ms[k][key]:>16.4f}" for k in ms))


if __name__ == "__main__":
    main()
