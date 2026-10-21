"""Aggregate every results/*/*/metrics.json into one markdown table.

    uv run python -m ts.summarize [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _T_label(r: dict, key: str) -> str:
    if key == "none":
        return "1"
    if key == "global":
        return f"{r['fit']['global']['T']:.3f}"
    return f"per-subject (λ={r['fit'][key]['lam']:g})"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows, od_rows = [], []
    for p in sorted(root.glob("*/*/metrics.json")):
        r = json.loads(p.read_text())
        c = r["config"]
        if p.parent.name.startswith("opendecisions_"):
            from .opendecisions import SETTINGS

            for key, st in r["settings"].items():
                sc = st["score"]
                ci = (r.get("bootstrap") or {}).get(key, {}).get("diff_vs_t1_ci")
                ci = "–" if key == "t1" or not ci else f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
                od_rows.append(
                    f"| {c['model']} | {p.parent.name} | {SETTINGS[key]} | {sc['capability']:.1f} | {sc['intelligence']:.1f} | "
                    f"{sc['calibration']:.1f} | {sc['calibration_items_tvd']:.3f} | {ci} |"
                )
            continue
        if "methods" not in r.get("test", {}):
            print(f"skipping {p} (old format; rerun ts.run on it)", file=sys.stderr)
            continue
        sets = [(f"{c['benchmark']} test", r["test"])] + [(f"{k} (test-only)", v) for k, v in r.get("extra", {}).items()]
        for eval_name, ev in sets:
            for key, t in ev["methods"].items():
                rows.append(
                    f"| {c['model']} | {p.parent.name} | {eval_name} | {key} | {_T_label(r, key)} | {t['n']} | {t['accuracy']:.4f} | "
                    f"{t['ece_15']:.4f} | {t['ece_10']:.4f} | {t['nll']:.4f} | {t['brier']:.4f} |"
                )
    out = [
        "## MCQ benchmarks\n",
        "| Model | Setting | Eval set | Method | T | n | Acc | ECE-15 | ECE-10 | NLL | Brier |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        *rows,
    ]
    if od_rows:
        out += [
            "\n## OpenDecisions\n",
            "| Model | Setting | Temperatures | Capability | Intelligence | Calibration | TVD | Δ vs T=1 (95% CI) |",
            "|---|---|---|---:|---:|---:|---:|---|",
            *od_rows,
        ]
    text = "\n".join(out)
    (root / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
