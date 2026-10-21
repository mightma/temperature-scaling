# OpenDecisions: `Qwen/Qwen3.5-9B` (`quyet-v1` prompt)

- 2352 items, dataset revision `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe`; scored with the official `score.py` (ts/vendor/opendecisions_score.py)
- One temperature per question type; fitted temperatures are NLL-optimal subject to T ≥ 1
- Cross-fitting: 5 folds of whole `group`s; an ungrouped item is its own group (2019 units)

## Capability

| Setting | Capability | Intelligence | Calibration | TVD (129 items) | T choice / yes-no / score |
|---|---:|---:|---:|---:|---|
| T = 1 (zero-shot) | **61.1** | 48.5 | 73.7 | 0.262 | 1.000 / 1.000 / 1.000 |
| Per-type T, cross-fitted by group | **66.9** | 47.5 | 86.3 | 0.234 | per fold, see below |
| Per-type T, fitted on all items (optimistic) | **67.0** | 47.5 | 86.4 | 0.234 | 1.666 / 1.796 / 1.717 |
| Fixed T (--fixed-T) | **65.8** | 47.8 | 83.9 | 0.236 | 1.540 / 1.540 / 1.540 |

Fixed T source: global T fitted on the MMLU calibration split (validation + dev) with the quyet-v1 prompt; no OpenDecisions data.

Paired bootstrap over groups (1000 resamples; an ungrouped item is its own group), 95% intervals:

| Setting | Capability CI | Capability − T=1 CI |
|---|---|---|
| T = 1 (zero-shot) | [57.0, 64.2] | – |
| Per-type T, cross-fitted by group | [62.6, 68.5] | [+3.7, +6.5] |
| Per-type T, fitted on all items (optimistic) | [62.8, 68.5] | [+3.8, +6.5] |
| Fixed T (--fixed-T) | [61.7, 68.0] | [+3.4, +5.6] |

## Per question type

| Setting | Type | Accuracy % | Competence | Calibration |
|---|---|---:|---:|---:|
| T = 1 (zero-shot) | choice | 71.5 | 62.3 | 59.0 |
| T = 1 (zero-shot) | yes/no | 76.1 | 47.7 | 78.7 |
| T = 1 (zero-shot) | score | 60.4 | 35.5 | 83.5 |
| Per-type T, cross-fitted by group | choice | 71.5 | 62.3 | 83.1 |
| Per-type T, cross-fitted by group | yes/no | 76.1 | 47.7 | 85.7 |
| Per-type T, cross-fitted by group | score | 60.4 | 32.7 | 90.2 |
| Per-type T, fitted on all items (optimistic) | choice | 71.5 | 62.3 | 83.1 |
| Per-type T, fitted on all items (optimistic) | yes/no | 76.1 | 47.7 | 85.7 |
| Per-type T, fitted on all items (optimistic) | score | 60.4 | 32.6 | 90.5 |
| Fixed T (--fixed-T) | choice | 71.5 | 62.3 | 79.3 |
| Fixed T (--fixed-T) | yes/no | 76.1 | 47.7 | 84.4 |
| Fixed T (--fixed-T) | score | 60.4 | 33.5 | 87.8 |

## Letter mass

Share of the full-vocabulary next-token probability on the item's option letters (both `" X"` and `"X"`): mean **0.997**, min 0.028, items below 0.5: 1.

## Cross-fitted temperatures per fold

| Fold | choice | yes/no | score |
|---|---:|---:|---:|
| 0 | 1.691 | 1.641 | 1.679 |
| 1 | 1.679 | 1.691 | 1.769 |
| 2 | 1.654 | 1.810 | 1.704 |
| 3 | 1.654 | 1.865 | 1.666 |
| 4 | 1.691 | 1.951 | 1.769 |
