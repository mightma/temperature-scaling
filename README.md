# temperature-scaling

Temperature-scaling calibration for LLMs on multiple-choice benchmarks, using **one-letter readout**: one forward pass per question, then a softmax over the option-letter logits (A/B/C/D…) at the answer position. A single temperature T (or one T per subject) is fitted on a calibration split and evaluated on held-out test sets.

One command fits and evaluates any Hugging Face causal LM. The default is `Qwen/Qwen3.5-9B`.

```bash
uv run python -m ts.run --model Qwen/Qwen3.5-9B --benchmark mmlu
```

## Results (Qwen3.5-9B, zero-shot)

T is fitted only on the MMLU calibration split (validation + dev, 1,816 items). None of the test sets below is used for fitting. Accuracy is identical before and after calibration, because T > 0 doesn't change the argmax.

| Prompt | Method | MMLU test ECE / NLL | MMLU-Pro ECE / NLL | MMLU-Redux ECE / NLL |
|---|---|---|---|---|
| `plain` | uncalibrated | 0.018 / 0.571 | 0.035 / 1.456 | **0.019 / 0.466** |
| `plain` | global TS (T = 1.03) | 0.016 / 0.571 | 0.038 / 1.455 | 0.022 / 0.468 |
| `quyet-v1` | uncalibrated | 0.080 / 0.665 | 0.119 / 1.582 | 0.053 / 0.525 |
| `quyet-v1` | global TS (T = 1.54) | 0.010 / 0.610 | **0.013 / 1.465** | 0.031 / 0.511 |

ECE uses 15 equal-width bins. MMLU-Redux keeps only the items annotated `error_type == ok`.

- **`plain` doesn't need calibration.** The plain-text prompt is already well calibrated, and on clean labels (MMLU-Redux) temperature scaling doesn't help.
- **`quyet-v1` does, and one global T is enough.** The chat prompt is over-confident, and a single global T fixes it. That T transfers from 4-option MMLU to ~10-option MMLU-Pro.
- **Per-subject TS mostly fits label noise.** It beats global TS on the original MMLU test set. The advantage largely disappears on human-verified MMLU-Redux and on MMLU-Pro.

The full write-up, in Chinese, is in **[doc/report.md](doc/report.md)**. It covers the implementation, all metrics (ECE-10, Brier, high-confidence region, bootstrap CIs, per-subject analysis) and the conclusions.

## How it works

1. **Prompt.** The question and lettered options, ending where the model should emit the letter. Three formats are available:
   - `plain`: raw text ending in `Answer:`.
   - `chat`: chat template, assistant turn pre-filled with `Answer:`.
   - `quyet-v1`: the decision-model prompt in [doc/prompt-format.md](doc/prompt-format.md).
2. **Readout.** One forward pass with no generation.
   - The LM head rows for the option letters are applied in **fp32** to the last hidden state. bf16 logits are quantised to steps of 0.125 at typical magnitudes, which is too coarse for calibration.
   - `" A"` and `"A"` are different tokens, so their logits are merged with logsumexp (`--letter-tokens both`).
   - The softmax is taken over the K letters only, not the whole vocabulary.
3. **Fit T** on the calibration split by minimising NLL. A log-spaced grid over 0.05–10 is followed by LBFGS on log T.
4. **Per-subject TS (optional, on by default).** Fits one T per subject, shrunk toward the global T: `T_s = argmin NLL_s(T) + λ (log T − log T_global)²`.
   - λ is chosen by subject-stratified 5-fold CV on the calibration split.
   - Subjects with no calibration data fall back to the global T.
5. **Evaluate** on every test set:
   - ECE and MCE (15 and 10 bins), NLL, Brier, mean confidence.
   - A high-confidence-region check and per-subject tables.
   - Reliability diagrams.

## Installation

Requires a CUDA GPU. Qwen3.5-9B in bf16 needs ~18 GB; it was tested on a single 24 GB A10G.

```bash
uv sync
```

`flash-linear-attention` is installed as a dependency for Qwen3.5's linear-attention layers. Without it, transformers falls back to a much slower PyTorch path.

## Usage

```bash
# Fit + evaluate (MMLU calibration; MMLU test, MMLU-Pro and MMLU-Redux as test sets)
uv run python -m ts.run --model Qwen/Qwen3.5-9B --benchmark mmlu --prompt-format plain
uv run python -m ts.run --model Qwen/Qwen3.5-9B --benchmark mmlu --prompt-format quyet-v1

# Quick smoke run on a subsample
uv run python -m ts.run --max-test 300 --calib-size 300 --output-dir /tmp/ts_quick

# Several models / benchmarks, then a summary table at results/summary.md
MODELS="Qwen/Qwen3.5-9B meta-llama/Llama-3.1-8B-Instruct" BENCHMARKS="mmlu arc_challenge" scripts/run_all.sh
uv run python -m ts.summarize
```

A full run takes about 30 minutes per prompt format on an A10G: ~33k forward passes across the calibration split and three test sets. Logits are cached, so re-running to change fitting or evaluation takes seconds.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | Any HF causal LM (or image-text-to-text checkpoint with a text decoder) |
| `--benchmark` | `mmlu` | Calibration benchmark: `mmlu`, `mmlu_pro`, `arc_challenge` |
| `--extra-test` | `mmlu_pro mmlu_redux` when `--benchmark mmlu` | Test-only sets evaluated with the fitted temperatures; pass the flag with no values to disable |
| `--prompt-format` | `plain` | `plain`, `chat`, `quyet-v1` |
| `--letter-tokens` | `both` | Letter tokens to read: `both` (merge `" A"` and `"A"`), `space`, `nospace` |
| `--no-subject-ts` | off | Skip per-subject TS |
| `--n-bins` | `15` | Bins for reliability diagrams and per-subject ECE (overall ECE is always reported at 15 and 10) |
| `--calib-size`, `--max-test` | all | Subsample the calibration / test sets |
| `--max-batch-tokens`, `--max-batch-size` | `16384`, `32` | Batching limits (batches are length-sorted and left-padded) |
| `--dtype`, `--attn-impl` | `bfloat16`, auto | Model loading |
| `--output-dir` | `results` | Output root |
| `--force` | off | Recompute cached logits |

### Datasets

| Set | Role | Source | Items |
|---|---|---|---:|
| MMLU validation + dev | calibration | `cais/mmlu` | 1,816 |
| MMLU test | test | `cais/mmlu` | 14,042 |
| MMLU-Pro test | test only | `TIGER-Lab/MMLU-Pro`. Items with `src = ori_mmlu-<subject>` (6,810) use that subject's T; the rest use the global T | 12,032 |
| MMLU-Redux 2.0 | test only | `edinburgh-dawg/mmlu-redux-2.0`, items with `error_type == ok` | 5,330 |

With `--benchmark mmlu_pro`, the calibration split is 2,000 items carved from MMLU-Pro with a fixed seed. With `--benchmark arc_challenge`, it is ARC-Challenge train + validation (1,418 items), and the test split is ARC-Challenge test (1,172 items).

### Outputs

Each run writes to `results/<model>/<benchmark>_<prompt-format>_<letter-tokens>/`:

| File | Contents |
|---|---|
| `report.md` | Fitted temperatures, metric tables for every set and method, high-confidence region, per-subject tables |
| `metrics.json` | Everything in the report, machine-readable |
| `reliability_test.png`, `reliability_<set>.png` | Reliability diagrams: uncalibrated / global TS / per-subject TS |
| `nll_vs_T.png` | Calibration-set NLL as a function of T |
| `logits_*.npz` | Cached option logits (git-ignored) |

### Using other models

The pipeline only needs `get_decoder()` and `get_output_embeddings()`, so most HF causal LMs work unchanged. Requirements:

- Every option letter (`"A"`, `" A"`, …) must be a single token. The run fails with a clear error otherwise; use `--letter-tokens space` or `nospace` if only one form is single-token.
- `chat` and `quyet-v1` need a tokenizer with a chat template. `enable_thinking=False` is passed to the template; templates that don't use it ignore it.
- `plain` works for both base and instruct models.

## Project layout

```
ts/
  data.py         benchmark loaders and calibration / test splits
  prompts.py      plain / chat / quyet-v1 prompt rendering
  logits.py       model loading and batched letter-logit extraction (fp32 head)
  calibration.py  temperature fitting, per-subject shrinkage + CV, ECE / NLL / Brier / MCE
  plots.py        reliability diagrams, NLL-vs-T curve
  run.py          end-to-end CLI
  summarize.py    aggregate results/*/*/metrics.json into results/summary.md
scripts/run_all.sh  run every (model, benchmark) pair
tests/              unit tests (fitting, metrics, shrinkage, quyet-v1 prompt parity)
doc/                report and prompt-format spec
results/            run outputs (metrics, reports, figures)
```

## Tests

```bash
uv run pytest -q tests
```

The quyet-v1 prompt tests download the Qwen3.5-9B tokenizer and check the rendered prompts against the examples in [doc/prompt-format.md](doc/prompt-format.md), character for character and token for token.

## Documentation

- [doc/report.md](doc/report.md): experiment report covering implementation, results and conclusions (Chinese).
- [doc/prompt-format.md](doc/prompt-format.md): the quyet-v1 prompt specification.

## References

- Guo et al., 2017. *On Calibration of Modern Neural Networks*. arXiv:1706.04599
- Kadavath et al., 2022. *Language Models (Mostly) Know What They Know*. arXiv:2207.05221
- Tian et al., 2023. *Just Ask for Calibration*. arXiv:2305.14975
- Wang et al., 2024. *MMLU-Pro*. arXiv:2406.01574
- Gema et al., 2024. *Are We Done with MMLU?* (MMLU-Redux). arXiv:2406.04127
