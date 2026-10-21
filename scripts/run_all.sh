#!/usr/bin/env bash
# Calibrate + evaluate every (model, benchmark) pair, then write results/summary.md.
# Usage: MODELS="Qwen/Qwen3.5-9B meta-llama/Llama-3.1-8B-Instruct" BENCHMARKS="mmlu arc_challenge" scripts/run_all.sh [extra ts.run args]
set -euo pipefail
MODELS=${MODELS:-"Qwen/Qwen3.5-9B"}
BENCHMARKS=${BENCHMARKS:-"mmlu"}
for m in $MODELS; do
  for b in $BENCHMARKS; do
    uv run python -m ts.run --model "$m" --benchmark "$b" "$@"
  done
done
uv run python -m ts.summarize
