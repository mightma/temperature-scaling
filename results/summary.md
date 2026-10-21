## MCQ benchmarks

| Model | Setting | Eval set | Method | T | n | Acc | ECE-15 | ECE-10 | NLL | Brier |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu test | none | 1 | 14042 | 0.7841 | 0.0179 | 0.0172 | 0.5711 | 0.2962 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu test | global | 1.026 | 14042 | 0.7841 | 0.0162 | 0.0161 | 0.5709 | 0.2959 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu test | subject_indep | per-subject (λ=0) | 14042 | 0.7841 | 0.0168 | 0.0127 | 0.6651 | 0.2970 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu test | subject_shrunk | per-subject (λ=10) | 14042 | 0.7841 | 0.0161 | 0.0133 | 0.5635 | 0.2937 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_pro (test-only) | none | 1 | 12032 | 0.5104 | 0.0348 | 0.0336 | 1.4556 | 0.6091 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_pro (test-only) | global | 1.026 | 12032 | 0.5104 | 0.0381 | 0.0359 | 1.4554 | 0.6093 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_pro (test-only) | subject_indep | per-subject (λ=0) | 12032 | 0.5104 | 0.0269 | 0.0285 | 1.6812 | 0.6176 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_pro (test-only) | subject_shrunk | per-subject (λ=10) | 12032 | 0.5104 | 0.0317 | 0.0310 | 1.4554 | 0.6096 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_redux (test-only) | none | 1 | 5330 | 0.8229 | 0.0193 | 0.0187 | 0.4662 | 0.2420 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_redux (test-only) | global | 1.026 | 5330 | 0.8229 | 0.0224 | 0.0217 | 0.4677 | 0.2422 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_redux (test-only) | subject_indep | per-subject (λ=0) | 5330 | 0.8229 | 0.0148 | 0.0101 | 0.6273 | 0.2495 |
| Qwen/Qwen3.5-9B | mmlu_plain_both | mmlu_redux (test-only) | subject_shrunk | per-subject (λ=10) | 5330 | 0.8229 | 0.0207 | 0.0175 | 0.4668 | 0.2422 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu test | none | 1 | 14042 | 0.7682 | 0.0796 | 0.0796 | 0.6653 | 0.3270 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu test | global | 1.540 | 14042 | 0.7682 | 0.0103 | 0.0065 | 0.6101 | 0.3150 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu test | subject_indep | per-subject (λ=0) | 14042 | 0.7682 | 0.0145 | 0.0134 | 0.7720 | 0.3160 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu test | subject_shrunk | per-subject (λ=10) | 14042 | 0.7682 | 0.0082 | 0.0088 | 0.5999 | 0.3118 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_pro (test-only) | none | 1 | 12032 | 0.5085 | 0.1191 | 0.1191 | 1.5821 | 0.6293 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_pro (test-only) | global | 1.540 | 12032 | 0.5085 | 0.0133 | 0.0108 | 1.4647 | 0.6079 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_pro (test-only) | subject_indep | per-subject (λ=0) | 12032 | 0.5085 | 0.0274 | 0.0274 | 1.9080 | 0.6181 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_pro (test-only) | subject_shrunk | per-subject (λ=10) | 12032 | 0.5085 | 0.0137 | 0.0125 | 1.4666 | 0.6084 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_redux (test-only) | none | 1 | 5330 | 0.8015 | 0.0530 | 0.0534 | 0.5252 | 0.2683 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_redux (test-only) | global | 1.540 | 5330 | 0.8015 | 0.0306 | 0.0323 | 0.5108 | 0.2640 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_redux (test-only) | subject_indep | per-subject (λ=0) | 5330 | 0.8015 | 0.0246 | 0.0257 | 0.7634 | 0.2712 |
| Qwen/Qwen3.5-9B | mmlu_quyet-v1_both | mmlu_redux (test-only) | subject_shrunk | per-subject (λ=10) | 5330 | 0.8015 | 0.0273 | 0.0272 | 0.5084 | 0.2633 |

## OpenDecisions

| Model | Setting | Temperatures | Capability | Intelligence | Calibration | TVD | Δ vs T=1 (95% CI) |
|---|---|---|---:|---:|---:|---:|---|
| Qwen/Qwen3.5-9B | opendecisions_quyet-v1_both | T = 1 (zero-shot) | 61.1 | 48.5 | 73.7 | 0.262 | – |
| Qwen/Qwen3.5-9B | opendecisions_quyet-v1_both | Per-type T, cross-fitted by group | 66.9 | 47.5 | 86.3 | 0.234 | [+3.7, +6.5] |
| Qwen/Qwen3.5-9B | opendecisions_quyet-v1_both | Per-type T, fitted on all items (optimistic) | 67.0 | 47.5 | 86.4 | 0.234 | [+3.8, +6.5] |
| Qwen/Qwen3.5-9B | opendecisions_quyet-v1_both | Fixed T (--fixed-T) | 65.8 | 47.8 | 83.9 | 0.236 | [+3.4, +5.6] |
