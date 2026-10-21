# Qwen3.5-9B 在 MCQ 和 OpenDecisions 上的温度缩放校准：实验报告

日期：2026-10-21 · 模型：`Qwen/Qwen3.5-9B`（bf16，单张 A10G 24 GB）· 代码：`ts/` · 原始结果：`results/`

## 摘要

我们用 one-letter readout（在 "Answer:" 之后读取 A/B/C/D… 的 next-token 概率）作为 MCQ 的置信度，在 MMLU 校准集
（1,816 条）上拟合温度，并在三个不参与拟合的测试集上评估：MMLU test（14,042）、MMLU-Pro（12,032）、MMLU-Redux 2.0（5,330）。
此外，我们还在 OpenDecisions（2,352 道决策题）上用官方评分器计算了 Capability（见 2.6 节）。

1. **结果取决于 prompt。**
   - 纯文本 prompt（`plain`）不校准就已经很好：最优 T = 1.03，干净标签上 ECE 1.9%，温度缩放没有带来收益。
   - [prompt 规格文档](prompt-format.md)规定的 `quyet-v1` chat prompt 明显过度自信（ECE 8.0%）。单一的全局 T = 1.54 就能修正，
     而且这个 T 能迁移到 MMLU-Pro：虽然选项从 4 个变成约 10 个，ECE 仍从 11.9% 降到 1.3%。
2. **按学科 TS 的收益大部分来自拟合标注噪声。**
   - 在 MMLU test 上它显著优于全局 TS，但到了经人工复核的 MMLU-Redux 和 MMLU-Pro 上，这个优势基本消失。
   - 典型例子是 virology：原版 MMLU 上看起来过度自信 +29 个百分点，在复核过的干净题目上实际只有 +1.8 个百分点。
3. **建议：** 默认用全局 TS。评估校准时以 MMLU-Redux 为准，而不是原版 MMLU test。
4. **OpenDecisions Capability（quyet-v1）：**
   - T=1 时为 61.1，复现了参考值 60.6。
   - 直接用 MMLU 拟合的 T=1.54、不碰任何 OpenDecisions 数据，提升到 65.8（+4.7，95% CI [+3.4, +5.6]）。
   - 每题型 T 按 group 交叉拟合，提升到 66.9（+5.8，CI [+3.7, +6.5]）。
   - 提升全部来自 Calibration（73.7 → 86.3），Intelligence 反而略降 1 分。

---

## 1. 实现

### 1.1 流程

`uv run python -m ts.run --model <HF 模型> --benchmark mmlu [--prompt-format plain|chat|quyet-v1]` 一条命令完成以下步骤：

1. 加载数据，划分校准集和测试集（`ts/data.py`）。
2. 每题做一次前向（不生成），读取 K 个选项字母的 logits（`ts/logits.py`）。结果按集合缓存为 `logits_*.npz`，重复运行时不再前向。
3. 只用校准集拟合温度（`ts/calibration.py`）。
4. 在各测试集上计算指标，画 reliability diagram，并输出 `report.md` / `metrics.json`（`ts/run.py`、`ts/plots.py`）。

OpenDecisions 用单独的入口 `uv run python -m ts.opendecisions --model <HF 模型>`（`ts/opendecisions.py`），流程是：
用 quyet-v1 渲染每道题 → 一次前向读取字母 logits → 按几种温度设置（T=1、按 group 交叉拟合、`--fixed-T` 指定）生成预测 →
用原样复制进 `ts/vendor/` 的官方 `score.py` 打分，详见 2.6 节。

`scripts/run_all.sh` 可以批量跑多个模型和 benchmark，`ts.summarize` 生成汇总表 `results/summary.md`。代码中不依赖具体模型的
类名或结构（模型通过 `get_decoder()` / `get_output_embeddings()` 通用访问），换模型只需要改 `--model`。

### 1.2 数据

| 集合 | 用途 | 来源 | 条数 |
|---|---|---|---:|
| 校准集 | 拟合 T（以及选择 λ） | MMLU 官方 validation（1,531）+ dev（285） | 1,816 |
| MMLU test | 只用于测试 | MMLU 官方 test | 14,042 |
| MMLU-Pro | 只用于测试 | `TIGER-Lab/MMLU-Pro` 完整 test split，大多为 10 个选项 | 12,032 |
| MMLU-Redux 2.0 | 只用于测试 | `edinburgh-dawg/mmlu-redux-2.0` 中 `error_type == ok` 的题 | 5,330 |

- 所有设置都是 zero-shot，所以 dev 不需要留作 few-shot 示例，可以并入校准集。
- MMLU-Redux 是 MMLU test 中每科 100 题的人工复核子集。只保留标签被确认无误的题，用来区分"模型校准差"和"标签错了"这两种情况。
- 三个测试集与校准集都没有重叠。

### 1.3 Prompt

| 格式 | 结构 | 读取位置 |
|---|---|---|
| `plain` | "The following is a multiple choice question about {subject}." + 题干 + "A. … D. …" + "Answer:"（纯文本，不套 chat template） | "Answer:" 之后 |
| `quyet-v1` | [prompt-format.md](prompt-format.md) 的格式：system message；user 中依次为 State / Question / type hint / Options / "Answer with one letter."；chat template 并关闭 thinking | assistant 回复开头（`</think>\n\n` 之后） |
| `chat` | 题干放进 user turn，assistant 预填 "Answer:"（本报告未评估） | "Answer:" 之后 |

- **quyet-v1 的实现方式：** 按 [prompt-format.md](prompt-format.md) 的规格实现，支持 choice / score / noul 三种题型、JSON state（缩进 1 格），以及 6,000 / 8,000 token 的截断规则。
- **quyet-v1 的验证：** `tests/test_prompts.py` 检查它与附录中 3 个渲染示例逐字一致，token 数也一致（226 / 209 / 194）。
  附录提到的参考实现 `decision_model/prompt.py` 在本地找不到，所以是按文字规格加这 3 个例子校验的。
- **MCQ 到 quyet-v1 的映射（附录未规定，是我们自己的选择）：** 题干作为 state，type 为 `choice`，
  instructions 为 "Which option is the correct answer to this {subject} question?"，各选项文字作为 description，保持原顺序。

### 1.4 字母 logits 的读取

- **只在 K 个选项字母上做 softmax，而不是整个词表。** 两种 prompt 下字母的概率占比（MMLU 均匀抽 50 题）差别很大：
  plain 只有 0.21（base 风格的续写会把概率分给其他 token），quyet-v1 达到 0.999。
- **带空格和不带空格的字母合并。** `" A"` 和 `"A"` 是不同的 token（`"Answer:A"` 甚至会合并成一个 `":A"` token）。
  默认对两种写法的 logits 取 logsumexp，等价于把概率相加。这一步是必要的：plain 下概率几乎全在 `" X"` 上，
  而 quyet-v1 下 `"X"` 约占 0.88、`" X"` 约占 0.12。
- **用 fp32 计算字母对应的输出层。** bf16 输出的 logits 在 16–32 这个量级的分辨率只有 0.125，对校准来说太粗。
  所以取最后位置的隐状态，只用 fp32 计算字母对应的那几行 LM head（同时处理了 bias、logit scale 和 softcapping）。
  这样得到的 logits 与整模型 bf16 输出的差异在舍入误差以内。
- **批处理：** 按长度排序后做 left padding。逐条前向与批量前向的差异与 padding 长度无关（相关系数 0.07），
  不加 padding 只改 batch 时差异大小也一样，属于 bf16 kernel 的数值噪声，概率最大相差 0.02。
- **速度：** 装上 flash-linear-attention（Qwen3.5 的线性注意力层需要）后，MMLU test 14,042 题约 9–14 分钟。

### 1.5 温度拟合

- **全局 TS：** 在校准集上最小化平均 NLL `−log p[正确选项]`。先在 0.05–10 的对数网格（400 个点）上搜索，再从最优点出发用 LBFGS 在 log T 上精调，取 NLL 更低的那个。
- **按学科 TS（收缩）：** 每个学科一个 T，并向全局 T 收缩：`T_s = argmin Σ_{i∈s} NLL_i(T) + λ·(log T − log T_global)²`。
  - λ=0 等于各学科完全独立拟合，λ=∞ 就退化为全局 TS。
  - λ 在校准集上用按学科分层的 5 折交叉验证选取，候选值为 {0, 1, 3, 10, 30, 100, 300, 1000, ∞}；每一折内部重新拟合全局 T，避免信息泄漏。
  - 测试集中没有对应学科的题（MMLU-Pro 中非 MMLU 来源的 43.4%）使用全局 T。
- **不改变预测：** 对所有方法、所有测试集都验证了 argmax 预测与校准前逐题一致，因此准确率不变。
  这是必然的，因为 T>0 时 softmax 不改变排序。

### 1.6 指标

- **ECE：** 等宽分箱，同时报告 15 bins 和 10 bins 两个版本。
- **其他指标：** NLL、Brier（K 类多分类形式）、MCE、平均置信度。
- **高置信区间：** 统计置信度 ≥ 0.8 / 0.9 / 0.95 的题的准确率和平均置信度。
- **分学科 / 分类别表：** 每个学科（MMLU-Pro 为类别）的 ECE 和 gap（= 平均置信度 − 准确率）。
- **显著性：** 方法之间的 NLL 差异用逐题配对的 bootstrap（2,000 次）给出 95% 置信区间。

### 1.7 测试

`uv run pytest -q tests`，共 15 个测试，全部通过：

- 能从合成数据中恢复已知的温度（T = 0.5 / 1 / 2.5，误差在 5% 以内）。
- 温度缩放不改变准确率。
- 缺失的选项（-inf）被正确处理。
- 按学科拟合与收缩的行为符合预期：λ=0 恢复各组真实的 T，λ=∞ 等于全局 T。
- 没有对应学科的题回退到全局 T。
- quyet-v1 与附录示例逐字、逐 token 一致。
- OpenDecisions：
  - 三种题型的字母与标签映射正确。
  - 完美预测经官方评分器得 100 分。
  - 交叉拟合时同一 group 的题落在同一折，所有温度都 ≥ 1。
  - `--fixed-T` 的参数解析正确。

---

## 2. 结果

拟合得到的参数：

| Prompt | 全局 T | 交叉验证选出的 λ | 收缩后学科 T 的范围 | 独立拟合时学科 T 的范围 |
|---|---:|---:|---|---|
| plain | 1.026 | 10 | 0.82–1.67 | 0.05–2.92 |
| quyet-v1 | 1.540 | 10 | 1.10–2.25 | 0.05–3.51 |

交叉验证 NLL 随 λ 的变化：

- plain：0.808（λ=0）→ **0.570（λ=10）** → 0.577（λ=∞）
- quyet-v1：0.753 → **0.612** → 0.619

独立拟合时有学科的 T 落到网格下限 0.05，说明每科十几条样本不足以单独确定 T。

### 2.1 主要指标（各测试集均不参与拟合）

**plain**

| 测试集 | 方法 | Acc | ECE-15 | ECE-10 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| MMLU test | 不校准 | 0.7841 | 0.0179 | 0.0172 | 0.5711 | 0.2962 |
| | 全局 TS | 0.7841 | 0.0162 | 0.0161 | 0.5709 | 0.2959 |
| | 按学科 TS（收缩） | 0.7841 | 0.0161 | 0.0133 | **0.5635** | **0.2937** |
| MMLU-Pro | 不校准 | 0.5104 | 0.0348 | 0.0336 | 1.4556 | **0.6091** |
| | 全局 TS | 0.5104 | 0.0381 | 0.0359 | 1.4554 | 0.6093 |
| | 按学科 TS（收缩） | 0.5104 | **0.0317** | **0.0310** | 1.4554 | 0.6096 |
| MMLU-Redux | 不校准 | 0.8229 | **0.0193** | **0.0187** | **0.4662** | **0.2420** |
| | 全局 TS | 0.8229 | 0.0224 | 0.0217 | 0.4677 | 0.2422 |
| | 按学科 TS（收缩） | 0.8229 | 0.0207 | 0.0175 | 0.4668 | 0.2422 |

**quyet-v1**

| 测试集 | 方法 | Acc | ECE-15 | ECE-10 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| MMLU test | 不校准 | 0.7682 | 0.0796 | 0.0796 | 0.6653 | 0.3270 |
| | 全局 TS | 0.7682 | 0.0103 | **0.0065** | 0.6101 | 0.3150 |
| | 按学科 TS（收缩） | 0.7682 | **0.0082** | 0.0088 | **0.5999** | **0.3118** |
| MMLU-Pro | 不校准 | 0.5085 | 0.1191 | 0.1191 | 1.5821 | 0.6293 |
| | 全局 TS | 0.5085 | **0.0133** | **0.0108** | **1.4647** | **0.6079** |
| | 按学科 TS（收缩） | 0.5085 | 0.0137 | 0.0125 | 1.4666 | 0.6084 |
| MMLU-Redux | 不校准 | 0.8015 | 0.0530 | 0.0534 | 0.5252 | 0.2683 |
| | 全局 TS | 0.8015 | 0.0306 | 0.0323 | 0.5108 | 0.2640 |
| | 按学科 TS（收缩） | 0.8015 | **0.0273** | **0.0272** | **0.5084** | **0.2633** |

各方法的预测与校准前逐题一致，所以同一个测试集内准确率不变。

MCE 不放进主表。它只看最差的那个区间，不考虑样本量。例如 quyet-v1 全局 TS 在 MMLU test 上的 MCE 是 0.266，
完全是由一个只有 **1 条**样本的区间（置信度 0.20–0.27）造成的；样本数不少于 268 的区间，偏差都在 0.06 以内。

### 2.2 方法间 NLL 差异的显著性（配对 bootstrap，95% CI）

| Prompt | 测试集 | 全局 TS − 不校准 | 按学科（收缩）− 全局 TS |
|---|---|---|---|
| plain | MMLU test | −0.0002 [−0.0005, +0.0001] | **−0.0074 [−0.0094, −0.0053]** |
| plain | MMLU-Redux | +0.0015 [+0.0011, +0.0019]（变差） | −0.0009 [−0.0030, +0.0011] |
| plain | MMLU-Pro | −0.0002 [−0.0006, +0.0003] | −0.0001 [−0.0020, +0.0019] |
| quyet-v1 | MMLU test | **−0.0552 [−0.0624, −0.0485]** | **−0.0101 [−0.0124, −0.0079]** |
| quyet-v1 | MMLU-Redux | **−0.0144 [−0.0235, −0.0066]** | −0.0024 [−0.0048, −0.0000] |
| quyet-v1 | MMLU-Pro | **−0.1174 [−0.1273, −0.1081]** | +0.0020 [−0.0003, +0.0042] |

### 2.3 高置信区间（置信度 ≥ 0.9）

| Prompt | 测试集 | 方法 | 占比 | 准确率 | 平均置信度 | 置信度 − 准确率 |
|---|---|---|---:|---:|---:|---:|
| plain | MMLU test | 不校准 | 46.6% | 0.970 | 0.964 | −0.006 |
| plain | MMLU-Redux | 不校准 | 51.7% | 0.986 | 0.964 | −0.022 |
| quyet-v1 | MMLU test | 不校准 | 60.2% | 0.928 | 0.983 | **+0.055** |
| quyet-v1 | MMLU test | 全局 TS | 43.5% | 0.965 | 0.967 | +0.002 |
| quyet-v1 | MMLU-Pro | 不校准 | 28.9% | 0.886 | 0.977 | **+0.091** |
| quyet-v1 | MMLU-Pro | 全局 TS | 15.9% | 0.950 | 0.959 | +0.009 |
| quyet-v1 | MMLU-Redux | 全局 TS | 46.7% | 0.982 | 0.968 | −0.014 |

### 2.4 Reliability diagram

- plain，MMLU test：`results/Qwen__Qwen3.5-9B/mmlu_plain_both/reliability_test.png`
- quyet-v1，MMLU test：`results/Qwen__Qwen3.5-9B/mmlu_quyet-v1_both/reliability_test.png`
- quyet-v1，MMLU-Pro：

![quyet-v1 MMLU-Pro](../results/Qwen__Qwen3.5-9B/mmlu_quyet-v1_both/reliability_mmlu_pro.png)

每次运行的目录里都有对应的 `reliability_mmlu_pro.png`、`reliability_mmlu_redux.png` 和 `nll_vs_T.png`。

### 2.5 学科层面：标注噪声

| 学科 | 测试集 | n | 准确率 | gap（不校准） | gap（全局 TS） | gap（按学科 TS） |
|---|---|---:|---:|---:|---:|---:|
| virology（plain） | MMLU test | 166 | 0.542 | +0.293 | +0.287 | +0.157（T=1.67） |
| virology（plain） | MMLU-Redux（ok） | 43 | 0.837 | +0.018 | +0.013 | **−0.111** |
| virology（quyet-v1） | MMLU test | 166 | 0.524 | +0.337 | +0.256 | +0.160（T=2.25） |
| virology（quyet-v1） | MMLU-Redux（ok） | 43 | 0.860 | +0.041 | −0.040 | **−0.138** |

- MMLU-Redux 把 virology 的 100 题中 57 题标为有问题，其中 33 题是答案标错。只看可信的 43 题，模型的准确率是 0.84，而且校准良好。
- 也就是说，原版 MMLU 上看到的"过度自信"，是模型给出了正确答案、标签却是错的。按学科 TS 把这部分噪声学进了 T，到了干净标签上就变得偏保守。
- 复核之后，plain 在全局 TS 下偏差最大的学科变成了 professional_law（+0.138，n=82）、college_computer_science（−0.091）和 astronomy（−0.089）。
  它们方向不同、幅度中等，而且这些学科的校准样本只有十几到几十条。

---

### 2.6 OpenDecisions Capability（quyet-v1）

**数据：** [OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions)，固定为 revision `ff0b640b`。
- 共 2,352 题：choice 1,178、yes/no 586、score 588，分为 easy / standard / judge / hard 四个难度层级。
- 其中 129 题带有真实概率分布（`target_probs`），用来计算 TVD。

**实现（`ts/opendecisions.py`）：**
- 每题用 quyet-v1 渲染：choice 选项保持 criteria 顺序；yes/no 题 A = true、B = false；score 题按等级从低到高排列。
- 一次前向，在该题的 K 个字母上做 softmax。
- 打分用原样复制进 `ts/vendor/` 的官方 `score.py`，没有自己重写。
- 用数据集自带的 `score.py` 对输出的 `predictions_*.jsonl` 重新打分，结果逐位一致。

**温度：** 每种题型一个 T，按论文的做法强制 T ≥ 1。论文的温度是在另外 3,884 道题上拟合的，这批题没有公开。所以比较以下几种来源：

| 温度 | Capability | Intelligence | Calibration | TVD（129 题） | 相对 T=1（95% CI） |
|---|---:|---:|---:|---:|---|
| T = 1（零样本） | 61.1 | 48.5 | 73.7 | 0.262 | – |
| MMLU 拟合的全局 T = 1.54（不用 OpenDecisions 数据） | 65.8 | 47.8 | 83.9 | 0.236 | +4.7 [+3.4, +5.6] |
| 每题型 T，按 group 5 折交叉拟合 | **66.9** | 47.5 | 86.3 | 0.234 | +5.8 [+3.7, +6.5] |
| 每题型 T，在全部题上拟合（偏乐观，仅作参考） | 67.0 | 47.5 | 86.4 | 0.234 | +5.8 [+3.8, +6.5] |

- **交叉拟合的折：** 由整个 `group` 组成，同一题的改写或扰动版本（每组最多 4 题）不会分到不同的折。没有 group 的 1,726 题各自单独成组。
- **置信区间：** 按 group 做配对 bootstrap（1,000 次），与论文的做法一致。
- **各折拟合出的 T：** choice 1.65–1.69，yes/no 1.64–1.95，score 1.67–1.77。

**分题型结果（T=1 → 交叉拟合）：**

| 题型 | 准确率 | 能力分 | Calibration |
|---|---:|---:|---:|
| choice | 71.5% | 62.3 → 62.3 | 59.0 → 83.1 |
| yes/no | 76.1% | 47.7 → 47.7 | 78.7 → 85.7 |
| score | 60.4% | 35.5 → 32.7 | 83.5 → 90.2 |

**和文档参考值的对照：** `doc/prompt-format.md` 里 quyet-v1 零样本的参考值是 Capability 60.6、I 48.3、C 72.9、TVD 0.262，字母概率占比均值 0.96、最小 0.03。
- 本仓库的结果是 61.1 / 48.5 / 73.7 / 0.262，字母概率占比 0.997 / 0.028。TVD 完全一致，其余相差不到 1 分。
- 剩余的差异可能来自三方面：硬件和数值精度不同（p5en 对 A10G）、我们用 fp32 计算字母 logits、我们合并了带空格和不带空格的字母。
- 最后一点也能解释为什么字母概率占比的均值更高。

**观察：**
- Qwen3.5-9B 在 quyet-v1 下对三种题型都过度自信，最优 T 都在 1.6–2.0 之间，和 MMLU 上的 1.54 接近。
- 所以一个在 MMLU 上拟合的温度，迁移过来就能拿到交叉拟合约 80% 的提升（+4.7 对 +5.8）。
- 温度缩放会让 score 题的能力分下降约 3 分（Intelligence 整体降 1 分）。原因是 score 题的能力分用的是期望等级的误差（nMAE）：温度升高后分布变平，期望等级会被拉向中间。
  这是 Capability 定义本身带来的权衡：
  提高 T 换来 Calibration +12.6，代价是 Intelligence −1.0。

## 3. 结论

1. **letter readout 的校准好坏主要取决于 prompt，而不是模型本身。**
   - 同一个 Qwen3.5-9B，plain prompt 下几乎完美校准（干净标签上 ECE 1.9%，高置信区间还略偏保守）。
   - quyet-v1 chat prompt 下明显过度自信：置信度 ≥ 0.9 的题，实际准确率比置信度低 5.5–9.1 个百分点。
   - plain 的表现与 Kadavath et al. 2022 一致：带字母选项的格式本身就有很好的校准。而在本实验中，加上 chat template、system message 和
     "只回答字母"的指令后，分布明显变尖。这是我们的观察，我们没有逐项拆分是哪个因素造成的。
2. **需要校准时，单一全局 T 就足够，而且能跨数据集迁移。**
   - quyet-v1 的 T = 1.54 只用 MMLU 拟合。迁移到选项更多、难度高得多的 MMLU-Pro 上，ECE 仍从 11.9% 降到 1.3%，NLL 下降 0.117；
     到 MMLU-Redux 上，ECE 从 5.3% 降到 3.1%。这和 Guo et al. 2017 的结论一致。
   - 对本身已经校准良好的 plain prompt，TS 的效果在噪声范围内，在干净标签上甚至略差（NLL +0.0015）。所以拟合出 T≈1 时，可以直接不校准。
3. **不推荐按学科 TS 作为默认方法。**
   - 它在 MMLU test 上的显著收益（NLL −0.007 到 −0.010），在 MMLU-Pro 上消失；在 MMLU-Redux 上 plain 不显著，quyet-v1 只是勉强显著（CI 上界 −0.0000）。
   - 主要原因有两个：一是 MMLU 原版标签的噪声会按学科集中出现（virology 最典型），二是每科只有 13–175 条校准样本。
   - 一定要用的话必须加收缩：独立拟合（λ=0）在所有测试集上都让 NLL 显著变差，比不校准还差。
4. **评估方法本身的教训：** 用原版 MMLU test 评估校准，会把标注错误误判为模型过度自信。校准结论应该在 MMLU-Redux 这类人工复核过的数据上确认。
5. **准确率代价：** quyet-v1 比 plain 低 1.6–2.1 个百分点（MMLU 0.768 vs 0.784，MMLU-Redux 0.802 vs 0.823），在 MMLU-Pro 上两者相当（0.509 vs 0.510）。
   如果实际部署必须用 quyet-v1，在它后面接一个全局 T 就能得到校准良好的置信度。
6. **OpenDecisions 上的结论与 MCQ 一致。**
   - quyet-v1 零样本的 Capability 为 61.1，复现了参考值 60.6。
   - 每题型温度（交叉拟合）把它提升到 66.9，全部来自 Calibration。
   - 只用 MMLU 拟合的单一 T 也能达到 65.8。
   - 也就是说，即使没有 OpenDecisions 风格的校准数据，用公开的 MCQ 数据拟合温度也能拿到大部分收益。

## 4. 局限与后续工作

- **只测了一个模型。** 仓库支持任意 HF 因果语言模型，但本报告只跑了 Qwen3.5-9B。不同模型（尤其是 base 模型和 instruct 模型）的最优 T 可能差别很大。
- **chat prompt 没有评估。** `chat` 格式只确认了能跑通，没有做完整评估。
- **quyet-v1 的 MCQ 映射是自己定的。** 附录没有规定 MCQ 怎么放进 State / Question，这一映射可能影响准确率和校准。
- **分学科指标的噪声。** 每科约 100 题时，分学科 ECE 本身噪声较大，而且样本少时 ECE 会系统性偏高；本报告的学科层面结论以 gap 和配对 bootstrap 为主。
- **校准集也有标注噪声。** MMLU 的 validation / dev 同样含有错误标签。如果用 MMLU-Redux 中干净的部分来拟合 T，可能得到更干净的估计；
  但 Redux 是 test 的子集，这样做需要重新划分测试集。
- **OpenDecisions 的温度：** 论文用来拟合温度的 3,884 道题没有公开，所以交叉拟合的数字和论文的设置并不完全相同。交叉拟合使用了 OpenDecisions 本身的数据；只有"MMLU 拟合的 T"那一行完全没用到它。
- **可选扩展：** 用依赖输入的自适应 TS（ATS）。按目前的结果，预期收益有限：交叉拟合和在全集上拟合的结果几乎一样（66.9 对 67.0），说明每题型一个 T 已经接近这种方法的上限。

## 5. 复现

```bash
uv sync
uv run python -m ts.run --model Qwen/Qwen3.5-9B --benchmark mmlu --prompt-format plain     # 约 30 分钟（A10G，含 MMLU-Pro / MMLU-Redux）
uv run python -m ts.run --model Qwen/Qwen3.5-9B --benchmark mmlu --prompt-format quyet-v1  # 约 30 分钟
uv run python -m ts.opendecisions --model Qwen/Qwen3.5-9B --fixed-T 1.5404 --fixed-T-note "MMLU-fitted T"  # 约 7 分钟
uv run python -m ts.summarize        # → results/summary.md
uv run pytest -q tests
```

logits 已缓存。删掉 `results/` 下对应的 `logits_*.npz` 或加 `--force` 可以重新计算前向；只改拟合或评估代码时，重跑只需要几秒。
