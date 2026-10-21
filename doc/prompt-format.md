# Prompt format (quyet-v1)

Moved verbatim from the original README appendix. Implemented in `ts/prompts.py` (`--prompt-format quyet-v1`); `tests/test_prompts.py` checks the implementation against the rendered examples below, character for character and token for token.

### 2.1 Prompt format (decided 2026-10-19: Quyet v1, aligned with Jev)

Requests use Jev's `/v1/systemone` shape: a **state** (text, a JSON object, or a list of conversation turns) and typed **questions** `{type, instructions, criteria}`.

- `choice`: criteria are `{label: description}` (or a list of labels). The answer keys are the labels.
- `score`: criteria are a list of levels, lowest first. The answer keys are `"0"`…`"K-1"`.
- `noul` (yes/no): criteria are optional `{"true": …, "false": …}`. The answer is P(true).

Each question is rendered in **Quyet's training prompt (quyet-v1)** with the chat template and thinking off. The next-token logits of the K letters give the probabilities: $p = \text{softmax}(z_{A..K}/T_{type})$.

The prompt is implemented in `decision_model/prompt.py` (github.com/Chevolier/Strands-Decider). `scripts/check_prompt_parity.py` confirms it is identical, character for character and token for token, to the `quyet` package on all 2,352 OpenDecisions items, with both the Qwen3.5 and Gemma 4 tokenizers.

```
[system] You are a decision model. Read the state and the question, then choose exactly one option. Base the decision only on the state and the option descriptions. Reply with the option letter only.
[user]   State:
         {state: the text as given, or JSON indented by 1 space}

         Question: {instructions}
         {type hint}

         Options:
         A. {description of option 1, or its label if it has none}
         B. ...

         Answer with one letter.
[assistant]  (thinking disabled; the letter logits are read here)

type hint:
  choice -> Choose the option that fits best.
  score  -> Choose the level that fits best (levels are ordered from lowest to highest).
  noul   -> Choose A if the statement is true for this state, B if it is not.
```

**Rules.**

- Choice options keep the request's order, and score levels are listed lowest first. Training shuffles choice options, but never score levels.
- Yes/no is always A = true, B = false. If no descriptions are given, the options read "true" and "false".
- Truncation: conversations keep their most recent turns and other states keep their beginning, within 6,000 state tokens and an 8,000-token prompt.
- Quyet trained with up to 10 options (A–J). We allow 16 (A–P).
- **quyet-v2** (Quyet-1.0-Large): no system message, no closing line, compact JSON states (about 68 fewer tokens). **semif** (Jeff / OpenDecisions `predict_letters.py`): the item as one JSON object. Both are kept for evaluation and as optional training augmentation.

**Rendered examples** (OpenDecisions items, Qwen3.5-9B chat template):

- choice: jb16-llm_router-9a24706d03 (answer C, extraction_json; 226 tokens)
    
    ```
    <|im_start|>system
    You are a decision model. Read the state and the question, then choose exactly one option. Base the decision only on the state and the option descriptions. Reply with the option letter only.<|im_end|>
    <|im_start|>user
    State:
    Here are three expense lines from my trip. Turn them into a JSON list with fields merchant, date, amount_usd.
    - Blue Cafe, 2024-03-02, $14.50
    - MetroCab, 2024-03-02, $23.00
    - Hotel Lira, 2024-03-04, $189.90
    
    Question: Classify the task type of this request for routing.
    Choose the option that fits best.
    
    Options:
    A. Casual small talk, no task.
    B. Compose original text.
    C. Pull given details into a structured JSON shape.
    D. Compute statistics or trends over the data.
    
    Answer with one letter.<|im_end|>
    <|im_start|>assistant
    <think>
    
    </think>
    
    ```
    
- noul with a JSON state: jb16-llm_router-53e98471ff (answer yes = A; 209 tokens)
    
    ```
    <|im_start|>system
    You are a decision model. Read the state and the question, then choose exactly one option. Base the decision only on the state and the option descriptions. Reply with the option letter only.<|im_end|>
    <|im_start|>user
    State:
    {
     "small_model": "minnow-s",
     "context_window_tokens": 32000,
     "request": {
      "prompt_tokens": 4100,
      "attachments": [
       {
        "name": "api_spec_v7.pdf",
        "tokens": 61000
       }
      ]
     }
    }
    
    Question: Does this request exceed minnow-s's context window?
    Choose A if the statement is true for this state, B if it is not.
    
    Options:
    A. The prompt plus attachments is larger than the context window.
    B. The prompt plus attachments fits in the context window.
    
    Answer with one letter.<|im_end|>
    <|im_start|>assistant
    <think>
    
    </think>
    
    ```
    
- score: jb16-llm_router-9e4a2e3bd8 (answer level 3 = D; 194 tokens)
    
    ```
    <|im_start|>system
    You are a decision model. Read the state and the question, then choose exactly one option. Base the decision only on the state and the option descriptions. Reply with the option letter only.<|im_end|>
    <|im_start|>user
    State:
    Implement a lock-free multi-producer multi-consumer queue in C++20 using std::atomic, and include an argument for why it is linearizable and safe from the ABA problem.
    
    Question: How difficult is this request?
    Choose the level that fits best (levels are ordered from lowest to highest).
    
    Options:
    A. trivial: a greeting, a lookup, or a formatting change
    B. routine: short standard code or an explanation a junior developer could write
    C. involved: a multi-part task or an investigation across several components
    D. expert: concurrent algorithms, formal correctness arguments, or security-critical design
    
    Answer with one letter.<|im_end|>
    <|im_start|>assistant
    <think>
    
    </think>
    
    ```
    

**Zero-shot check on OpenDecisions** (Qwen3.5-9B, T = 1, run 2026-10-19 on the p5en). Letter mass is the share of next-token probability on the option letters.

| Prompt | Capability | Intelligence | Calibration | TVD | Letter mass, mean / min |
| --- | --- | --- | --- | --- | --- |
| **quyet-v1** | **60.6** | 48.3 | **72.9** | 0.262 | 0.96 / 0.03 |
| semif | 59.1 | 51.6 | 66.6 | 0.250 | 0.999 / 0.93 |
| quyet-v2 | 49.1 | 33.5 | 64.6 | 0.366 | 0.23 / 0.0000001 |
- quyet-v1 scores highest on Capability, mostly through Calibration, but loses 3.3 Intelligence points against semif.
- quyet-v2 needs a trained model: without the system message and the closing line, the untrained model puts only 23% of its probability on the letters. So we train on v1.
- Validation: through the same predictor, Quyet-1.0-Large (v2 prompt, its own temperatures) scores 81.5, against 81.3 with the `quyet` package. The median difference in probabilities is 0.0007, and 10 of 2,352 top answers change, because the package batches items with padding. The semif path reproduces OpenDecisions' `predict_letters.py` exactly.

**Training-target format.** Each training row is stored in the same request shape: `{"state", "question": {type, instructions, criteria}, "target": {answer key: probability}, "source", "license"}`. The trainer renders it with `decision_model.prompt.encode`, so training and serving use the same prompt.
