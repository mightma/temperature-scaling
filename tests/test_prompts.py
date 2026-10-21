"""quyet-v1 parity with the rendered examples in doc/prompt-format.md (text and token count)."""

import re
from pathlib import Path

import pytest

from ts.prompts import build_prompt, render_quyet_v1

SPEC = Path(__file__).resolve().parents[1] / "doc" / "prompt-format.md"

# The three OpenDecisions requests shown in the spec, with their documented token counts.
EXAMPLES = [
    (
        "jb16-llm_router-9a24706d03",
        226,
        "Here are three expense lines from my trip. Turn them into a JSON list with fields merchant, date, amount_usd.\n"
        "- Blue Cafe, 2024-03-02, $14.50\n- MetroCab, 2024-03-02, $23.00\n- Hotel Lira, 2024-03-04, $189.90",
        {
            "type": "choice",
            "instructions": "Classify the task type of this request for routing.",
            "criteria": {
                "chitchat": "Casual small talk, no task.",
                "writing": "Compose original text.",
                "extraction_json": "Pull given details into a structured JSON shape.",
                "analysis": "Compute statistics or trends over the data.",
            },
        },
    ),
    (
        "jb16-llm_router-53e98471ff",
        209,
        {
            "small_model": "minnow-s",
            "context_window_tokens": 32000,
            "request": {"prompt_tokens": 4100, "attachments": [{"name": "api_spec_v7.pdf", "tokens": 61000}]},
        },
        {
            "type": "noul",
            "instructions": "Does this request exceed minnow-s's context window?",
            "criteria": {
                "true": "The prompt plus attachments is larger than the context window.",
                "false": "The prompt plus attachments fits in the context window.",
            },
        },
    ),
    (
        "jb16-llm_router-9e4a2e3bd8",
        194,
        "Implement a lock-free multi-producer multi-consumer queue in C++20 using std::atomic, and include an argument "
        "for why it is linearizable and safe from the ABA problem.",
        {
            "type": "score",
            "instructions": "How difficult is this request?",
            "criteria": [
                "trivial: a greeting, a lookup, or a formatting change",
                "routine: short standard code or an explanation a junior developer could write",
                "involved: a multi-part task or an investigation across several components",
                "expert: concurrent algorithms, formal correctness arguments, or security-critical design",
            ],
        },
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def _spec_example(item_id: str) -> str:
    text = SPEC.read_text()
    block = re.search(rf"{re.escape(item_id)}.*?\n    ```\n(.*?)\n    ```", text, re.S).group(1)
    return "\n".join(line[4:] for line in block.split("\n")) + "\n"


@pytest.mark.parametrize("item_id,n_tokens,state,question", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_quyet_v1_matches_spec(tokenizer, item_id, n_tokens, state, question):
    text = render_quyet_v1(state, question, tokenizer)
    assert text == _spec_example(item_id)
    assert len(tokenizer.encode(text, add_special_tokens=False)) == n_tokens


def test_mcq_in_quyet_v1(tokenizer):
    item = {"question": "What is 2 + 2?", "choices": ["3", "4", "5", "22"], "answer": 1, "subject": "elementary_mathematics"}
    text = build_prompt(item, tokenizer, "quyet-v1")
    assert "State:\nWhat is 2 + 2?\n\nQuestion: Which option is the correct answer to this elementary mathematics question?" in text
    assert "Options:\nA. 3\nB. 4\nC. 5\nD. 22\n\nAnswer with one letter.<|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
