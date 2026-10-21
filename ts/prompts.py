"""Prompt construction.

``plain`` / ``chat``: question + lettered options, ending in ``Answer:``.
``quyet-v1``: the decision-model prompt specified in doc/prompt-format.md (system message,
State / Question / type hint / Options, "Answer with one letter."), chat template with
thinking off; the letter logits are read at the start of the assistant turn.
"""

from __future__ import annotations

import json

from .data import LETTERS

QUYET_SYSTEM = (
    "You are a decision model. Read the state and the question, then choose exactly one option. "
    "Base the decision only on the state and the option descriptions. Reply with the option letter only."
)
QUYET_HINTS = {
    "choice": "Choose the option that fits best.",
    "score": "Choose the level that fits best (levels are ordered from lowest to highest).",
    "noul": "Choose A if the statement is true for this state, B if it is not.",
}
QUYET_LETTERS = "ABCDEFGHIJKLMNOP"  # up to 16 options
MAX_STATE_TOKENS = 6000
MAX_PROMPT_TOKENS = 8000


def format_question(item: dict) -> str:
    subject = item["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", item["question"].strip()]
    lines += [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(item["choices"])]
    return "\n".join(lines)


def _apply_chat(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------- quyet-v1


def quyet_options(question: dict) -> list[str]:
    """Option lines (without letters) for a typed question ``{type, instructions, criteria}``."""
    qtype, criteria = question["type"], question.get("criteria")
    if qtype == "choice":
        # {label: description} keeps the request's order; a bare list of labels reads as the labels
        if isinstance(criteria, dict):
            return [str(desc).strip() if desc else str(label) for label, desc in criteria.items()]
        return [str(label) for label in criteria]
    if qtype == "score":
        return [str(level).strip() for level in criteria]  # lowest first, never reordered
    if qtype == "noul":
        criteria = criteria or {}
        return [str(criteria.get("true") or "true").strip(), str(criteria.get("false") or "false").strip()]
    raise ValueError(f"unknown question type {qtype!r}")


def _render_state(state) -> str:
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, indent=1, ensure_ascii=False)


def _truncate_state(state, tokenizer, max_tokens: int):
    """Conversations keep their most recent turns; other states keep their beginning."""
    if tokenizer is None:
        return _render_state(state)
    if isinstance(state, list):
        turns = list(state)
        while len(turns) > 1 and len(tokenizer.encode(_render_state(turns), add_special_tokens=False)) > max_tokens:
            turns.pop(0)
        state = turns
    text = _render_state(state)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        text = tokenizer.decode(ids[:max_tokens])
    return text


def quyet_v1_messages(state, question: dict, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[dict]:
    options = quyet_options(question)
    if len(options) > len(QUYET_LETTERS):
        raise ValueError(f"at most {len(QUYET_LETTERS)} options are supported, got {len(options)}")
    user = (
        f"State:\n{_truncate_state(state, tokenizer, max_state_tokens)}\n\n"
        f"Question: {str(question['instructions']).strip()}\n{QUYET_HINTS[question['type']]}\n\n"
        "Options:\n" + "\n".join(f"{QUYET_LETTERS[i]}. {o}" for i, o in enumerate(options)) + "\n\nAnswer with one letter."
    )
    return [{"role": "system", "content": QUYET_SYSTEM}, {"role": "user", "content": user}]


def render_quyet_v1(state, question: dict, tokenizer) -> str:
    text = _apply_chat(tokenizer, quyet_v1_messages(state, question, tokenizer))
    n = len(tokenizer.encode(text, add_special_tokens=False))
    if n > MAX_PROMPT_TOKENS:
        # shrink the state budget by the overflow and render again
        budget = max(1, MAX_STATE_TOKENS - (n - MAX_PROMPT_TOKENS))
        text = _apply_chat(tokenizer, quyet_v1_messages(state, question, tokenizer, budget))
    return text


def mcq_to_quyet(item: dict) -> tuple[str, dict]:
    """MCQ item -> (state, choice question): the question text is the state, the
    answer texts are the option descriptions (order kept, so letter i = choice i)."""
    subject = item["subject"].replace("_", " ")
    question = {
        "type": "choice",
        "instructions": f"Which option is the correct answer to this {subject} question?",
        "criteria": {LETTERS[i]: str(c).strip() for i, c in enumerate(item["choices"])},
    }
    return item["question"], question


# ---------------------------------------------------------------------------- dispatch


def build_prompt(item: dict, tokenizer, fmt: str) -> str:
    """``plain``: raw text ending in "Answer:" (works for base and instruct models).
    ``chat``: the question goes through the chat template (thinking disabled), and the
    assistant turn is pre-filled with "Answer:" so the next token is the letter.
    ``quyet-v1``: see module docstring."""
    if fmt == "quyet-v1":
        return render_quyet_v1(*mcq_to_quyet(item), tokenizer)
    body = format_question(item)
    if fmt == "plain":
        return body + "\nAnswer:"
    if fmt == "chat":
        k = len(item["choices"])
        user = body + f"\n\nAnswer with only the letter ({LETTERS[0]}-{LETTERS[k - 1]}) of the correct option."
        return _apply_chat(tokenizer, [{"role": "user", "content": user}]) + "Answer:"
    raise ValueError(f"unknown prompt format {fmt!r}")


PROMPT_FORMATS = ("plain", "chat", "quyet-v1")
