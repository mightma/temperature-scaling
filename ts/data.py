"""MCQ benchmark loaders.

Every loader returns ``{"calib": [...], "test": [...]}`` where each item is a dict
``{"question": str, "choices": list[str], "answer": int, "subject": str}``.
The calibration split is the only data used to fit T; the test split is held out.
"""

from __future__ import annotations

import random

from datasets import concatenate_datasets, load_dataset

LETTERS = "ABCDEFGHIJ"


def _mmlu(seed: int, calib_size: int | None) -> dict[str, list[dict]]:
    ds = load_dataset("cais/mmlu", "all")
    calib = concatenate_datasets([ds["validation"], ds["dev"]])

    def conv(split):
        return [
            {"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]), "subject": r["subject"]}
            for r in split
        ]

    return {"calib": _subsample(conv(calib), calib_size, seed), "test": conv(ds["test"])}


def _mmlu_pro(seed: int, calib_size: int | None) -> dict[str, list[dict]]:
    # MMLU-Pro only ships a 70-item validation set, so carve a calibration split off the
    # test set with a fixed seed; the remainder is the held-out test set.
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    items = [
        {"question": r["question"], "choices": list(r["options"]), "answer": int(r["answer_index"]), "subject": r["category"]}
        for split in ("validation", "test")
        for r in ds[split]
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    n = calib_size or 2000
    return {"calib": items[:n], "test": items[n:]}


def _arc_challenge(seed: int, calib_size: int | None) -> dict[str, list[dict]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def conv(split):
        out = []
        for r in split:
            labels = list(r["choices"]["label"])
            # A few items use numeric labels ("1".."4"); options are re-lettered A.. in order.
            out.append(
                {
                    "question": r["question"],
                    "choices": list(r["choices"]["text"]),
                    "answer": labels.index(r["answerKey"]),
                    "subject": "arc_challenge",
                }
            )
        return out

    calib = conv(ds["train"]) + conv(ds["validation"])
    return {"calib": _subsample(calib, calib_size, seed), "test": conv(ds["test"])}


def _subsample(items: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


BENCHMARKS = {
    "mmlu": _mmlu,
    "mmlu_pro": _mmlu_pro,
    "arc_challenge": _arc_challenge,
}


# ---------------------------------------------------------------- test-only evaluation sets
# Never used for fitting. Items carry ``subject`` (the key used to look up a per-subject T
# fitted on the calibration split; unknown keys fall back to the global T) and ``group``
# (how results are broken down in the report).


def _mmlu_pro_test() -> list[dict]:
    # Full MMLU-Pro test split (the 70-item validation split is the CoT few-shot pool).
    # 6,810 items come from original MMLU and keep their MMLU subject ("ori_mmlu-<subject>");
    # the rest (stemez / theoremQA / scibench) have no MMLU subject.
    out = []
    for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
        src = r["src"]
        subject = src.split("ori_mmlu-", 1)[1] if src.startswith("ori_mmlu-") else r["category"]
        out.append(
            {"question": r["question"], "choices": list(r["options"]), "answer": int(r["answer_index"]), "subject": subject, "group": r["category"]}
        )
    return out



def _mmlu_redux_test() -> list[dict]:
    # MMLU-Redux 2.0: 100 re-annotated MMLU test questions per subject. Only items annotated
    # error_type == "ok" are kept, so the labels are trusted.
    subjects = sorted(set(load_dataset("cais/mmlu", "all", split="dev")["subject"]))
    out = []
    for s in subjects:
        for r in load_dataset("edinburgh-dawg/mmlu-redux-2.0", s, split="test"):
            if r["error_type"] == "ok":
                out.append({"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]), "subject": s, "group": s})
    return out


EXTRA_TESTS = {
    "mmlu_pro": _mmlu_pro_test,
    "mmlu_redux": _mmlu_redux_test,
}


def load_extra_test(name: str, max_items: int | None = None, seed: int = 0) -> list[dict]:
    if name not in EXTRA_TESTS:
        raise ValueError(f"unknown extra test set {name!r}; choose from {sorted(EXTRA_TESTS)}")
    return _subsample(EXTRA_TESTS[name](), max_items, seed)


def load_benchmark(name: str, seed: int = 0, calib_size: int | None = None, max_test: int | None = None) -> dict[str, list[dict]]:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}")
    splits = BENCHMARKS[name](seed, calib_size)
    if max_test is not None:
        splits["test"] = _subsample(splits["test"], max_test, seed)
    return splits
