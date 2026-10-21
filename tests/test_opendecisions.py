"""OpenDecisions glue: letter <-> label mapping, scoring through the vendored scorer, cross-fit folds."""

import numpy as np

from ts.opendecisions import gold_index, temperatures, to_probs
from ts.vendor.opendecisions_score import score as official_score


def _tasks():
    choice = {
        "id": "c",
        "tier": "hard",
        "labels": ["x", "y", "z"],
        "expected": "y",
        "question": {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y", "z": "Z"}},
    }
    noul = {
        "id": "n",
        "tier": "easy",
        "labels": ["no", "yes"],
        "expected": "yes",
        "question": {"type": "noul", "instructions": "?", "criteria": {"true": "t", "false": "f"}},
    }
    score = {
        "id": "s",
        "tier": "judge",
        "labels": ["0", "1", "2", "3"],
        "expected": 2,
        "question": {"type": "score", "instructions": "?", "criteria": ["a", "b", "c", "d"]},
    }
    return [choice, noul, score]


def test_gold_index_follows_quyet_option_order():
    c, n, s = _tasks()
    assert gold_index(c) == 1  # criteria order == labels
    assert gold_index(n) == 0  # A = true = "yes"
    assert gold_index(s) == 2


def test_to_probs_matches_scorer_label_format():
    c, n, s = _tasks()
    assert to_probs(c, np.array([0.1, 0.7, 0.2])) == {"x": 0.1, "y": 0.7, "z": 0.2}
    assert to_probs(n, np.array([0.8, 0.2])) == {"yes": 0.8, "no": 0.2}
    assert to_probs(s, np.array([0.1, 0.2, 0.6, 0.1])) == {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}


def test_perfect_and_uniform_predictions():
    tasks = _tasks()
    k = [3, 2, 4]
    perfect = [{"id": t["id"], "probs": to_probs(t, np.eye(kk)[gold_index(t)])} for t, kk in zip(tasks, k)]
    best = official_score(tasks, perfect)
    assert all(abs(best[k] - 100) < 1e-9 for k in ("intelligence", "calibration", "capability"))
    uniform = [{"id": t["id"], "probs": to_probs(t, np.full(kk, 1 / kk))} for t, kk in zip(tasks, k)]
    assert official_score(tasks, uniform)["intelligence"] < 50


def test_crossfit_keeps_groups_together_and_respects_min_T():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.normal(0, 4, size=(n, 4))  # sharp logits: the NLL-optimal T is > 1
    y = rng.integers(0, 4, size=n)
    types = np.array(["choice", "noul", "score", "choice"] * (n // 4))
    groups = np.array([f"g{i // 2}" for i in range(n)])  # pairs of items share a group
    item_T, fit = temperatures(z, y, types, groups, min_T=1.0, max_T=20.0, n_folds=5, seed=0)
    assert (item_T["t1"] == 1).all()
    assert (item_T["crossfit"] >= 1).all() and (item_T["insample"] >= 1).all()
    assert len(fit["crossfit_folds"]) == 5
    # both items of a group get their T from the same fold, so for the same type they get the same T
    same_type_pairs = [(i, i + 1) for i in range(0, n, 2) if types[i] == types[i + 1]]
    assert all(item_T["crossfit"][i] == item_T["crossfit"][j] for i, j in same_type_pairs)


def test_parse_fixed_T():
    import pytest

    from ts.opendecisions import parse_fixed_T

    assert parse_fixed_T("1.5") == {"choice": 1.5, "noul": 1.5, "score": 1.5}
    assert parse_fixed_T("5.96,7.24,7.24") == {"choice": 5.96, "noul": 7.24, "score": 7.24}
    with pytest.raises(ValueError):
        parse_fixed_T("1,2")
