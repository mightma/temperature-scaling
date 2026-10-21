import numpy as np

from ts.calibration import ece, fit_temperature, metrics, softmax


def _sample(T_true, n=20000, k=4, seed=0):
    # Labels drawn from softmax(z / T_true): the NLL-optimal temperature is T_true.
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, size=(n, k))
    p = softmax(z, T_true)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z, y


def test_recovers_known_temperature():
    for T_true in (0.5, 1.0, 2.5):
        z, y = _sample(T_true)
        assert abs(fit_temperature(z, y)["T"] - T_true) / T_true < 0.05


def test_scaling_preserves_accuracy_and_lowers_ece():
    z, y = _sample(2.0)
    before, after = metrics(z, y), metrics(z, y, fit_temperature(z, y)["T"])
    assert before["accuracy"] == after["accuracy"]
    assert after["ece_15"] < before["ece_15"] and after["nll"] < before["nll"]


def test_missing_options_are_ignored():
    z = np.array([[2.0, 1.0, -np.inf, -np.inf], [0.0, 1.0, 3.0, -np.inf]])
    p = softmax(z, 1.5)
    assert np.allclose(p.sum(1), 1) and (p[0, 2:] == 0).all()
    assert np.isfinite(metrics(z, np.array([0, 2]), 1.5)["nll"])


def test_ece_perfect_and_worst():
    assert ece(np.ones(10), np.ones(10)) == 0.0
    assert ece(np.ones(10), np.zeros(10)) == 1.0


def test_group_temperatures_and_shrinkage():
    from ts.calibration import fit_group_temperatures, select_shrinkage

    za, ya = _sample(0.5, n=4000, seed=1)
    zb, yb = _sample(3.0, n=4000, seed=2)
    z, y = np.concatenate([za, zb]), np.concatenate([ya, yb])
    g = np.array(["a"] * len(ya) + ["b"] * len(yb))
    T0 = fit_temperature(z, y)["T"]
    indep = fit_group_temperatures(z, y, g, T0, lam=0)
    assert abs(indep["a"] - 0.5) < 0.05 and abs(indep["b"] - 3.0) < 0.3
    assert fit_group_temperatures(z, y, g, T0, lam=np.inf) == {"a": T0, "b": T0}
    shrunk = fit_group_temperatures(z, y, g, T0, lam=1e4)
    assert indep["a"] < shrunk["a"] < T0 < shrunk["b"] < indep["b"]
    # with plenty of data per group and truly different temperatures, CV should not pick global TS
    assert np.isfinite(select_shrinkage(z, y, g)["lam"])


def test_unknown_subjects_fall_back_to_global_T():
    from ts.run import apply_methods

    details = {"global": {"T": 1.5}, "subject_shrunk": {"lam": 10.0, "T": {"virology": 2.0}}}
    temps = apply_methods(np.array(["virology", "physics"]), details)
    assert temps["none"].tolist() == [1.0, 1.0]
    assert temps["global"].tolist() == [1.5, 1.5]
    assert temps["subject_shrunk"].tolist() == [2.0, 1.5]
