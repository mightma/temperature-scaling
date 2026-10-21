"""Temperature fitting and calibration metrics on option-restricted logits.

Logits are [N, K] with -inf for options that don't exist on a given question.
"""

from __future__ import annotations

import numpy as np
import torch


def _per_item(T):
    """Scalar T, or one temperature per item -> broadcastable column."""
    T = np.asarray(T, dtype=np.float64)
    return T.reshape(-1, 1) if T.ndim else T


def softmax(z: np.ndarray, T=1.0) -> np.ndarray:
    z = z.astype(np.float64) / _per_item(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(z: np.ndarray, y: np.ndarray, T=1.0) -> float:
    t = torch.as_tensor(z, dtype=torch.float64) / torch.as_tensor(_per_item(T))
    return float(-torch.log_softmax(t, dim=1)[torch.arange(len(y)), torch.as_tensor(y)].mean())


def fit_temperature(z: np.ndarray, y: np.ndarray, t_min: float = 0.05, t_max: float = 10.0) -> dict:
    """Minimise validation NLL over a single scalar T.

    A log-spaced grid over [t_min, t_max] gives a global starting point (NLL in T is not
    guaranteed convex), then LBFGS on log T refines it. The better of the two is kept."""
    grid = np.exp(np.linspace(np.log(t_min), np.log(t_max), 400))
    t_logits = torch.as_tensor(z, dtype=torch.float64)
    y_t = torch.as_tensor(y)
    rows = torch.arange(len(y))
    grid_nll = np.array([float(-torch.log_softmax(t_logits / T, 1)[rows, y_t].mean()) for T in grid])
    T_grid = float(grid[grid_nll.argmin()])

    log_t = torch.tensor(np.log(T_grid), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -torch.log_softmax(t_logits / log_t.exp(), 1)[rows, y_t].mean()
        loss.backward()
        return loss

    opt.step(closure)
    T_lbfgs = float(np.clip(log_t.exp().item(), t_min, t_max))
    T = T_lbfgs if nll(z, y, T_lbfgs) <= nll(z, y, T_grid) else T_grid
    return {"T": T, "T_grid": T_grid, "T_lbfgs": T_lbfgs, "grid": grid.tolist(), "grid_nll": grid_nll.tolist()}


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    gaps = [abs(correct[bins == b].mean() - conf[bins == b].mean()) for b in range(n_bins) if (bins == b).any()]
    return float(max(gaps)) if gaps else 0.0


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": int(m.sum()),
                "acc": float(correct[m].mean()) if m.any() else None,
                "conf": float(conf[m].mean()) if m.any() else None,
            }
        )
    return out


def metrics(z: np.ndarray, y: np.ndarray, T=1.0, n_bins: tuple[int, ...] = (15, 10)) -> dict:
    p = softmax(z, T)
    pred = p.argmax(1)
    conf = p.max(1)
    correct = (pred == y).astype(np.float64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    out = {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "nll": nll(z, y, T),
        "brier": float(((p - onehot) ** 2).sum(1).mean()),
        "mean_confidence": float(conf.mean()),
    }
    for b in n_bins:
        out[f"ece_{b}"] = ece(conf, correct, b)
        out[f"mce_{b}"] = mce(conf, correct, b)
    return out


def _grid_nll_sums(z: np.ndarray, y: np.ndarray, log_grid: np.ndarray) -> np.ndarray:
    """Summed NLL of (z, y) at every temperature exp(log_grid). Returns [len(grid)]."""
    t = torch.as_tensor(z, dtype=torch.float64)[None] / torch.as_tensor(np.exp(log_grid))[:, None, None]
    lp = torch.log_softmax(t, dim=2)[:, torch.arange(len(y)), torch.as_tensor(y)]
    return (-lp.sum(1)).numpy()


def fit_group_temperatures(
    z: np.ndarray, y: np.ndarray, groups: np.ndarray, T_global: float, lam: float, t_min: float = 0.05, t_max: float = 10.0
) -> dict[str, float]:
    """One T per group, shrunk toward T_global (MAP with a Gaussian prior on log T):

        T_g = argmin_T  sum_{i in g} NLL_i(T) + lam * (log T - log T_global)^2

    lam = 0 fits every group independently; lam -> inf recovers the single global T.
    Solved on a fine log-spaced grid (step ~0.5%), which is exact enough and robust for tiny groups."""
    log_grid = np.linspace(np.log(t_min), np.log(t_max), 1200)
    prior = lam * (log_grid - np.log(T_global)) ** 2 if np.isfinite(lam) else None
    out = {}
    for g in np.unique(groups):
        if prior is None:
            out[str(g)] = float(T_global)
            continue
        m = groups == g
        obj = _grid_nll_sums(z[m], y[m], log_grid) + prior
        out[str(g)] = float(np.exp(log_grid[obj.argmin()]))
    return out


def item_temperatures(groups: np.ndarray, group_T: dict[str, float], default: float) -> np.ndarray:
    return np.array([group_T.get(str(g), default) for g in groups])


def select_shrinkage(
    z: np.ndarray, y: np.ndarray, groups: np.ndarray, lams=(0, 1, 3, 10, 30, 100, 300, 1000, np.inf), n_folds: int = 5, seed: int = 0
) -> dict:
    """Pick lam by K-fold cross-validated NLL on the calibration set only.

    Folds are stratified by group so every group appears in every training fold. The global
    T is re-fit inside each fold so nothing leaks from the held-out fold."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
    cv = {lam: 0.0 for lam in lams}
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        T0 = fit_temperature(z[tr], y[tr])["T"]
        for lam in lams:
            gT = fit_group_temperatures(z[tr], y[tr], groups[tr], T0, lam)
            cv[lam] += nll(z[te], y[te], item_temperatures(groups[te], gT, T0)) * te.sum()
    cv = {lam: v / len(y) for lam, v in cv.items()}
    best = min(cv, key=cv.get)
    return {"lam": float(best), "cv_nll": {str(k): v for k, v in cv.items()}}
