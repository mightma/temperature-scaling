"""Reliability diagrams and the validation NLL-vs-T curve."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibration import reliability_bins, softmax

BEFORE = "#2a78d6"  # categorical slot 1
AFTER = "#eb6834"  # categorical slot 2
SUBJECT = "#1baf7a"  # categorical slot 3
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability_diagram(z, y, panels, path, n_bins=15, title=""):
    """``panels``: list of (name, per-item temperatures); colours follow the fixed slot order."""
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True, squeeze=False)
    for col, ((name, temp), color) in enumerate(zip(panels, (BEFORE, AFTER, SUBJECT))):
        p = softmax(z, temp)
        conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
        bins = reliability_bins(conf, correct, n_bins)
        width = 1.0 / n_bins
        centers = np.array([(b["lo"] + b["hi"]) / 2 for b in bins])
        acc = np.array([b["acc"] if b["acc"] is not None else np.nan for b in bins])
        mconf = np.array([b["conf"] if b["conf"] is not None else np.nan for b in bins])
        counts = np.array([b["count"] for b in bins])
        ece = np.nansum(counts / counts.sum() * np.abs(acc - mconf))

        ax = axes[0, col]
        _style(ax)
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="Perfect calibration")
        ax.bar(centers, np.nan_to_num(acc), width=width - 0.006, color=color, label="Accuracy in bin")
        ax.plot(mconf, acc, "o", color=INK, markersize=4, label="(mean conf, acc)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy", color=INK_2)
        ax.set_title(f"{name}\nECE$_{{{n_bins}}}$ = {ece:.4f}", color=INK, fontsize=11, loc="left")
        if col == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)

        ax = axes[1, col]
        _style(ax)
        ax.bar(centers, counts / counts.sum(), width=width - 0.006, color=color)
        ax.set_xlabel("Confidence (max option probability)", color=INK_2)
        ax.set_ylabel("Share of items", color=INK_2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def nll_curve(fit: dict, path, title=""):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _style(ax)
    ax.plot(fit["grid"], fit["grid_nll"], color=BEFORE, linewidth=2)
    ax.axvline(1.0, color=INK_2, linewidth=1, linestyle="--")
    ax.axvline(fit["T"], color=AFTER, linewidth=2)
    ax.annotate(f"T* = {fit['T']:.3f}", (fit["T"], max(fit["grid_nll"])), xytext=(6, -12), textcoords="offset points", color=INK, fontsize=9)
    ax.annotate("T = 1", (1.0, max(fit["grid_nll"])), xytext=(-34, -12), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Temperature T (log scale)", color=INK_2)
    ax.set_ylabel("Calibration-set NLL", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
