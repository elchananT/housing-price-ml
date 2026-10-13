"""One consistent plotting style and the figure helpers shared by all notebooks.

Design rules: white background, dark-ink text, thin light grid, a single teal
highlight colour for "the thing to look at", amber for contrast, and *titles that
state the finding* (with a muted one-line subtitle for the how/what).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from . import config

INK = "#1b2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
TEAL = "#0b6e75"
TEAL_LIGHT = "#a8d5d8"
AMBER = "#e08a1e"
RED = "#c0392b"
BLUE = "#2f5d9e"
PURPLE = "#6c5ba7"
GREY = "#9ca3af"
PALETTE = [TEAL, AMBER, BLUE, PURPLE, RED, GREY]

usd = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")
usd_m = FuncFormatter(lambda v, _: f"${v:,.0f}")


def set_style() -> None:
    """Apply the project-wide matplotlib/seaborn theme."""
    sns.set_theme(style="white")
    mpl.rcParams.update({
        "figure.figsize": (8.5, 4.8), "figure.dpi": 110, "savefig.dpi": 160,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "legend.frameon": False, "axes.titleweight": "bold",
    })  # fmt: skip


def headline(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned finding-style title with an optional muted subtitle (offsets in points)."""
    ax.set_title("")
    kw = {"xycoords": "axes fraction", "textcoords": "offset points", "ha": "left", "va": "bottom"}
    ax.annotate(title, (0, 1), xytext=(0, 24 if subtitle else 8), fontsize=13.5,
                fontweight="bold", color=INK, **kw)  # fmt: skip
    if subtitle:
        ax.annotate(subtitle, (0, 1), xytext=(0, 8), fontsize=9.5, color=MUTED, **kw)


def save_fig(fig, name: str) -> Path:
    """Save to ``reports/figures/<name>.png`` (scratch dir in smoke mode)."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_model_comparison(table: pd.DataFrame, baselines: tuple[str, ...] = ()) -> plt.Figure:
    """Forest plot of mean CV RMSLE with corrected 95% CI, best model on top.

    Models whose mean is more than 2x the best (e.g. the dummy) are drawn as an
    off-scale label so they do not squash the interesting differences.
    """
    t = table.sort_values("rmsle", ascending=False).reset_index(drop=True)
    best = t["rmsle"].idxmin()
    tied = int((t["p_vs_best"].isna() | (t["p_vs_best"] > 0.05)).sum())
    near = t[t["rmsle"] < 2 * t.loc[best, "rmsle"]]
    lo, hi = near["ci_low"].min(), near["ci_high"].max()
    pad = 0.08 * (hi - lo)
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 1.9))
    for i, r in t.iterrows():
        colour = TEAL if i == best else (GREY if r["model"] in baselines else BLUE)
        if r.rmsle >= 2 * t.loc[best, "rmsle"]:
            ax.text(hi + pad, i, f"{r.rmsle:.4f}  (off scale)", va="center", ha="right",
                    fontsize=9.5, color=MUTED)  # fmt: skip
            continue
        ax.plot([r.ci_low, r.ci_high], [i, i], color=colour, lw=3, solid_capstyle="round")
        ax.plot(r.rmsle, i, "o", color=colour, ms=8, mec="white", mew=1.5)
        ax.text(r.ci_high + 0.002, i, f"{r.rmsle:.4f}", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks(range(len(t)), t["model"])
    ax.set_xlabel("RMSLE, repeated 5-fold CV (lower is better)")
    ax.grid(axis="y", visible=False)
    title = f"{t.loc[best, 'model']} has the lowest error, but {tied} models are statistically tied"
    headline(
        ax, title, "Mean RMSLE with 95% corrected-resampling CI; tied = paired p > 0.05 vs best"
    )
    return fig


def plot_pred_vs_actual(y_true, y_pred, title: str, subtitle: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = [min(np.min(y_true), np.min(y_pred)) * 0.9, max(np.max(y_true), np.max(y_pred)) * 1.1]
    ax.plot(lim, lim, color=GREY, lw=1.2, ls="--", zorder=1)
    ax.scatter(y_true, y_pred, s=16, color=TEAL, alpha=0.55, edgecolor="none", zorder=2)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    ticks = [50_000, 100_000, 200_000, 400_000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        axis.set_major_formatter(usd)
        axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    headline(ax, title, subtitle)
    return fig


def plot_importance(
    imp: pd.DataFrame, title: str, subtitle: str | None = None, top: int = 15, err: bool = True
) -> plt.Figure:
    """Horizontal bars; ``imp`` has columns ``feature``, ``importance`` (and ``std``)."""
    d = imp.sort_values("importance", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.6))
    ax.barh(
        d["feature"],
        d["importance"],
        color=TEAL,
        height=0.68,
        xerr=d["std"] if err and "std" in d else None,
        error_kw={"ecolor": MUTED, "lw": 1},
    )
    ax.grid(axis="y", visible=False)
    headline(ax, title, subtitle)
    return fig


def plot_learning_curve(sizes, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    """``curves[name] = (train_rmsle, valid_rmsle)`` each of shape (n_sizes, n_folds)."""
    fig, ax = plt.subplots()
    for (name, (tr, va)), colour in zip(curves.items(), PALETTE, strict=False):
        for arr, ls, lab in ((tr, "--", "train"), (va, "-", "validation")):
            m, s = arr.mean(1), arr.std(1)
            ax.plot(sizes, m, ls, color=colour, lw=2, label=f"{name} ({lab})")
            if lab == "validation":
                ax.fill_between(sizes, m - s, m + s, color=colour, alpha=0.15)
    ax.set_xlabel("Training rows")
    ax.set_ylabel("RMSLE")
    ax.legend(ncol=2, fontsize=9)
    return fig


def plot_segment_errors(
    seg: pd.DataFrame, title: str, subtitle: str | None = None, overall: float | None = None
) -> plt.Figure:
    """Bars of RMSLE per segment (``segment``, ``rmsle``, ``n``)."""
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(seg) + 1.6))
    colours = [AMBER if overall and v > overall * 1.25 else TEAL for v in seg["rmsle"]]
    ax.barh(seg["segment"].astype(str), seg["rmsle"], color=colours, height=0.66)
    for i, (v, n) in enumerate(zip(seg["rmsle"], seg["n"], strict=True)):
        ax.text(v + 0.002, i, f"{v:.3f}  (n={n})", va="center", fontsize=9, color=INK)
    if overall:
        ax.axvline(overall, color=INK, lw=1, ls="--")
        ax.text(overall, len(seg) - 0.4, " overall", fontsize=9, color=INK, va="bottom")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSLE")
    headline(ax, title, subtitle)
    return fig


def plot_intervals(
    y_true, lower, upper, pred, title: str, subtitle: str | None = None
) -> plt.Figure:
    """Hold-out houses sorted by predicted price with their interval; misses in amber."""
    y_true, lower, upper, pred = map(np.asarray, (y_true, lower, upper, pred))
    order = np.argsort(pred)
    x = np.arange(len(order))
    inside = ((y_true >= lower) & (y_true <= upper))[order]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.fill_between(
        x,
        lower[order],
        upper[order],
        color=TEAL_LIGHT,
        alpha=0.7,
        lw=0,
        label="prediction interval",
    )
    ax.plot(x, pred[order], color=TEAL, lw=1.4, label="prediction")
    ax.scatter(
        x[inside], y_true[order][inside], s=11, color=INK, zorder=3, label="actual (covered)"
    )
    ax.scatter(
        x[~inside],
        y_true[order][~inside],
        s=26,
        color=AMBER,
        zorder=4,
        edgecolor="white",
        linewidth=0.6,
        label="actual (missed)",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        mpl.ticker.FixedLocator([50_000, 100_000, 200_000, 400_000, 800_000])
    )
    ax.yaxis.set_major_formatter(usd)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Hold-out houses, sorted by predicted price")
    ax.set_ylabel("Sale price")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    headline(ax, title, subtitle)
    return fig


def fig_headline(fig, title: str, subtitle: str | None = None, y: float = 1.04) -> None:
    """Figure-level version of :func:`headline` for multi-panel figures."""
    fig.text(0.0, y + (0.06 if subtitle else 0), title, fontsize=13.5, fontweight="bold",
             color=INK, ha="left", va="bottom")  # fmt: skip
    if subtitle:
        fig.text(0.0, y, subtitle, fontsize=9.5, color=MUTED, ha="left", va="bottom")
