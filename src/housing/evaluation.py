"""Evaluation: metrics, repeated K-fold CV with honest confidence intervals."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold

from . import config


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmsle(y_true, y_pred) -> float:
    """Root mean squared logarithmic error = RMSE of log1p prices (the Kaggle metric)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(np.asarray(y_true))) ** 2)))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSLE plus metrics in dollars and percent, for human interpretation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    ss_res = np.sum((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    ss_tot = np.sum((np.log1p(y_true) - np.log1p(y_true).mean()) ** 2)
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae_usd": float(np.mean(np.abs(y_pred - y_true))),
        "rmse_usd": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mdape_pct": float(100 * np.median(ape)),
        "mape_pct": float(100 * np.mean(ape)),
        "r2_log": float(1 - ss_res / ss_tot),
    }


def bootstrap_ci(
    y_true, y_pred, metric=rmsle, n_boot: int = 2000, level: float = 0.95, seed: int = config.SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``metric`` over rows (for a single hold-out set)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Repeated K-fold cross-validation
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    """Per-fold scores and out-of-fold predictions of one model."""

    name: str
    folds: pd.DataFrame  # one row per (repeat, fold)
    oof: np.ndarray  # OOF prediction averaged over repeats (log-mean in price space)
    n_train: int
    n_test: int

    @property
    def rmsle(self) -> np.ndarray:
        return self.folds["rmsle"].to_numpy()


def _fit_score(model, X, y, train_idx, test_idx):
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_time = time.perf_counter() - t0
    pred = m.predict(X.iloc[test_idx])
    return test_idx, pred, {**regression_metrics(y.iloc[test_idx], pred), "fit_s": fit_time}


def repeated_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    name: str = "model",
    n_splits: int = config.CV_FOLDS,
    n_repeats: int = config.CV_REPEATS,
    seed: int = config.SEED,
    n_jobs: int = -1,
) -> CVResult:
    """Repeated K-fold CV. All fitting (including preprocessing) happens inside each fold."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(cv.split(X))
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_score)(model, X, y, tr, te) for tr, te in splits)
    rows, log_sum, count = [], np.zeros(len(X)), np.zeros(len(X))
    for k, (idx, pred, metrics) in enumerate(out):
        rows.append({"repeat": k // n_splits, "fold": k % n_splits, **metrics})
        log_sum[idx] += np.log1p(pred)
        count[idx] += 1
    oof = np.expm1(log_sum / count)
    n_test = len(X) // n_splits
    return CVResult(name, pd.DataFrame(rows), oof, len(X) - n_test, n_test)


def corrected_ci(
    scores: np.ndarray, n_train: int, n_test: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean and CI using the Nadeau-Bengio corrected resampled variance.

    Scores from overlapping training sets are positively correlated, so the naive
    ``s / sqrt(n)`` standard error is too small. The correction multiplies the
    variance term ``1/n`` by ``(1 + n_test/n_train * n)``, i.e. uses
    ``(1/n + n_test/n_train) * s^2``.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    var = scores.var(ddof=1) * (1 / n + n_test / n_train)
    half = stats.t.ppf(0.5 + level / 2, df=n - 1) * np.sqrt(var)
    return float(scores.mean()), float(scores.mean() - half), float(scores.mean() + half)


def paired_corrected_pvalue(a: CVResult, b: CVResult) -> float:
    """Two-sided corrected resampled t-test on paired per-fold RMSLE differences."""
    d = a.rmsle - b.rmsle
    n = len(d)
    if np.allclose(d.std(ddof=1), 0):
        return 1.0
    t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / n + a.n_test / a.n_train))
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def comparison_table(results: dict[str, CVResult], reference: str | None = None) -> pd.DataFrame:
    """Leaderboard sorted by mean RMSLE, with corrected 95% CI and paired p-value vs best."""
    rows = []
    for name, r in results.items():
        mean, lo, hi = corrected_ci(r.rmsle, r.n_train, r.n_test)
        rows.append({
            "model": name, "rmsle": mean, "ci_low": lo, "ci_high": hi,
            "rmsle_std_folds": r.rmsle.std(ddof=1),
            "mae_usd": r.folds["mae_usd"].mean(), "mdape_pct": r.folds["mdape_pct"].mean(),
            "r2_log": r.folds["r2_log"].mean(), "fit_s": r.folds["fit_s"].mean(),
        })  # fmt: skip
    table = pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)
    ref = reference or table.loc[0, "model"]
    table["p_vs_best"] = [
        np.nan if m == ref else paired_corrected_pvalue(results[m], results[ref])
        for m in table["model"]
    ]
    return table
