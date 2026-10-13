"""Prediction intervals via conformal prediction, two flavours.

* :class:`ConformalResidualInterval` (**shipped**): ``point * exp(+-q)`` where ``q`` is the
  conformal quantile of out-of-fold absolute log residuals, optionally computed
  separately per group (Mondrian conformal). Simple, centred on the shipped point
  prediction, and the sharpest of the methods compared in notebook 03.
* :class:`ConformalQuantileInterval`: conformalised quantile regression (CQR), kept as the
  adaptive-width alternative that notebook 03 benchmarks against.

CQR recipe (Romano, Patterson & Candes, 2019), adapted to cross-fitting:

1. Train two quantile regressors (alpha/2 and 1 - alpha/2) on ``log1p(price)``.
2. Get **out-of-fold** quantile predictions for every training row with K-fold CV.
3. Conformity score ``E_i = max(lo_i - y_i, y_i - hi_i)``; its ``ceil((n+1)(1-alpha))/n``
   empirical quantile ``q`` is the margin by which the raw band must be widened (or can
   be shrunk, if ``q < 0``) to reach the nominal coverage.
4. Refit the quantile models on all data; the interval is ``[lo - q, hi + q]``.

Coverage is then *checked* on a hold-out set that played no role in steps 1-4. The
guarantee is marginal (average over houses), not conditional on price segment, and
relies on exchangeability. Using out-of-fold scores with a refit model is an
approximation of split conformal; the empirical hold-out coverage is the real test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from . import config
from .models import _lgbm, _pipe


def _quantile_model(alpha: float, params: dict | None):
    return _pipe("tree", _lgbm(params, objective="quantile", alpha=alpha))


class ConformalQuantileInterval(BaseEstimator):
    """Marginal ``1 - alpha`` prediction intervals in price space."""

    def __init__(
        self,
        alpha: float = 0.1,
        lgbm_params: dict | None = None,
        n_splits: int = config.CV_FOLDS,
        seed: int = config.SEED,
    ):
        self.alpha = alpha
        self.lgbm_params = lgbm_params
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series):
        ylog = np.log1p(np.asarray(y, dtype=float))
        lo_m = _quantile_model(self.alpha / 2, self.lgbm_params)
        hi_m = _quantile_model(1 - self.alpha / 2, self.lgbm_params)

        oof_lo, oof_hi = np.zeros(len(X)), np.zeros(len(X))
        for tr, te in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(X):
            oof_lo[te] = clone(lo_m).fit(X.iloc[tr], ylog[tr]).predict(X.iloc[te])
            oof_hi[te] = clone(hi_m).fit(X.iloc[tr], ylog[tr]).predict(X.iloc[te])
        lo, hi = np.minimum(oof_lo, oof_hi), np.maximum(oof_lo, oof_hi)
        scores = np.maximum(lo - ylog, ylog - hi)
        self.q_ = conformal_quantile(scores, self.alpha)

        self.lo_model_ = clone(lo_m).fit(X, ylog)
        self.hi_model_ = clone(hi_m).fit(X, ylog)
        return self

    def raw_band(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Un-calibrated quantile band (log space), kept for diagnostics."""
        a, b = self.lo_model_.predict(X), self.hi_model_.predict(X)
        return np.minimum(a, b), np.maximum(a, b)

    def predict_interval(
        self, X: pd.DataFrame, calibrated: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds in dollars."""
        lo, hi = self.raw_band(X)
        if calibrated:
            lo, hi = lo - self.q_, hi + self.q_
        return np.expm1(lo), np.expm1(hi)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected ``1 - alpha`` empirical quantile of non-conformity scores."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def sale_group(X: pd.DataFrame) -> np.ndarray:
    """Mondrian groups: ordinary sales vs. non-normal ones (foreclosure, partial, family...).

    Notebook 03 shows non-normal sales are priced far less accurately, so a single pooled
    error margin under-covers them and over-covers normal sales.
    """
    return np.where(X["SaleCondition"].to_numpy() == "Normal", "normal", "non-normal")


class ConformalResidualInterval(BaseEstimator):
    """Split-free conformal intervals ``point * exp(+-q_g)`` from out-of-fold log residuals.

    Parameters
    ----------
    alpha
        Miscoverage level (0.1 -> 90 % intervals).
    group_fn
        Optional ``f(X) -> array of group labels`` for Mondrian (group-conditional)
        calibration; groups with fewer than ``min_group_size`` rows fall back to the pooled
        margin. Must be a module-level function so the fitted object can be pickled.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        group_fn=None,
        n_splits: int = config.CV_FOLDS,
        min_group_size: int = 60,
        seed: int = config.SEED,
    ):
        self.alpha = alpha
        self.group_fn = group_fn
        self.n_splits = n_splits
        self.min_group_size = min_group_size
        self.seed = seed

    def fit(self, estimator, X: pd.DataFrame, y: pd.Series):
        """Calibrate on out-of-fold predictions of an *unfitted* point estimator."""
        ylog = np.log1p(np.asarray(y, dtype=float))
        oof = np.zeros(len(X))
        for tr, te in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(X):
            m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
            oof[te] = np.log1p(m.predict(X.iloc[te]))
        resid = np.abs(ylog - oof)
        self.q_pooled_ = conformal_quantile(resid, self.alpha)
        self.q_ = {}
        if self.group_fn is not None:
            groups = self.group_fn(X)
            for g in np.unique(groups):
                r = resid[groups == g]
                self.q_[str(g)] = (
                    conformal_quantile(r, self.alpha)
                    if len(r) >= self.min_group_size
                    else self.q_pooled_
                )
        return self

    def margins(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row log-space half-width."""
        if self.group_fn is None:
            return np.full(len(X), self.q_pooled_)
        return np.array([self.q_.get(str(g), self.q_pooled_) for g in self.group_fn(X)])

    def predict_interval(self, X: pd.DataFrame, point_pred) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds in dollars around ``point_pred``."""
        centre, q = np.log1p(np.asarray(point_pred, dtype=float)), self.margins(X)
        return np.expm1(centre - q), np.expm1(centre + q)


def abs_residual_interval(y_log: np.ndarray, oof_log_pred: np.ndarray, alpha: float = 0.1) -> float:
    """Half-width (log space) of a constant-width conformal band from OOF residuals."""
    return conformal_quantile(np.abs(np.asarray(y_log) - np.asarray(oof_log_pred)), alpha)


def coverage_report(y_true, lower, upper) -> dict[str, float]:
    """Empirical coverage and mean relative width of an interval."""
    y_true, lower, upper = map(np.asarray, (y_true, lower, upper))
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(inside.mean()),
        "mean_width_usd": float(np.mean(upper - lower)),
        "median_rel_width": float(np.median((upper - lower) / y_true)),
    }


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (used for coverage estimates)."""
    from scipy import stats

    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)
