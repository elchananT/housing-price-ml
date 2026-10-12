"""Model zoo.

Every model is ``TransformedTargetRegressor(Pipeline(preprocess, regressor))``:
the regressor is trained on ``log1p(price)`` (see notebook 01 for why) and the
whole object predicts **dollars**, so downstream code never has to remember to
invert the transform.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor

from . import config
from .features import AmesFeatureBuilder, make_preprocessor

SEED = config.SEED

LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 12,
    "min_child_samples": 12,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}


def log_target(regressor) -> TransformedTargetRegressor:
    """Fit ``regressor`` on ``log1p(y)``; predictions come back in original units."""
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def _lgbm(params: dict[str, Any] | None = None, **extra) -> LGBMRegressor:
    return LGBMRegressor(
        **{**LGBM_DEFAULTS, **(params or {}), **extra},
        random_state=SEED, n_jobs=1, verbose=-1,
    )  # fmt: skip


def _xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=700, learning_rate=0.03, max_depth=3, min_child_weight=3,
        subsample=0.7, colsample_bytree=0.5, reg_lambda=1.0,
        tree_method="hist", random_state=SEED, n_jobs=1,
    )  # fmt: skip


def _ols_three_features() -> Pipeline:
    """Deliberately naive baseline: OLS on log living area, overall quality, neighbourhood."""
    cols = ColumnTransformer(
        [
            ("area", FunctionTransformer(np.log1p), ["GrLivArea"]),
            ("quality", "passthrough", ["OverallQual"]),
            ("hood", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Neighborhood"]),
        ]
    )
    return Pipeline([("cols", cols), ("ols", LinearRegression())])


def build(name: str, *, lgbm_params: dict[str, Any] | None = None) -> TransformedTargetRegressor:
    """Instantiate a named model from the zoo (unfitted)."""
    return _FACTORIES[name](lgbm_params)


def _pipe(kind: str, reg, cat_encoding: str | None = None) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(kind, cat_encoding)), ("model", reg)])


def _blend(lgbm_params) -> VotingRegressor:
    members = [
        ("ridge", _pipe("linear", _ridge())),
        ("lgbm", _pipe("tree", _lgbm(lgbm_params))),
        ("xgb", _pipe("tree", _xgb())),
    ]
    return VotingRegressor(members, weights=[1, 1, 1])


def _enet() -> ElasticNetCV:
    return ElasticNetCV(
        l1_ratio=[0.2, 0.6, 1.0],
        alphas=np.logspace(-3.5, -0.5, 25),
        cv=5,
        max_iter=20000,
        tol=1e-3,
        random_state=SEED,
    )


def _ridge() -> RidgeCV:
    return RidgeCV(alphas=np.logspace(-1, 2.5, 30))


_FACTORIES: dict[str, Callable[[dict | None], TransformedTargetRegressor]] = {
    "Dummy (mean log-price)": lambda p: log_target(DummyRegressor(strategy="mean")),
    "OLS (area, quality, neighbourhood)": lambda p: log_target(_ols_three_features()),
    "Ridge": lambda p: log_target(_pipe("linear", _ridge())),
    "ElasticNet": lambda p: log_target(_pipe("linear", _enet())),
    "Random forest": lambda p: log_target(
        _pipe("tree", RandomForestRegressor(n_estimators=300, max_features=0.33,
                                            min_samples_leaf=2, random_state=SEED, n_jobs=1))
    ),
    "Extra trees": lambda p: log_target(
        _pipe("tree", ExtraTreesRegressor(n_estimators=300, max_features=0.5,
                                          min_samples_leaf=2, random_state=SEED, n_jobs=1))
    ),
    "HistGradientBoosting": lambda p: log_target(
        _pipe("tree", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400,
                                                    max_leaf_nodes=12, min_samples_leaf=12,
                                                    l2_regularization=1.0, random_state=SEED))
    ),
    "LightGBM": lambda p: log_target(_pipe("tree", _lgbm(p))),
    "XGBoost": lambda p: log_target(_pipe("tree", _xgb())),
    "Blend (Ridge + LightGBM + XGBoost)": lambda p: log_target(_blend(p)),
}  # fmt: skip

MODEL_NAMES: list[str] = list(_FACTORIES)
BASELINES = MODEL_NAMES[:2]
__all__ = ["AmesFeatureBuilder", "MODEL_NAMES", "BASELINES", "LGBM_DEFAULTS", "build", "log_target"]
