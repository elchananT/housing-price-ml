"""Feature engineering as scikit-learn transformers.

Leakage policy
--------------
* :class:`AmesFeatureBuilder` is *stateless*: every value it creates is a
  row-wise function of that row alone (no statistic is learned from data).
* Everything that learns from data (imputation values, skew detection, scaling,
  category vocabularies, target encoding) lives in the ``ColumnTransformer``
  returned by :func:`make_preprocessor` and is therefore fit **only on the
  training portion of each CV fold** when used inside a ``Pipeline``.

Missingness policy
------------------
In Ames a NaN in many categorical columns means *"this house has no such
feature"* (no pool, no garage, no basement, ...), not "unknown". We encode that
as an explicit ``"None"`` level, but only when a companion numeric column
confirms the absence (e.g. ``GarageArea == 0``). NaNs that remain after this
rule are genuinely missing and are imputed inside the CV folds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from . import config

NONE_LEVEL = "None"

# column -> predicate telling us that the NaN really means "feature absent".
_Rule = Callable[[pd.DataFrame], pd.Series]
ABSENCE_RULES: dict[str, _Rule] = {
    **{c: (lambda d: d["GarageArea"] == 0) for c in
       ("GarageType", "GarageFinish", "GarageQual", "GarageCond")},
    **{c: (lambda d: d["TotalBsmtSF"] == 0) for c in
       ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")},
    "FireplaceQu": lambda d: d["Fireplaces"] == 0,
    "PoolQC": lambda d: d["PoolArea"] == 0,
    # No companion numeric column exists: NaN is documented as "None" in the data dictionary.
    "Alley": lambda d: pd.Series(True, index=d.index),
    "Fence": lambda d: pd.Series(True, index=d.index),
    "MiscFeature": lambda d: pd.Series(True, index=d.index),
}  # fmt: skip

_QUAL = {NONE_LEVEL: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    **{c: _QUAL for c in ("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                          "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC")},
    "BsmtExposure": {NONE_LEVEL: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {NONE_LEVEL: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {NONE_LEVEL: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {NONE_LEVEL: 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7,
                   "Typ": 8},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}  # fmt: skip


class AmesFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless, row-wise cleaning and feature engineering for raw Ames rows.

    Steps: (1) structural-absence ``NaN -> "None"``; (2) order-aware encoding of
    quality-like columns; (3) ``MSSubClass`` treated as nominal; (4) engineered
    features (total area, bathrooms, ages, presence flags); (5) drop ``Id`` and the
    now-redundant ``GarageYrBlt``.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy()
        d = d.drop(columns=[c for c in (config.ID_COLUMN, config.TARGET) if c in d.columns])

        # 1. structural absence -> explicit "None" level (only where confirmed)
        for col, rule in ABSENCE_RULES.items():
            mask = d[col].isna() & rule(d)
            d[col] = d[col].astype(object).where(~mask, NONE_LEVEL)

        # 2. ordinal encodings (unknown / still-missing levels stay NaN -> imputed in-fold)
        for col, mapping in ORDINAL_MAPS.items():
            d[col] = d[col].map(mapping).astype(float)

        # 3. dwelling type is a code, not a quantity
        d["MSSubClass"] = d["MSSubClass"].astype(int).astype(str)

        # 4. engineered features
        sold = d["YrSold"]
        d["TotalSF"] = d["TotalBsmtSF"] + d["1stFlrSF"] + d["2ndFlrSF"]
        d["TotalBath"] = (
            d["FullBath"] + 0.5 * d["HalfBath"] + d["BsmtFullBath"] + 0.5 * d["BsmtHalfBath"]
        )
        d["TotalPorchSF"] = (
            d["OpenPorchSF"] + d["EnclosedPorch"] + d["3SsnPorch"] + d["ScreenPorch"]
            + d["WoodDeckSF"]
        )  # fmt: skip
        d["HouseAge"] = (sold - d["YearBuilt"]).clip(lower=0)
        d["RemodAge"] = (sold - d["YearRemodAdd"]).clip(lower=0)
        garage_age = sold - d["GarageYrBlt"]
        d["GarageAge"] = garage_age.where(garage_age >= 0)  # NaN (-> imputed) if no garage
        d["IsRemodeled"] = (d["YearRemodAdd"] != d["YearBuilt"]).astype(int)
        d["HasGarage"] = (d["GarageArea"] > 0).astype(int)
        d["HasBasement"] = (d["TotalBsmtSF"] > 0).astype(int)
        d["Has2ndFloor"] = (d["2ndFlrSF"] > 0).astype(int)
        d["HasPool"] = (d["PoolArea"] > 0).astype(int)
        d["HasFireplace"] = (d["Fireplaces"] > 0).astype(int)
        d["IsNewSale"] = (d["YrSold"] == d["YearBuilt"]).astype(int)

        # 5. redundant / identifier columns
        return d.drop(columns=["GarageYrBlt"])


class SkewedLog1p(BaseEstimator, TransformerMixin):
    """``log1p`` for non-negative columns whose skewness exceeds ``threshold``.

    The set of columns is decided in ``fit`` (training fold only), so the rule
    cannot be influenced by validation rows.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        skew = X.skew()
        nonneg = (X.min() >= 0).to_numpy()
        self.mask_ = ((skew.abs() > self.threshold).to_numpy()) & nonneg
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        out = np.asarray(X, dtype=float).copy()
        out[:, self.mask_] = np.log1p(out[:, self.mask_])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(kind: str = "linear", cat_encoding: str | None = None) -> Pipeline:
    """Feature builder + learned preprocessing.

    Parameters
    ----------
    kind
        ``"linear"``: skew-corrected, standardised numerics + one-hot categoricals.
        ``"tree"``: median-imputed numerics + an encoding chosen by ``cat_encoding``.
    cat_encoding
        For ``kind="tree"``: ``"target"`` (cross-fitted target encoding, default),
        ``"ordinal"`` or ``"onehot"``.
    """
    num_sel = make_column_selector(dtype_include="number")
    cat_sel = make_column_selector(dtype_exclude="number")

    if kind == "linear":
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("skew", SkewedLog1p()),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist", min_frequency=0.01,
                        sparse_output=False,
                    ),
                ),
            ]
        )  # fmt: skip
    elif kind == "tree":
        cat_encoding = cat_encoding or "target"
        numeric = SimpleImputer(strategy="median")
        encoders = {
            "target": TargetEncoder(smooth="auto", random_state=config.SEED),
            "ordinal": OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
            ),
            "onehot": OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False
            ),
        }
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", encoders[cat_encoding]),
            ]
        )
    else:
        raise ValueError(f"unknown preprocessor kind: {kind!r}")

    columns = ColumnTransformer(
        [("num", numeric, num_sel), ("cat", categorical, cat_sel)],
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", AmesFeatureBuilder()), ("columns", columns)])
