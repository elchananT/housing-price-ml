import numpy as np
import pandas as pd

from housing import data
from housing.features import NONE_LEVEL, AmesFeatureBuilder, SkewedLog1p, make_preprocessor


def _features(df):
    X, _ = data.split_xy(df)
    return AmesFeatureBuilder().fit(X).transform(X)


def test_structural_absence_becomes_none_level(raw):
    out = AmesFeatureBuilder().transform(data.split_xy(raw)[0])
    no_pool = raw["PoolArea"] == 0
    assert (out.loc[no_pool, "PoolQC"] == 0).all()  # ordinal 0 == "None"
    assert (
        out["Alley"].notna().all()
        and (out["Alley"] == NONE_LEVEL).sum() == raw["Alley"].isna().sum()
    )
    assert out.loc[raw["GarageArea"] == 0, "HasGarage"].eq(0).all()


def test_genuine_missing_is_not_disguised_as_absence(raw):
    """Row with a basement but NaN BsmtExposure is a data gap -> must stay NaN (imputed in-fold)."""
    out = _features(raw)
    gap = raw["BsmtExposure"].isna() & (raw["TotalBsmtSF"] > 0)
    assert gap.sum() == 1
    assert out.loc[gap, "BsmtExposure"].isna().all()
    assert out["MasVnrType"].isna().sum() == raw["MasVnrType"].isna().sum() == 8


def test_builder_is_row_wise_so_batch_composition_cannot_leak(raw):
    X, _ = data.split_xy(raw)
    b = AmesFeatureBuilder().fit(X)
    full = b.transform(X)
    part = b.transform(X.iloc[10:40])
    pd.testing.assert_frame_equal(full.iloc[10:40], part)


def test_engineered_features_are_consistent(raw):
    out = _features(raw)
    assert (out["TotalSF"] == raw["TotalBsmtSF"] + raw["1stFlrSF"] + raw["2ndFlrSF"]).all()
    assert (out["HouseAge"] >= 0).all() and (out["RemodAge"] >= 0).all()
    assert "GarageYrBlt" not in out and "Id" not in out
    assert out["MSSubClass"].map(type).eq(str).all()


def test_skewed_log1p_learns_columns_from_fit_data_only():
    rng = np.random.default_rng(0)
    train = np.column_stack([rng.lognormal(0, 1.5, 500), rng.normal(5, 1, 500)])
    t = SkewedLog1p().fit(train)
    assert t.mask_.tolist() == [True, False]
    new = np.array([[3.0, 3.0]])
    np.testing.assert_allclose(t.transform(new), [[np.log1p(3.0), 3.0]])


def test_preprocessor_statistics_come_from_training_rows_only(raw):
    """Imputation medians must equal the *train* medians, not the full-data ones."""
    X, y = data.split_xy(raw)
    tr = X.iloc[:300]
    prep = make_preprocessor("linear").fit(tr, np.log1p(y.iloc[:300]))
    imputer = prep["columns"].named_transformers_["num"]["impute"]
    cols = list(prep["columns"].transformers_[0][2])
    stats = dict(zip(cols, imputer.statistics_, strict=True))
    feats_tr = prep["features"].transform(tr)
    feats_all = prep["features"].transform(X)
    assert stats["LotFrontage"] == feats_tr["LotFrontage"].median()
    assert feats_tr["LotFrontage"].median() != feats_all["LotFrontage"].median()


def test_preprocessor_outputs_finite_matrix_and_handles_unseen_levels(raw):
    X, y = data.split_xy(raw)
    for kind in ("linear", "tree"):
        prep = make_preprocessor(kind).fit(X.iloc[:400], np.log1p(y.iloc[:400]))
        Z = prep.transform(X.iloc[400:])
        assert np.isfinite(np.asarray(Z, dtype=float)).all()
    odd = X.iloc[[0]].copy()
    odd["Neighborhood"] = "Atlantis"
    assert np.isfinite(np.asarray(prep.transform(odd), dtype=float)).all()
