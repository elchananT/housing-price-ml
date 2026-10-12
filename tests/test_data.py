import pandas as pd
import pytest

from housing import config, data


def test_shape_and_columns(raw):
    assert raw.shape == (data.N_ROWS, data.N_FEATURES + 2)
    assert set(data.FEATURE_COLUMNS) <= set(raw.columns)
    assert len(data.FEATURE_COLUMNS) == data.N_FEATURES
    assert raw[config.ID_COLUMN].is_unique


def test_dtypes_and_target(raw):
    for col in data.NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(raw[col]), col
    assert raw[config.TARGET].between(10_000, 1_000_000).all()
    assert raw[config.TARGET].notna().all()


def test_missingness_is_only_where_documented(raw):
    """NaN must be confined to columns where Ames documents 'feature absent' or a few true gaps."""
    with_nan = set(raw.columns[raw.isna().any()])
    expected = {
        "LotFrontage", "Alley", "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Electrical", "FireplaceQu",
        "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PoolQC",
        "Fence", "MiscFeature",
    }  # fmt: skip
    assert with_nan == expected


def test_split_is_deterministic_disjoint_and_stratified(raw):
    a_tr, a_ho = data.train_holdout_split(raw)
    b_tr, b_ho = data.train_holdout_split(raw)
    pd.testing.assert_frame_equal(a_ho, b_ho)
    assert set(a_tr["Id"]).isdisjoint(a_ho["Id"])
    assert len(a_tr) + len(a_ho) == len(raw)
    assert abs(len(a_ho) / len(raw) - config.HOLDOUT_FRACTION) < 0.01
    # stratification keeps the price distribution aligned
    assert abs(a_tr["SalePrice"].median() / a_ho["SalePrice"].median() - 1) < 0.05


def test_validate_raw_rejects_bad_frames(raw):
    with pytest.raises(data.DataContractError, match="missing feature columns"):
        data.validate_raw(raw.drop(columns=["GrLivArea"]))
    bad = raw.copy()
    bad.loc[0, "SalePrice"] = -5
    with pytest.raises(data.DataContractError, match="strictly positive"):
        data.validate_raw(bad)
    bad = raw.copy()
    bad["LotArea"] = bad["LotArea"].astype(str)
    with pytest.raises(data.DataContractError, match="numeric"):
        data.validate_raw(bad)


def test_split_xy_drops_id_and_target(raw):
    X, y = data.split_xy(raw)
    assert "Id" not in X and "SalePrice" not in X
    assert len(X) == len(y)
