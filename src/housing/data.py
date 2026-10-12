"""Data access: download, cache, validate and split the Ames housing data.

Source: OpenML dataset ``house_prices`` (id 42165), a no-login mirror of the
Ames housing data used in the Kaggle "House Prices" competition. See
``docs/DATA_CARD.md`` for provenance and licence notes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------
N_ROWS = 1460
N_FEATURES = 79  # 81 columns minus Id and SalePrice

NUMERIC_COLUMNS: tuple[str, ...] = (
    "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold",
)  # fmt: skip

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
    "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
    "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition",
)  # fmt: skip

FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


class DataContractError(ValueError):
    """Raised when a dataframe does not satisfy the expected schema."""


def validate_raw(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Check that ``df`` follows the raw Ames schema and return it unchanged.

    Raises
    ------
    DataContractError
        On missing columns, wrong dtypes kind, or an invalid target.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"missing feature columns: {missing}")
    bad_numeric = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_numeric:
        raise DataContractError(f"columns expected numeric but are not: {bad_numeric}")
    if require_target:
        if config.TARGET not in df.columns:
            raise DataContractError(f"target column {config.TARGET!r} is missing")
        y = df[config.TARGET]
        if y.isna().any() or (y <= 0).any():
            raise DataContractError("target must be strictly positive and never missing")
    return df


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------
def download_data(force: bool = False, path: Path | None = None) -> Path:
    """Fetch the Ames data from OpenML (no login needed) and cache it as parquet.

    Parameters
    ----------
    force
        Re-download even when the cache file exists.
    path
        Destination file; defaults to ``data/ames_house_prices.parquet``.
    """
    from sklearn.datasets import fetch_openml

    path = Path(path) if path else config.DATA_FILE
    if path.exists() and not force:
        log.info("data already cached at %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading OpenML dataset %r (id %s)", config.OPENML_NAME, config.OPENML_ID)
    bunch = fetch_openml(
        data_id=config.OPENML_ID,
        as_frame=True,
        data_home=str(path.parent / ".openml_cache"),
    )
    df = bunch.frame.copy()
    # Normalise string columns to plain objects so the parquet round-trip is stable.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    validate_raw(df)
    df.to_parquet(path, index=False)
    return path


def load_data(path: Path | None = None, *, download: bool = True) -> pd.DataFrame:
    """Load the cached raw dataframe (downloading it first if necessary).

    Missing values are left exactly as delivered: in Ames a NaN in a
    categorical column usually means "feature absent", which is handled
    downstream in :mod:`housing.features`, never here.
    """
    path = Path(path) if path else config.DATA_FILE
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; run `make data`")
        download_data(path=path)
    df = pd.read_parquet(path)
    return validate_raw(df)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into the 79 feature columns (no ``Id``) and the price target."""
    return df.loc[:, list(FEATURE_COLUMNS)].copy(), df[config.TARGET].astype(float)


def train_holdout_split(
    df: pd.DataFrame,
    *,
    holdout_fraction: float = config.HOLDOUT_FRACTION,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/holdout split, stratified on price quintiles.

    The holdout is touched exactly once, for the final model's headline number.
    """
    strata = pd.qcut(df[config.TARGET], 5, labels=False)
    train, hold = train_test_split(
        df, test_size=holdout_fraction, random_state=seed, stratify=strata
    )
    return train.reset_index(drop=True), hold.reset_index(drop=True)


def smoke_subsample(df: pd.DataFrame, n: int = 400, seed: int = config.SEED) -> pd.DataFrame:
    """Row subsample used by the CI notebook smoke test."""
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
