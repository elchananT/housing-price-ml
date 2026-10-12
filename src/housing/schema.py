"""Input schema for inference (pydantic v2), derived from the Ames data dictionary.

``CATEGORY_LEVELS`` lists every level observed in the training data (the raw data uses
a few labels that differ from the official documentation, e.g. ``"C (all)"``,
``"Twnhs"``). A test asserts that all training rows validate against this schema, so
the two cannot drift apart silently. Categorical fields that are legitimately absent
for some houses (no alley, no pool, ...) accept ``None``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Generated once from data/ames_house_prices.parquet (see tests/test_schema.py).
CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "MSZoning": ("'C (all)'", "FV", "RH", "RL", "RM"),
    "Street": ("Grvl", "Pave"),
    "Alley": ("Grvl", "Pave"),
    "LotShape": ("IR1", "IR2", "IR3", "Reg"),
    "LandContour": ("Bnk", "HLS", "Low", "Lvl"),
    "Utilities": ("AllPub", "NoSeWa"),
    "LotConfig": ("Corner", "CulDSac", "FR2", "FR3", "Inside"),
    "LandSlope": ("Gtl", "Mod", "Sev"),
    "Neighborhood": (
        "Blmngtn",
        "Blueste",
        "BrDale",
        "BrkSide",
        "ClearCr",
        "CollgCr",
        "Crawfor",
        "Edwards",
        "Gilbert",
        "IDOTRR",
        "MeadowV",
        "Mitchel",
        "NAmes",
        "NPkVill",
        "NWAmes",
        "NoRidge",
        "NridgHt",
        "OldTown",
        "SWISU",
        "Sawyer",
        "SawyerW",
        "Somerst",
        "StoneBr",
        "Timber",
        "Veenker",
    ),
    "Condition1": ("Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRAn", "RRNe", "RRNn"),
    "Condition2": ("Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRAn", "RRNn"),
    "BldgType": ("1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"),
    "HouseStyle": ("1.5Fin", "1.5Unf", "1Story", "2.5Fin", "2.5Unf", "2Story", "SFoyer", "SLvl"),
    "RoofStyle": ("Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"),
    "RoofMatl": ("ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"),
    "Exterior1st": (
        "'Wd Sdng'",
        "AsbShng",
        "AsphShn",
        "BrkComm",
        "BrkFace",
        "CBlock",
        "CemntBd",
        "HdBoard",
        "ImStucc",
        "MetalSd",
        "Plywood",
        "Stone",
        "Stucco",
        "VinylSd",
        "WdShing",
    ),
    "Exterior2nd": (
        "'Brk Cmn'",
        "'Wd Sdng'",
        "'Wd Shng'",
        "AsbShng",
        "AsphShn",
        "BrkFace",
        "CBlock",
        "CmentBd",
        "HdBoard",
        "ImStucc",
        "MetalSd",
        "Other",
        "Plywood",
        "Stone",
        "Stucco",
        "VinylSd",
    ),
    "MasVnrType": ("BrkCmn", "BrkFace", "None", "Stone"),
    "ExterQual": ("Ex", "Fa", "Gd", "TA"),
    "ExterCond": ("Ex", "Fa", "Gd", "Po", "TA"),
    "Foundation": ("BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"),
    "BsmtQual": ("Ex", "Fa", "Gd", "TA"),
    "BsmtCond": ("Fa", "Gd", "Po", "TA"),
    "BsmtExposure": ("Av", "Gd", "Mn", "No"),
    "BsmtFinType1": ("ALQ", "BLQ", "GLQ", "LwQ", "Rec", "Unf"),
    "BsmtFinType2": ("ALQ", "BLQ", "GLQ", "LwQ", "Rec", "Unf"),
    "Heating": ("Floor", "GasA", "GasW", "Grav", "OthW", "Wall"),
    "HeatingQC": ("Ex", "Fa", "Gd", "Po", "TA"),
    "CentralAir": ("N", "Y"),
    "Electrical": ("FuseA", "FuseF", "FuseP", "Mix", "SBrkr"),
    "KitchenQual": ("Ex", "Fa", "Gd", "TA"),
    "Functional": ("Maj1", "Maj2", "Min1", "Min2", "Mod", "Sev", "Typ"),
    "FireplaceQu": ("Ex", "Fa", "Gd", "Po", "TA"),
    "GarageType": ("2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd"),
    "GarageFinish": ("Fin", "RFn", "Unf"),
    "GarageQual": ("Ex", "Fa", "Gd", "Po", "TA"),
    "GarageCond": ("Ex", "Fa", "Gd", "Po", "TA"),
    "PavedDrive": ("N", "P", "Y"),
    "PoolQC": ("Ex", "Fa", "Gd"),
    "Fence": ("GdPrv", "GdWo", "MnPrv", "MnWw"),
    "MiscFeature": ("Gar2", "Othr", "Shed", "TenC"),
    "SaleType": ("COD", "CWD", "Con", "ConLD", "ConLI", "ConLw", "New", "Oth", "WD"),
    "SaleCondition": ("Abnorml", "AdjLand", "Alloca", "Family", "Normal", "Partial"),
}

# Columns whose name is not a valid Python identifier get an internal name + alias.
_ALIASES = {"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThreeSeasonPorch"}

# (ge, le) sanity bounds; deliberately wider than the training range where sensible.
_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "MSSubClass": (20, 190), "LotFrontage": (0, 1000), "LotArea": (100, 1_000_000),
    "OverallQual": (1, 10), "OverallCond": (1, 10), "YearBuilt": (1800, 2100),
    "YearRemodAdd": (1800, 2100), "GarageYrBlt": (1800, 2100), "MoSold": (1, 12),
    "YrSold": (1990, 2100), "BsmtFullBath": (0, 10), "BsmtHalfBath": (0, 10),
    "FullBath": (0, 10), "HalfBath": (0, 10), "BedroomAbvGr": (0, 20), "KitchenAbvGr": (0, 10),
    "TotRmsAbvGrd": (0, 40), "Fireplaces": (0, 10), "GarageCars": (0, 10),
    "GrLivArea": (100, 20_000), "1stFlrSF": (100, 20_000),
}  # fmt: skip

# Numeric fields that may legitimately be NaN in the source data.
_NULLABLE_NUMERIC = {"LotFrontage", "MasVnrArea", "GarageYrBlt"}
# Categorical fields that are absent for some houses, or missing in a handful of rows.
_NULLABLE_CATEGORICAL = {
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "Electrical", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature",
}  # fmt: skip


def _fields() -> dict:
    fields: dict = {}
    for col in NUMERIC_COLUMNS:
        ge, le = _BOUNDS.get(col, (0, None))
        typ = float if col not in {"MSSubClass"} else int
        if col in _NULLABLE_NUMERIC:
            fields[_ALIASES.get(col, col)] = (
                typ | None,
                Field(default=None, ge=ge, le=le, alias=col),
            )
        else:
            fields[_ALIASES.get(col, col)] = (typ, Field(..., ge=ge, le=le, alias=col))
    for col in CATEGORICAL_COLUMNS:
        literal = Literal[CATEGORY_LEVELS[col]]  # type: ignore[valid-type]
        if col in _NULLABLE_CATEGORICAL:
            fields[col] = (literal | None, Field(default=None, alias=col))
        else:
            fields[col] = (literal, Field(..., alias=col))
    return fields


HouseFeatures = create_model(
    "HouseFeatures",
    __config__=ConfigDict(populate_by_name=True, extra="forbid"),
    __doc__="One house, using the original Ames column names (see docs/DATA_CARD.md).",
    **_fields(),
)  # fmt: skip


class PricePrediction(BaseModel):
    """Model output for one house."""

    price: float = Field(description="Point estimate of the sale price in USD")
    lower: float = Field(description="Lower end of the prediction interval, USD")
    upper: float = Field(description="Upper end of the prediction interval, USD")
    interval_level: float = Field(description="Nominal coverage of [lower, upper], e.g. 0.9")
