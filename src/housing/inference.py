"""Inference: schema-validated predictions from the saved model bundle.

>>> from housing.inference import load_model, predict_one
>>> model = load_model()                       # doctest: +SKIP
>>> predict_one(house_dict, model)             # doctest: +SKIP
PricePrediction(price=..., lower=..., upper=..., interval_level=0.9)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from . import config
from .data import FEATURE_COLUMNS, NUMERIC_COLUMNS
from .interval import ConformalQuantileInterval
from .schema import HouseFeatures, PricePrediction


@dataclass
class HousingModel:
    """Everything needed at inference time, saved as one joblib file."""

    point: Any  # fitted TransformedTargetRegressor -> dollars
    interval: ConformalQuantileInterval
    metadata: dict = field(default_factory=dict)

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict for a dataframe with the 79 raw feature columns."""
        X = X.loc[:, list(FEATURE_COLUMNS)]
        lower, upper = self.interval.predict_interval(X)
        price = np.asarray(self.point.predict(X), dtype=float)
        # Guard against the (rare) case where the point forecast leaves the band.
        lower, upper = np.minimum(lower, price), np.maximum(upper, price)
        return pd.DataFrame({"price": price, "lower": lower, "upper": upper}, index=X.index)


def load_model(path: str | Path | None = None) -> HousingModel:
    """Load the bundle written by ``housing.train`` (default ``models/housing_model.joblib``)."""
    path = Path(path) if path else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make train` first")
    return joblib.load(path)


def validate_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Validate raw records against :class:`HouseFeatures` and return a model-ready frame.

    Raises ``pydantic.ValidationError`` listing the offending fields.
    """
    rows = []
    for i, rec in enumerate(records):
        try:
            rows.append(HouseFeatures.model_validate(dict(rec)).model_dump(by_alias=True))
        except ValidationError as exc:
            raise ValueError(f"record {i} is invalid:\n{exc}") from exc
    if not rows:
        raise ValueError("no records given")
    frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    frame[list(NUMERIC_COLUMNS)] = frame[list(NUMERIC_COLUMNS)].astype(float)
    return frame


def predict_batch(
    records: Iterable[Mapping[str, Any]], model: HousingModel | None = None
) -> list[PricePrediction]:
    """Validate and predict. Output order matches input order."""
    model = model or load_model()
    out = model.predict_frame(validate_records(records))
    level = float(1 - model.interval.alpha)
    return [
        PricePrediction(price=r.price, lower=r.lower, upper=r.upper, interval_level=level)
        for r in out.itertuples(index=False)
    ]


def predict_one(record: Mapping[str, Any], model: HousingModel | None = None) -> PricePrediction:
    """Validate and predict a single house."""
    return predict_batch([record], model)[0]


def read_metadata(path: str | Path | None = None) -> dict:
    """Read ``models/model_metadata.json``."""
    return json.loads(Path(path or config.MODEL_METADATA_PATH).read_text())
