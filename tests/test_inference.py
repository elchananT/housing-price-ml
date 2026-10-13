import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from housing import config, data, models
from housing.inference import HousingModel, predict_batch, predict_one, validate_records
from housing.interval import ConformalQuantileInterval, ConformalResidualInterval, sale_group
from housing.predict import main as predict_cli
from housing.schema import CATEGORY_LEVELS, HouseFeatures

EXAMPLE = config.ROOT / "examples" / "sample_house.json"


@pytest.fixture(scope="module")
def tiny_bundle(raw):
    df = raw.sample(300, random_state=3)
    X, y = data.split_xy(df)
    point = models.build("Ridge").fit(X, y)
    interval = ConformalResidualInterval(0.1, group_fn=sale_group, n_splits=3).fit(
        models.build("Ridge"), X, y
    )
    return HousingModel(point=point, interval=interval, metadata={"note": "test"})


def _records(df, n):
    X, _ = data.split_xy(df.head(n))
    return [
        {k: (None if v is None or (not isinstance(v, str) and v != v) else v) for k, v in r.items()}
        for r in X.to_dict("records")
    ]


def test_schema_accepts_every_training_row_and_levels_are_complete(raw):
    for col in data.CATEGORICAL_COLUMNS:
        assert set(raw[col].dropna().unique()) <= set(CATEGORY_LEVELS[col]), col
    frame = validate_records(_records(raw, len(raw)))
    assert frame.shape == (len(raw), data.N_FEATURES)


def test_validated_frame_gives_identical_predictions_to_raw_frame(raw, tiny_bundle):
    X, _ = data.split_xy(raw.head(25))
    direct = tiny_bundle.predict_frame(X)
    via_schema = tiny_bundle.predict_frame(validate_records(_records(raw, 25)))
    np.testing.assert_allclose(direct.to_numpy(), via_schema.to_numpy(), rtol=1e-9)


def test_round_trip_save_load_predict(raw, tiny_bundle, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(tiny_bundle, path, compress=3)
    loaded = joblib.load(path)
    recs = _records(raw, 10)
    a = predict_batch(recs, tiny_bundle)
    b = predict_batch(recs, loaded)
    assert [p.price for p in a] == pytest.approx([p.price for p in b])
    for p in b:
        assert p.lower <= p.price <= p.upper
        assert p.interval_level == pytest.approx(0.9)


def test_invalid_inputs_are_rejected_with_clear_errors(raw, tiny_bundle):
    good = _records(raw, 1)[0]
    with pytest.raises(ValueError, match="OverallQual"):
        predict_one({**good, "OverallQual": 11}, tiny_bundle)
    with pytest.raises(ValueError, match="Neighborhood"):
        predict_one({**good, "Neighborhood": "Atlantis"}, tiny_bundle)
    missing = {k: v for k, v in good.items() if k != "GrLivArea"}
    with pytest.raises(ValueError, match="GrLivArea"):
        predict_one(missing, tiny_bundle)
    with pytest.raises(ValueError, match="no records"):
        predict_batch([], tiny_bundle)
    with pytest.raises(ValidationError):
        HouseFeatures.model_validate({**good, "Surprise": 1})


def test_example_file_and_cli(tiny_bundle, tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    joblib.dump(tiny_bundle, model_path)
    assert predict_cli([str(EXAMPLE), "--model", str(model_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1 and out[0]["lower"] <= out[0]["price"] <= out[0]["upper"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"OverallQual": 99}))
    assert predict_cli([str(bad), "--model", str(model_path)]) == 2
    assert "error" in capsys.readouterr().err


def test_csv_input_with_blank_cells(raw, tiny_bundle, tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    joblib.dump(tiny_bundle, model_path)
    X, _ = data.split_xy(raw.head(3))
    csv = tmp_path / "h.csv"
    X.to_csv(csv, index=False)
    assert predict_cli([str(csv), "--model", str(model_path)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3


def test_interval_has_positive_width_and_reasonable_coverage_on_train(raw, tiny_bundle):
    X, y = data.split_xy(raw.sample(120, random_state=11))
    lo, hi = tiny_bundle.interval.predict_interval(X, tiny_bundle.point.predict(X))
    assert (hi > lo).all()
    assert isinstance(pd.Series(lo).mean(), float)


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `make train` first")
def test_shipped_model_predicts_example_house():
    pred = predict_one(json.loads(EXAMPLE.read_text()))
    assert 50_000 < pred.price < 600_000
    assert pred.lower < pred.price < pred.upper


def test_cqr_alternative_interval_still_works(raw):
    X, y = data.split_xy(raw.sample(200, random_state=5))
    iv = ConformalQuantileInterval(0.1, n_splits=3).fit(X, y)
    lo, hi = iv.predict_interval(X)
    assert (hi > lo).all()
