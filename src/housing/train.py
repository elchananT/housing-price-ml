"""Train the final model, evaluate it once on the hold-out set, and save the bundle.

    python -m housing.train            # or: make train

Protocol
--------
1. Stratified 80/20 train/hold-out split (seeded).
2. Fit on the 80 % -> hold-out metrics (the unbiased headline numbers) and interval coverage.
3. Refit on 100 % of the data -> the model that is saved and shipped. Its hold-out score is
   therefore *optimistic by construction*; the numbers in the metadata refer to step 2.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost

from . import __version__, config
from .data import FEATURE_COLUMNS, load_data, split_xy, train_holdout_split
from .evaluation import bootstrap_ci, regression_metrics
from .inference import HousingModel
from .interval import ConformalQuantileInterval, coverage_report
from .models import build
from .tune import load_params

FINAL_MODEL = "Blend (Ridge + LightGBM + XGBoost)"
INTERVAL_ALPHA = 0.1


def train_and_evaluate(model_name: str = FINAL_MODEL, lgbm_params: dict | None = None):
    """Run protocol steps 1-2 and return fitted artefacts plus metrics."""
    df = load_data()
    train, hold = train_holdout_split(df)
    X_tr, y_tr = split_xy(train)
    X_ho, y_ho = split_xy(hold)

    t0 = time.perf_counter()
    point = build(model_name, lgbm_params=lgbm_params).fit(X_tr, y_tr)
    interval = ConformalQuantileInterval(INTERVAL_ALPHA, lgbm_params).fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    pred = point.predict(X_ho)
    metrics = regression_metrics(y_ho, pred)
    metrics["rmsle_ci95_bootstrap"] = list(bootstrap_ci(y_ho, pred))
    lo, hi = interval.predict_interval(X_ho)
    cov = coverage_report(y_ho, lo, hi)
    raw_lo, raw_hi = interval.predict_interval(X_ho, calibrated=False)
    cov_raw = coverage_report(y_ho, raw_lo, raw_hi)
    return {
        "point": point, "interval": interval, "metrics": metrics, "coverage": cov,
        "coverage_uncalibrated": cov_raw, "fit_seconds": fit_s,
        "n_train": len(train), "n_holdout": len(hold), "df": df,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=FINAL_MODEL)
    args = ap.parse_args(argv)

    params = load_params()
    print(f"tuned LightGBM params: {'found' if params else 'not found, using defaults'}")
    res = train_and_evaluate(args.model, params)
    print(json.dumps({"holdout": res["metrics"], "interval_coverage": res["coverage"]}, indent=2))

    # Step 3: refit on everything for the shipped artefact.
    X_all, y_all = split_xy(res["df"])
    t0 = time.perf_counter()
    point = build(args.model, lgbm_params=params).fit(X_all, y_all)
    interval = ConformalQuantileInterval(INTERVAL_ALPHA, params).fit(X_all, y_all)
    refit_s = time.perf_counter() - t0

    metadata = {
        "package_version": __version__,
        "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "seed": config.SEED,
        "lgbm_params": params or "defaults",
        "target_transform": "log1p",
        "interval": {
            "method": "conformalised quantile regression (LightGBM)",
            "nominal_coverage": 1 - INTERVAL_ALPHA,
            "q_log": res["interval"].q_,
        },
        "n_rows_train_split": res["n_train"],
        "n_rows_holdout": res["n_holdout"],
        "n_rows_shipped_model": len(X_all),
        "holdout_metrics_of_80pct_model": res["metrics"],
        "holdout_interval_coverage": res["coverage"],
        "fit_seconds_80pct": round(res["fit_seconds"], 1),
        "refit_seconds_100pct": round(refit_s, 1),
        "features": list(FEATURE_COLUMNS),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = HousingModel(point=point, interval=interval, metadata=metadata)
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    size_mb = config.MODEL_PATH.stat().st_size / 1e6
    print(f"saved {config.MODEL_PATH} ({size_mb:.2f} MB) and {config.MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
