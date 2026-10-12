"""Project-wide constants: paths, seeds and run-size switches."""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42
TARGET = "SalePrice"
ID_COLUMN = "Id"

# Repository root: src/housing/config.py -> parents[2]
ROOT = Path(os.environ.get("HOUSING_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# CI smoke mode: same code paths, tiny budgets, scratch output (never overwrites real results).
SMOKE = os.environ.get("HOUSING_SMOKE", "0") == "1"
FIG_DIR = REPORTS_DIR / ("_smoke/figures" if SMOKE else "figures")
RESULTS_DIR = REPORTS_DIR / ("_smoke/results" if SMOKE else "results")

OPENML_NAME = "house_prices"
OPENML_ID = 42165
DATA_FILE = DATA_DIR / "ames_house_prices.parquet"

MODEL_PATH = MODELS_DIR / "housing_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
BEST_PARAMS_PATH = RESULTS_DIR / "best_lgbm_params.json"

HOLDOUT_FRACTION = 0.2
CV_FOLDS = 5
CV_REPEATS = 2 if SMOKE else 3
