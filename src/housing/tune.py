"""Hyper-parameter search with Optuna (bounded runtime, seeded, TPE sampler)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from . import config
from .evaluation import repeated_cv
from .models import build


def suggest_lgbm(trial: optuna.Trial) -> dict:
    """Search space for the LightGBM regressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 32),
        "min_child_samples": trial.suggest_int("min_child_samples", 4, 40),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 120,
    timeout: float = 300.0,
    seed: int = config.SEED,
) -> tuple[optuna.Study, dict, float]:
    """Minimise 5-fold CV RMSLE. Returns ``(study, best_params, wall_seconds)``.

    The search uses a CV split seeded differently from the evaluation CV
    (``seed + 1``), which reduces - but does not remove - selection bias; the
    untouched hold-out set gives the unbiased number.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm(trial)
        res = repeated_cv(
            build("LightGBM", lgbm_params=params), X, y, n_repeats=1, seed=seed + 1, n_jobs=-1
        )
        return float(np.mean(res.rmsle))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study, dict(study.best_params), time.perf_counter() - t0


def save_params(params: dict, path: Path | None = None) -> Path:
    path = Path(path or config.BEST_PARAMS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, sort_keys=True))
    return path


def load_params(path: Path | None = None) -> dict | None:
    """Tuned params written by notebook 02 / ``tune``; ``None`` if not available."""
    path = Path(path or config.BEST_PARAMS_PATH)
    return json.loads(path.read_text()) if path.exists() else None
