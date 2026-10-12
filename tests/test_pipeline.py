import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_predict

from housing import data, models
from housing.evaluation import (
    comparison_table,
    corrected_ci,
    regression_metrics,
    repeated_cv,
    rmsle,
)


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_every_model_fits_on_a_tiny_sample_and_predicts_dollars(sample, name):
    X, y = data.split_xy(sample)
    model = models.build(name).fit(X.iloc[:160], y.iloc[:160])
    pred = model.predict(X.iloc[160:])
    assert pred.shape == (len(X) - 160,)
    assert np.isfinite(pred).all()
    assert pred.min() > 10_000 and pred.max() < 2_000_000  # dollars, not log-dollars


def test_pipeline_beats_dummy_on_sample(sample):
    X, y = data.split_xy(sample)
    kw = dict(n_splits=3, n_repeats=1, n_jobs=1)
    dummy = repeated_cv(models.build("Dummy (mean log-price)"), X, y, **kw)
    ridge = repeated_cv(models.build("Ridge"), X, y, **kw)
    assert ridge.rmsle.mean() < 0.6 * dummy.rmsle.mean()


def test_no_leakage_when_target_is_pure_noise(raw):
    """If any step peeked at validation targets, a shuffled target would still be 'predictable'.

    Includes the cross-fitted TargetEncoder on high-cardinality columns - the classic leak.
    """
    X, y = data.split_xy(raw.sample(400, random_state=1))
    y_noise = y.sample(frac=1, random_state=7).reset_index(drop=True)
    X = X.reset_index(drop=True)
    for name in ("LightGBM", "Ridge"):
        model = models.build(name)
        pred = cross_val_predict(model, X, y_noise, cv=KFold(4, shuffle=True, random_state=0))
        score = rmsle(y_noise, pred)
        baseline = rmsle(y_noise, np.full(len(y_noise), np.expm1(np.log1p(y_noise).mean())))
        assert score > 0.97 * baseline, f"{name}: {score:.3f} vs dummy {baseline:.3f}"


def test_predictions_do_not_depend_on_other_rows_in_the_batch(sample):
    X, y = data.split_xy(sample)
    model = models.build("Ridge").fit(X.iloc[:150], y.iloc[:150])
    full = model.predict(X.iloc[150:])
    single = np.array([model.predict(X.iloc[[i]])[0] for i in range(150, 160)])
    np.testing.assert_allclose(full[:10], single, rtol=1e-9)


def test_metrics_and_confidence_interval_helpers():
    y = np.array([100_000.0, 200_000.0])
    assert rmsle(y, y) == 0
    assert regression_metrics(y, y)["mae_usd"] == 0
    assert rmsle(y, y * np.e) == pytest.approx(1.0, abs=0.05)
    mean, lo, hi = corrected_ci(np.array([0.10, 0.12, 0.11, 0.13, 0.09]), 800, 200)
    assert lo < mean < hi
    # corrected CI must be wider than the naive one
    naive_half = 2.776 * np.std([0.10, 0.12, 0.11, 0.13, 0.09], ddof=1) / np.sqrt(5)
    assert (hi - lo) / 2 > naive_half


def test_comparison_table_is_sorted_and_has_pvalues(sample):
    X, y = data.split_xy(sample)
    res = {
        n: repeated_cv(models.build(n), X, y, name=n, n_splits=3, n_repeats=2, n_jobs=1)
        for n in ("Dummy (mean log-price)", "Ridge")
    }
    t = comparison_table(res)
    assert t.loc[0, "model"] == "Ridge"
    assert t["rmsle"].is_monotonic_increasing
    assert np.isnan(t.loc[0, "p_vs_best"]) and t.loc[1, "p_vs_best"] < 0.05


class _DisjointnessSpy(BaseEstimator, RegressorMixin):
    """Fails loudly if it is asked to predict rows that it was fit on."""

    def fit(self, X, y):
        self.train_index_ = set(X.index)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        overlap = self.train_index_ & set(X.index)
        assert not overlap, f"{len(overlap)} validation rows were seen during fit"
        return np.full(len(X), self.mean_)


def test_cv_never_scores_a_row_it_was_fit_on(sample):
    X, y = data.split_xy(sample)
    res = repeated_cv(_DisjointnessSpy(), X, y, n_splits=4, n_repeats=2, n_jobs=1)
    assert len(res.folds) == 8
    # every row is validated exactly once per repeat
    assert np.isfinite(res.oof).all()
