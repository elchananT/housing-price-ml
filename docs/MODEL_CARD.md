# Model card: Ames house-price regressor

## Summary
Blend (unweighted mean in log space) of Ridge, LightGBM (Optuna-tuned) and XGBoost, each in its own preprocessing pipeline, trained on `log1p(SalePrice)`; predictions are in USD. A 90 % prediction interval comes from Mondrian conformal calibration (separate margin for normal / non-normal sales). Artifact: `models/housing_model.joblib` (0.5 MB; scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, pandas 3.0.6, Python 3.12; see `models/model_metadata.json`). Load only from a source you trust (joblib uses pickle).

## Intended use
Educational/portfolio demonstration of a rigorous tabular-regression workflow; rough price estimates for houses resembling the Ames 2006-2010 sales. **Not** for appraisal, lending, tax assessment, insurance or any decision affecting people, and not for other cities or years.

## Data
Ames housing data (OpenML 42165), 1,460 sales; see `DATA_CARD.md`. Evaluation split: 1,168 train / 292 hold-out (stratified on price quintile, seed 42). The shipped model is refit on all 1,460 rows; its own hold-out score is therefore optimistic and is not reported. Numbers below are for the 80 % model.

## Performance (measured)
| Metric | Value |
|---|---|
| CV RMSLE, 5-fold x 3 (train split) | 0.1140 (95 % CI 0.0985-0.1295) |
| Hold-out RMSLE (n=292) | 0.1433 (95 % bootstrap CI 0.1036-0.1873); 0.1235 without one flagged partial-sale outlier |
| Hold-out median / mean abs. % error | 4.8 % / 9.1 % |
| Hold-out MAE / RMSE | \$13,652 / \$29,053 |
| Hold-out R2 on log price | 0.862 |
| 90 % interval coverage on hold-out | 90.8 % (95 % CI 86.9-93.6 %); normal sales 91.9 %, non-normal 84.1 % (n=44) |
| Median interval width | 31 % of price (about \$60k mean) |

Error by segment (notebook 03): no systematic bias, but higher error for the cheapest/priciest predicted houses, quality 1-4 and 9-10, and non-normal sales (about twice the RMSLE of normal sales). The worst 1 % of houses account for 37 % of squared error.

## Inputs and validation
79 raw Ames columns, validated by `housing.schema.HouseFeatures` (types, ranges, known category levels; unknown fields rejected). Years outside 1990-2100 are rejected; years outside 2006-2010 are accepted but are extrapolation.

## Ethical and fairness considerations
Neighborhood is among the strongest predictors and can encode socio-economic and historical segregation patterns. No fairness analysis was done; do not deploy for consequential decisions without one.

## Caveats
Random (not temporal) split; small hold-out; Optuna tuned only LightGBM; selecting the interval method after seeing hold-out results is mild selection (calibration itself never used the hold-out); conformal coverage is marginal and approximate; importances are associative.

## Reproduce
`make setup data test notebooks`; `python -m housing.train` retrains and rewrites the artifact and metadata.
