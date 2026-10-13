# Data card: Ames housing (OpenML `house_prices`)

| | |
|---|---|
| **Name** | Ames Housing data, as mirrored on OpenML: `house_prices`, dataset id 42165, version 1 |
| **Original author** | Dean De Cock, "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project", *Journal of Statistics Education* 19(3), 2011 |
| **Access** | `sklearn.datasets.fetch_openml(data_id=42165)`, no login; `housing.data.download_data()` caches it as `data/ames_house_prices.parquet` (git-ignored) |
| **Size** | 1,460 rows x 81 columns (`Id`, 79 features, `SalePrice`); this is the labelled half of the Kaggle "House Prices" split, not the full 2,930-row Ames table |
| **Content** | Residential sales in Ames, Iowa, sold 2006-2010: 36 numeric and 43 categorical descriptors of size, quality, age, location, and sale circumstances |
| **Target** | `SalePrice` in USD (34,900 to 755,000, median 163,000, skew 1.88) |
| **Licence** | OpenML lists the licence as "NA" (unspecified). The data is published for educational use, and Kaggle's competition terms apply to its Kaggle copy. **This repository does not redistribute the data**; it downloads it on demand. Check the terms before commercial use. |

## Known properties (verified in `notebooks/01_eda.ipynb`)

* **Missingness is mostly structural.** 6,965 missing cells in 19 columns; 6,687 (96 %) mean "feature absent" (confirmed by `PoolArea`, `GarageArea`, `TotalBsmtSF`, `Fireplaces`; Alley/Fence/MiscFeature documented as "None"). 278 are real gaps: `LotFrontage` 259, `MasVnrType`/`MasVnrArea` 8 each, `Electrical` 1, and one row each in `BsmtExposure`, `BsmtFinType2`.
* **Outliers**: four houses above 4,000 sq ft; two are partial sales sold far below what their size suggests (Ids 524, 1299). Kept in all evaluations.
* **No duplicates**; some category labels differ from the official documentation (e.g. `C (all)`, `Twnhs`); the schema in `housing/schema.py` follows the data.
* **Coverage limits**: one city, one five-year window spanning the 2008 crash, no coordinates, no listing text.
* **Sensitive attributes**: none recorded, but `Neighborhood` can proxy for socio-economic patterns.
