# Housing price regression

A beginner-friendly notebook that downloads the Kaggle House Prices data, cleans mixed data types and missing values, trains a baseline and a tuned random-forest regressor, then exports predictions.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
unzip data/house-prices-advanced-regression-techniques.zip -d data
jupyter lab housing_price_model.ipynb
```

Create a Kaggle API token first (`~/.kaggle/kaggle.json`). The notebook expects `data/train.csv` and never fills a target-price value with a made-up number.
