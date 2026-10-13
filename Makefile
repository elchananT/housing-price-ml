.PHONY: setup data test lint notebooks notebooks-smoke train clean help
PYTHON_VERSION ?= 3.12
VENV := .venv
BIN := $(VENV)/bin
NB := notebooks/01_eda.ipynb notebooks/02_modeling.ipynb notebooks/03_error_analysis_and_explainability.ipynb
EXEC := $(BIN)/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

setup:  ## create .venv and install pinned dependencies + the package
	uv venv --python $(PYTHON_VERSION) $(VENV)
	uv pip install --python $(BIN)/python -r requirements.txt
	uv pip install --python $(BIN)/python -e . --no-deps

data:  ## download the Ames data from OpenML into data/ (no login)
	$(BIN)/python -c "from housing.data import download_data; print(download_data())"

test:  ## run the unit tests (downloads data first if needed)
	$(BIN)/python -m pytest

lint:  ## ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

notebooks:  ## execute all notebooks top to bottom, in place (full run, ~4 min)
	$(EXEC) $(NB)
	$(MAKE) train

notebooks-smoke:  ## fast subsample run used in CI; writes to reports/_smoke, notebooks untouched
	rm -rf reports/_smoke && mkdir -p reports/_smoke/nb
	HOUSING_SMOKE=1 $(BIN)/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3000 \
		--output-dir reports/_smoke/nb $(NB)

train:  ## fit the final model, evaluate on the hold-out, write models/
	$(BIN)/python -m housing.train

clean:  ## remove caches
	rm -rf .pytest_cache .ruff_cache reports/_smoke
