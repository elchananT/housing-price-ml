import warnings

import pandas as pd
import pytest

from housing import data

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """Full raw Ames data (downloaded once into data/ if absent)."""
    return data.load_data()


@pytest.fixture(scope="session")
def sample(raw) -> pd.DataFrame:
    """A small, fixed subsample for fast model-fitting tests."""
    return raw.sample(220, random_state=0).reset_index(drop=True)
