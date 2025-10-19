import pandas as pd
import pytest
import os

DATA_PATH = "/home/jupyter/week-2/DVC-WEEK2-IITM/data/v1/data.csv"

@pytest.fixture
def iris_data():
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df

def test_no_missing_values(iris_data):
    """Ensure there are no null values in the dataset."""
    assert not iris_data.isnull().values.any(), "Dataset contains missing values!"

def test_column_names(iris_data):
    """Verify required columns are present."""
    required_cols = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width",
        "species"
    ]
    for col in required_cols:
        assert col in iris_data.columns, f"Missing column: {col}"

