import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

MODEL_PATH = "/home/jupyter/week-2/DVC-WEEK2-IITM/models/model_default.pkl"
DATA_PATH = "/home/jupyter/week-2/DVC-WEEK2-IITM/data/v1/data.csv"

@pytest.fixture
def model():
    model = joblib.load(MODEL_PATH)
    return model

@pytest.fixture
def test_data():
    df = pd.read_csv(DATA_PATH)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    return X, y

def test_model_accuracy(model, test_data):
    """Verify model achieves reasonable accuracy."""
    X, y = test_data
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.85, f"Model accuracy too low: {acc:.2f}"
