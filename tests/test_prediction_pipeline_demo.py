import pytest
import pandas as pd
import numpy as np
from prediction_pipeline_demo import data_preparation, data_split, train_model, eval_model

@pytest.fixture
def housing_data_sample():
    rows = []
    base = {
        "price": 13_300_000,
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished",
    }
    statuses = ["furnished", "unfurnished", "semi-furnished"]
    # Create 30 varied rows so train/test split is meaningful
    for i in range(30):
        rows.append({
            **base,
            "price": base["price"] - i * 100_000,
            "area": base["area"] + i * 150,
            "bedrooms": 3 + (i % 3),
            "bathrooms": 2 + (i % 3),
            "stories": 2 + (i % 3),
            "mainroad": "yes" if i % 2 == 0 else "no",
            "guestroom": "no" if i % 3 else "yes",
            "basement": "no" if (i+1) % 3 else "yes",
            "hotwaterheating": "no" if i % 4 else "yes",
            "airconditioning": "yes" if i % 5 else "no",
            "parking": 1 + (i % 3),
            "prefarea": "yes" if i % 2 == 0 else "no",
            "furnishingstatus": statuses[i % 3],
        })
    return pd.DataFrame(rows)

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints have same length
    assert feature_df.shape[0] == len(target_series)
    # Features are numeric/bool after one-hot
    numeric_bool_cols = feature_df.select_dtypes(include=(np.number, bool)).shape[1]
    assert feature_df.shape[1] == numeric_bool_cols

def test_data_preparation_columns(housing_data_sample):
    """data_preparation should produce one-hot columns for furnishingstatus."""
    feature_df, _ = data_preparation(housing_data_sample)
    # All three statuses present, so 3 one-hot columns expected
    assert feature_df.shape[1] == 3
    # No raw 'furnishingstatus' column should remain
    assert "furnishingstatus" not in feature_df.columns

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split_returns_four_parts(feature_target_sample):
    parts = data_split(*feature_target_sample)
    # Returns exactly 4 parts
    assert isinstance(parts, tuple)
    assert len(parts) == 4

def test_data_split_sizes(feature_target_sample):
    """Test/train sizes respect the 33% test_size split."""
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    total = len(X_train) + len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert total == len(feature_target_sample[0])
    # ~33% goes to test
    assert len(X_test) == pytest.approx(total * 0.33, abs=2)

def test_train_model_returns_model(feature_target_sample):
    """train_model should return a fitted LinearRegression."""
    from sklearn.linear_model import LinearRegression
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    model = train_model(X_train, y_train)
    assert isinstance(model, LinearRegression)
    # A fitted model has coef_ attribute
    assert hasattr(model, "coef_")

def test_end_to_end_train_and_eval(feature_target_sample):
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    model = train_model(X_train, y_train)
    score = eval_model(X_test, y_test, model)
    # score is a float and finite
    assert isinstance(score, float)
    assert np.isfinite(score)

def test_eval_model_range(feature_target_sample):
    """R^2 score on in-sample data should be <= 1.0."""
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    model = train_model(X_train, y_train)
    score = eval_model(X_test, y_test, model)
    assert score <= 1.0
