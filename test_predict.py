from pathlib import Path

import pytest

from src.predict import load_model, predict_customer


@pytest.fixture(scope="module")
def model():
    return load_model(Path(__file__).parents[1] / "models" / "churn_model.pkl")


@pytest.fixture
def valid_customer():
    return {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }


def test_probability_is_between_zero_and_one(model, valid_customer):
    result = predict_customer(model, valid_customer)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_band"] in {"Low", "Medium", "High"}


def test_missing_fields_produce_clear_error(model, valid_customer):
    customer = valid_customer.copy()
    customer.pop("Contract")
    with pytest.raises(ValueError, match="Missing required fields"):
        predict_customer(model, customer)


def test_negative_charges_are_rejected(model, valid_customer):
    customer = valid_customer.copy()
    customer["MonthlyCharges"] = -1
    with pytest.raises(ValueError, match="MonthlyCharges"):
        predict_customer(model, customer)


def test_unknown_categorical_value_is_rejected(model, valid_customer):
    customer = valid_customer.copy()
    customer["Contract"] = "Invalid contract"
    with pytest.raises(ValueError, match="Invalid value for Contract"):
        predict_customer(model, customer)


def test_impossible_tenure_is_rejected(model, valid_customer):
    customer = valid_customer.copy()
    customer["tenure"] = 73
    with pytest.raises(ValueError, match="tenure"):
        predict_customer(model, customer)
