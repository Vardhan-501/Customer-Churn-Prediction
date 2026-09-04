from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.pkl"

NUMERIC_RANGES = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (0, float("inf")),
    "TotalCharges": (0, float("inf")),
}


def load_model(model_path: Path = MODEL_PATH):
    """Load the serialized sklearn pipeline."""
    return joblib.load(model_path)


def model_schema(model) -> dict[str, list[Any]]:
    """Return expected numeric and categorical columns and category values."""
    preprocessor = model.named_steps["preprocessor"]
    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.transformers_[1][1]
    categories = {
        column: list(values)
        for column, values in zip(categorical_columns, encoder.categories_)
    }
    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "categories": categories,
    }


def validate_customer(model, customer: dict[str, Any]) -> None:
    """Raise ValueError when customer input is incomplete or invalid."""
    schema = model_schema(model)
    expected = schema["numeric"] + schema["categorical"]
    missing = [column for column in expected if column not in customer]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    for column in schema["numeric"]:
        value = customer[column]
        if value is None or isinstance(value, bool):
            raise ValueError(f"{column} must be numeric")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if numeric_value != numeric_value:
            raise ValueError(f"{column} cannot be NaN")
        lower, upper = NUMERIC_RANGES.get(column, (0, float("inf")))
        if numeric_value < lower or numeric_value > upper:
            raise ValueError(f"{column} must be between {lower} and {upper}")

    for column in schema["categorical"]:
        if customer[column] not in schema["categories"][column]:
            allowed = ", ".join(map(str, schema["categories"][column]))
            raise ValueError(f"Invalid value for {column}: {customer[column]!r}. Allowed: {allowed}")


def predict_customer(model, customer: dict[str, Any]) -> dict[str, float | int | str]:
    """Validate one customer and return prediction, probability, and risk band."""
    validate_customer(model, customer)
    schema = model_schema(model)
    input_df = pd.DataFrame([customer])[schema["numeric"] + schema["categorical"]]
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_band": risk_band(probability),
    }


def risk_band(probability: float, low: float = 0.30, high: float = 0.60) -> str:
    """Map probability to provisional, documented risk bands."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if probability < low:
        return "Low"
    if probability < high:
        return "Medium"
    return "High"
