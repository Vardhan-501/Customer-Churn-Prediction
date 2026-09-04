from pathlib import Path

import pandas as pd

from src.explain import explain_prediction
from src.predict import load_model
from src.recommendations import recommend_actions


CUSTOMER = {
    "SeniorCitizen": 0,
    "tenure": 2,
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_shap_explanation_contains_features():
    model = load_model(Path(__file__).parent / "models" / "churn_model.pkl")
    feature_order = list(model.named_steps["preprocessor"].feature_names_in_)
    explanation = explain_prediction(model, pd.DataFrame([CUSTOMER])[feature_order])
    assert not explanation.empty
    assert {"feature", "impact", "absolute_impact", "display_feature"}.issubset(explanation.columns)
    assert explanation["absolute_impact"].is_monotonic_decreasing


def test_recommendations_include_reasons():
    actions = recommend_actions(CUSTOMER, 0.80, pd.DataFrame())
    assert actions
    assert all(item["reason"] for item in actions)
    assert any(item["priority"] == "High" for item in actions)
