from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path).drop_duplicates().copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])
    y = df.pop("Churn").map({"Yes": 1, "No": 0}).astype(int)
    return df, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def build_estimators(X: pd.DataFrame) -> dict[str, Pipeline]:
    def pipeline(classifier):
        return Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("classifier", classifier),
        ])

    return {
        "majority_baseline": pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
        )),
        "gradient_boosting": pipeline(GradientBoostingClassifier(random_state=42)),
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "f1": "f1"}
    rows = []
    for name, estimator in build_estimators(X).items():
        scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            **{f"{metric}_mean": float(scores[f"test_{metric}"].mean()) for metric in scoring},
            **{f"{metric}_std": float(scores[f"test_{metric}"].std()) for metric in scoring},
        })
    return pd.DataFrame(rows)


def threshold_analysis(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: Iterable[float] = (0.30, 0.40, 0.50, 0.60, 0.70),
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    fitted = clone(estimator).fit(X_train, y_train)
    probabilities = fitted.predict_proba(X_test)[:, 1]
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "contact_rate": float(predictions.mean()),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
        })
    return pd.DataFrame(rows)
