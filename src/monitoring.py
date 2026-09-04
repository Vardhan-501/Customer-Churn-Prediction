import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-rate and basic numeric summaries for a dataset."""
    rows = []
    for column in df.columns:
        rows.append({
            "feature": column,
            "missing_rate": float(df[column].isna().mean()),
            "unique_values": int(df[column].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def numeric_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compare numeric means, medians, and standard deviations between datasets."""
    rows = []
    for column in columns:
        ref = pd.to_numeric(reference[column], errors="coerce")
        cur = pd.to_numeric(current[column], errors="coerce")
        rows.append({
            "feature": column,
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "mean_delta": float(cur.mean() - ref.mean()),
            "reference_median": float(ref.median()),
            "current_median": float(cur.median()),
            "reference_std": float(ref.std()),
            "current_std": float(cur.std()),
        })
    return pd.DataFrame(rows)


def labeled_performance(y_true, probabilities, threshold: float = 0.60) -> dict[str, float]:
    """Calculate performance metrics when later-period labels are available."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }
