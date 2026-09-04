from typing import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_segments(
    df: pd.DataFrame,
    feature_columns: list[str],
    k_values: Iterable[int] = (2, 3, 4, 5, 6),
):
    """Fit KMeans segments and select k using the silhouette score."""
    matrix = df[feature_columns].copy().apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median())
    scaled = StandardScaler().fit_transform(matrix)
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(scaled)
        scores[k] = float(silhouette_score(scaled, labels))
    best_k = max(scores, key=scores.get)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    output = df.copy()
    output["segment"] = model.fit_predict(scaled)
    return model, output, scores


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize segment size, churn, and charge/tenure medians."""
    profile = df.groupby("segment").agg(
        customers=("segment", "size"),
        churn_rate=("Churn", "mean") if "Churn" in df.columns else ("segment", "size"),
        median_tenure=("tenure", "median"),
        median_monthly_charges=("MonthlyCharges", "median"),
    ).reset_index()
    if "Churn" not in df.columns:
        profile["churn_rate"] = float("nan")
    return profile
