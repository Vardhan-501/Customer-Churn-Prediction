import pandas as pd

from src.monitoring import data_quality_summary, labeled_performance
from src.segmentation import fit_segments, profile_segments


def test_segmentation_selects_and_profiles_clusters():
    frame = pd.DataFrame({
        "tenure": [1, 2, 3, 60, 61, 62],
        "MonthlyCharges": [90, 92, 88, 40, 42, 38],
        "TotalCharges": [90, 184, 264, 2400, 2562, 2356],
        "SeniorCitizen": [0, 0, 1, 0, 1, 0],
        "Churn": [1, 1, 0, 0, 0, 0],
    })
    _, segmented, scores = fit_segments(frame, ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"], k_values=(2, 3))
    assert set(scores) == {2, 3}
    assert "segment" in segmented.columns
    profile = profile_segments(segmented)
    assert profile["customers"].sum() == len(frame)


def test_monitoring_outputs_expected_metrics():
    frame = pd.DataFrame({"a": [1, None, 2], "b": ["x", "x", "y"]})
    quality = data_quality_summary(frame)
    assert set(quality["feature"]) == {"a", "b"}
    metrics = labeled_performance([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2], threshold=0.6)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
