import pandas as pd
import shap


def _positive_class_values(shap_values, positive_class: int = 1):
    """Normalize SHAP's version-dependent binary-classification output."""
    if isinstance(shap_values, list):
        return shap_values[positive_class][0]
    if getattr(shap_values, "ndim", 0) == 3:
        return shap_values[0, :, positive_class]
    return shap_values[0]


def explain_prediction(model, input_df: pd.DataFrame, positive_class: int = 1, top_n: int = 8) -> pd.DataFrame:
    """Return the strongest individual feature contributions for one customer."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=input_df.index)
    explainer = shap.TreeExplainer(classifier)
    values = _positive_class_values(explainer.shap_values(transformed_df), positive_class)
    explanation = pd.DataFrame({
        "feature": feature_names,
        "impact": values,
    })
    explanation["display_feature"] = (
        explanation["feature"].str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    explanation["absolute_impact"] = explanation["impact"].abs()
    return explanation.sort_values("absolute_impact", ascending=False).head(top_n).reset_index(drop=True)
