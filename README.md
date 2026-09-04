# Explainable Customer Retention Decision-Support System

This project is an explainable customer-retention decision-support application built on the IBM Telco Customer Churn dataset. The saved supervised-learning pipeline remains the source of truth for predictions. Individual explanations, recommendations, segmentation, monitoring, and the optional language-model summary are separate supporting layers.

## Architecture

```text
Customer input
      |
Input validation
      |
Saved churn pipeline
      |
Probability + provisional risk band
      |
+-------------------+----------------------+------------------+
| Individual SHAP   | Rule-based retention | Customer         |
| explanation       | investigation actions| segmentation     |
+-------------------+----------------------+------------------+
      |
Verified structured result
      |
Optional structured AI business summary
      |
Streamlit interface
```

## Implemented features

The application currently provides validated customer prediction, provisional Low/Medium/High risk bands, customer-specific SHAP contributions, transparent recommendation rules with reasons, silhouette-based segmentation, cross-validation metrics, threshold analysis, a monitoring prototype, and an optional strict-JSON AI summary. The AI summary is disabled until an API key is explicitly configured.

## Verified model comparison

The following values were generated with stratified 5-fold cross-validation using the included dataset and evaluation workflow.

| Model | ROC-AUC mean | ROC-AUC std | PR-AUC mean | PR-AUC std | Recall mean | F1 mean |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.5000 | 0.0000 | 0.2654 | 0.0002 | 0.0000 | 0.0000 |
| Logistic regression | 0.8450 | 0.0135 | 0.6555 | 0.0275 | 0.8020 | 0.6258 |
| Random forest | 0.8427 | 0.0125 | 0.6525 | 0.0281 | 0.7549 | 0.6257 |
| Gradient boosting | 0.8469 | 0.0107 | 0.6625 | 0.0235 | 0.5313 | 0.5908 |

The current saved application model is the random forest pipeline. At a provisional 0.60 threshold on the held-out split, it achieved 0.5672 precision, 0.6658 recall, 190 false positives, 125 false negatives, and a 0.3116 predicted contact rate. Threshold selection must ultimately reflect retention capacity and intervention cost.

## Project structure

```text
.
├── app.py
├── data/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/churn_model.pkl
├── notebooks/Customer_Churn_Project.ipynb
├── model_comparison.csv
├── threshold_analysis.csv
├── src/
│   ├── ai_summary.py
│   ├── explain.py
│   ├── metrics.py
│   ├── monitoring.py
│   ├── predict.py
│   ├── recommendations.py
│   └── segmentation.py
└── tests/
    ├── test_explain_and_recommendations.py
    └── test_predict.py
```

## Run locally

```bash
python3 -m pip install -r requirements.txt
pytest -q
streamlit run app.py
```

The serialized model was originally created with scikit-learn 1.7.2. The current development environment uses a newer scikit-learn release and emits a compatibility warning when loading the artifact. Before deployment, retrain and save the model in a pinned environment, or pin the exact training dependency versions.

## Optional AI summary

Set `OPENAI_API_KEY` through the deployment environment or Streamlit secrets. Never commit `.env`, `.streamlit/secrets.toml`, API keys, or private customer data. The summary request receives only verified probability, risk-band, explanation, recommendation, and limitation fields. It must not invent metrics, causes, discounts, or customer history.

## Responsible-AI limitations

SHAP contributions describe associations in the fitted model and do not establish causation. Recommendations are investigation suggestions, not guaranteed retention solutions or automatic treatments. Segments are analytical groupings and should not be presented as natural customer types without validation. Monitoring is a prototype until later-period data and labels are supplied. Human review remains required for customer-facing decisions.

## Next development steps

The next production-quality steps are to pin or reproduce the model-training environment, add tests for segmentation and monitoring, compare category and risk distributions between reference and later datasets, calibrate probabilities, select a threshold using an explicit retention-capacity assumption, and deploy only after reviewing privacy, fairness, and operational controls.

## Original project attribution

The application was upgraded from the public repository [Vardhan-501/Customer-Churn-Prediction](https://github.com/Vardhan-501/Customer-Churn-Prediction). The source dataset is the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Confirm dataset licensing and attribution requirements before redistribution.
