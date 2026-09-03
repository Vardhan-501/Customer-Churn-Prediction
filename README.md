# Customer Churn Prediction

An end-to-end machine-learning project that estimates telecom customer churn risk from account, service, contract, billing, and demographic information. The project combines exploratory analysis, reproducible preprocessing, model evaluation, feature analysis, and an interactive Streamlit application.

[Launch the live Streamlit application](https://ibm-telco-churn-prediction.streamlit.app/)

## Business problem

Customer churn reduces recurring revenue and increases acquisition pressure. A churn-risk model can help a telecom team prioritize retention investigations, provided that predictions are evaluated carefully and used with an explicit understanding of intervention costs.

This project answers two related questions:

1. Which customer attributes are associated with churn in the available historical data?
2. Can a classification model identify customers with elevated churn risk well enough to support targeted retention analysis?

The model is a decision-support prototype. It does not prove that any feature causes churn, and predictions should not be used as the sole basis for customer treatment.

## Project highlights

- Cleans real-world customer data, including numeric values stored as text.
- Examines churn patterns across tenure, contracts, charges, services, and payment methods.
- Checks multicollinearity using variance inflation factor analysis.
- Uses a preprocessing-and-model pipeline to reduce train/test transformation leakage.
- Handles class imbalance with class-weighted Random Forest classification.
- Reports precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Persists the trained pipeline with `joblib`.
- Provides an interactive Streamlit application with a churn-probability gauge.
- Includes feature-importance visualization for global model interpretation.

## Dataset

- **Dataset:** IBM Telco Customer Churn.
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Records:** 7,043 customer records.
- **Features:** 21 columns, including demographic, account, billing, contract, and service fields.
- **Target:** `Churn`, indicating whether the customer left the service.
- **Source:** [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> Confirm the dataset license and attribution requirements before redistributing the dataset.

## Analytical workflow

```text
Raw data
   ↓
Data-quality checks and type conversion
   ↓
Exploratory data analysis
   ↓
Feature and target definition
   ↓
Stratified train/test split
   ↓
ColumnTransformer: scaling + one-hot encoding
   ↓
Class-weighted Random Forest pipeline
   ↓
Holdout evaluation and error analysis
   ↓
Persisted model
   ↓
Streamlit prediction application
```

The notebook is available at [`Customer_Churn_Project.ipynb`](Customer_Churn_Project.ipynb), and the application is available at [`app.py`](app.py).

## Data preparation

The workflow addresses the following data issues:

- Converts `TotalCharges` from text to numeric values.
- Handles blank or invalid numeric entries.
- Separates the target variable from explanatory features.
- Identifies numeric and categorical columns.
- Applies scaling only within the modeling pipeline.
- Applies one-hot encoding with unknown-category handling.
- Uses stratification to preserve the churn proportion in the train and test sets.

Document the exact missing-value treatment, row-removal count, and final feature list here after the final notebook run:

| Preparation item | Verified result |
|---|---|
| Rows before cleaning | **[X,XXX]** |
| Rows after cleaning | **[X,XXX]** |
| Blank `TotalCharges` values | **[XXX]** |
| Churn rate | **[XX.XX%]** |
| Numeric features | **[List or link to notebook]** |
| Categorical features | **[List or link to notebook]** |

## Modeling approach

The current model uses a `scikit-learn` `Pipeline` containing a `ColumnTransformer` and a class-weighted `RandomForestClassifier`. This keeps preprocessing and prediction together and makes the persisted model easier to reuse in the application.

Recommended experiment table for the next version:

| Model | Validation method | ROC-AUC | PR-AUC | Recall | F1-score | Notes |
|---|---|---:|---:|---:|---:|---|
| Majority-class baseline | Stratified cross-validation | [ ] | [ ] | [ ] | [ ] | Reference point |
| Logistic regression | Stratified cross-validation | [ ] | [ ] | [ ] | [ ] | Interpretable baseline |
| Random Forest | Stratified cross-validation | [ ] | [ ] | [ ] | [ ] | Current model |
| Gradient boosting | Stratified cross-validation | [ ] | [ ] | [ ] | [ ] | Candidate comparison |

The current notebook reports a stratified 80/20 train/test split and a Random Forest evaluation. For a stronger production-oriented result, add repeated stratified cross-validation, PR-AUC, score variability, threshold analysis, calibration, and a comparison against a simple baseline.

## Model performance

The current README reports the following approximate holdout results. Re-run the notebook and replace these values with exact outputs before using them in applications:

| Metric | Current reported result | Verified final result |
|---|---:|---:|
| ROC-AUC | ~0.84 | **[ ]** |
| Accuracy | ~0.80 | **[ ]** |
| Churn precision | ~0.65 | **[ ]** |
| Churn recall | ~0.70 | **[ ]** |
| Churn F1-score | ~0.67 | **[ ]** |
| PR-AUC | Not currently reported | **[ ]** |

Because churn is an imbalanced classification problem, accuracy should not be treated as the primary metric. The preferred metric depends on the business cost of missing a likely churner versus contacting a customer who would have stayed.

## Error analysis and decision threshold

The application currently uses the model’s predicted class to display a high- or low-risk label. A stronger business version should document:

- The probability threshold used for intervention.
- The expected cost of false negatives.
- The expected cost of false positives.
- The retention capacity available to the business.
- Precision and recall at the chosen threshold.
- Calibration quality of predicted probabilities.

Add a confusion-matrix interpretation here:

> At the selected threshold of **[threshold]**, the model identifies **[X%]** of observed churners while generating **[Y%]** false-positive contacts. This threshold was selected because **[business rationale]**.

## Explainability

The app displays global Random Forest feature importance. These values describe how the fitted model uses features overall; they do not prove that a feature causes churn and they do not constitute a complete explanation for every individual prediction.

Recommended next steps are permutation importance, partial-dependence or accumulated-local-effect plots, and SHAP explanations for individual predictions. Add appropriate caveats around correlation, data quality, and fairness when publishing explanations.

## Streamlit application

The application accepts customer information through a sidebar form and returns:

- Predicted churn class.
- Churn probability.
- A probability gauge.
- An input summary.
- Global top-feature importance visualization.
- A plain-language explanation of the risk label.

The app code is in [`app.py`](app.py). The persisted pipeline is in `churn_model.pkl`.

## Repository structure

```text
Customer-Churn-Prediction/
├── .gitignore
├── Customer_Churn_Project.ipynb
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app.py
├── churn_model.pkl
└── requirements.txt
```

## How to run locally

```bash
git clone https://github.com/Vardhan-501/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit. The current repository includes a pre-trained model artifact. If the model is retrained, document the training date, feature schema, library versions, evaluation split, and model hash or version.

## Reproducibility checklist

- [ ] Pin dependency versions in `requirements.txt`.
- [ ] Document the Python version.
- [ ] Record the random seed.
- [ ] Add a standalone training script or clearly document the notebook training cells.
- [ ] Save the exact feature list and target definition.
- [ ] Report cross-validation results and the final test-set result separately.
- [ ] Add input validation for missing, invalid, and out-of-range values.
- [ ] Add batch CSV scoring for operational use cases.
- [ ] Add a model card documenting intended use, limitations, and ethical considerations.

## Limitations and responsible use

This dataset represents historical customer behavior and may not reflect current products, prices, service quality, or customer populations. The model may encode historical patterns that are not appropriate for intervention decisions. It should not be used to deny service, penalize customers, or make decisions without human review.

The model reports association and predictive performance, not causation. Any retention intervention should be evaluated with a controlled experiment or another appropriate evaluation design.

## Future improvements

- Compare logistic regression, gradient boosting, and other baseline models.
- Add stratified cross-validation and PR-AUC.
- Tune and justify the intervention threshold.
- Calibrate predicted probabilities.
- Add SHAP or permutation-based explanations.
- Add batch scoring and input validation.
- Add model monitoring for feature drift and performance decay.
- Add fairness checks across relevant customer groups where legally and ethically appropriate.
- Track model versions and experiments with MLflow.

## Author

**Priyavardhan Akula**

- Email: [priyavardhanakula114433@gmail.com](mailto:priyavardhanakula114433@gmail.com)
- LinkedIn: [linkedin.com/in/priyavardhanakula](https://www.linkedin.com/in/priyavardhanakula)
- GitHub: [github.com/Vardhan-501](https://github.com/Vardhan-501)
