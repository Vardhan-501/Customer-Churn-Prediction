# 📊 Telco Customer Churn Prediction

<p align="center">
  <em>An end-to-end machine learning project that predicts customer churn risk using account and service data — trained, evaluated, and deployed as an interactive web app.</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-5.24-3F4F75?style=flat-square&logo=plotly&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

<p align="center">
  <a href="#-live-demo"><b>Live Demo</b></a> •
  <a href="#-features"><b>Features</b></a> •
  <a href="#-tech-stack"><b>Tech Stack</b></a> •
  <a href="#-installation--setup"><b>Setup</b></a> •
  <a href="#-model-performance"><b>Model Performance</b></a>
</p>

---

## 🔗 Live Demo

**🚀 [Try the app here](https://ibm-telco-churn-prediction.streamlit.app/)


## 📌 Overview

Customer churn — when a customer stops using a company's service — directly impacts revenue, and acquiring a new customer typically costs far more than retaining an existing one. This project builds a machine learning pipeline that predicts **whether a telecom customer is likely to churn**, based on their account details, contract type, billing information, and subscribed services.

The project covers the complete ML workflow:
- Data cleaning and exploratory analysis on real-world, imperfect data
- Multicollinearity analysis (VIF)
- A trained, evaluated classification model
- An interactive Streamlit dashboard for live predictions
- Public deployment for anyone to try

---

## ✨ Features

- 🔍 **Exploratory Data Analysis** — visualizes churn patterns across tenure, contract type, charges, and services
- 📐 **Multicollinearity check (VIF)** — validates feature independence before modeling
- 🤖 **Random Forest Classifier** — trained with class balancing to handle imbalanced churn data
- 📊 **Model evaluation** — precision, recall, F1-score, ROC-AUC, confusion matrix, and feature importance
- 🖥️ **Interactive dashboard** — enter a customer's details and get a live churn prediction with probability gauge
- 📈 **Explainability** — shows the top factors driving each prediction
- ☁️ **Deployed on Streamlit Cloud** — no installation needed to try it

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization (EDA)** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier) |
| **Statistical Analysis** | statsmodels (VIF / multicollinearity) |
| **Model Persistence** | joblib |
| **Web App / Dashboard** | Streamlit |
| **Interactive Charts (app)** | Plotly |
| **Development** | Jupyter Notebook, VS Code |
| **Version Control & Hosting** | Git, GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📂 Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)

- **7,043** customer records, **21** features
- Includes demographic info (gender, senior citizen status, partner/dependents), account info (tenure, contract, billing), subscribed services (internet, phone, streaming, security add-ons), and the target label (`Churn`: Yes/No)
- Real-world data quality issues handled in this project: `TotalCharges` stored as text with blank values, class imbalance (~27% churn rate)

---

## 📁 Project Structure

```
telco-churn-prediction/
├── .streamlit/
│   └── config.toml                     # Dashboard theme (dark/green)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Customer_Churn_Prediction.ipynb     # Full analysis: EDA → cleaning → VIF → modeling → prediction
├── train_model.py                       # Standalone training script
├── app.py                               # Streamlit dashboard app
├── churn_model.pkl                      # Saved trained pipeline (preprocessing + model)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get the dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` folder.

### 5. Train the model *(optional — a pre-trained `churn_model.pkl` is already included)*
```bash
python train_model.py
```
Or run through `Customer_Churn_Prediction.ipynb` step by step in Jupyter.

### 6. Run the app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | ~0.84 |
| Accuracy | ~0.80 |
| Precision (Churn class) | ~0.65 |
| Recall (Churn class) | ~0.70 |
| F1-score (Churn class) | ~0.67 |


**Top predictive features:** Contract type, tenure, monthly charges, internet service type, and payment method.

---

## 🧠 How Prediction Works

1. Customer details are collected through the sidebar form
2. Inputs are passed through a saved `scikit-learn` pipeline that automatically:
   - Scales numeric features (`StandardScaler`)
   - One-hot encodes categorical features (`OneHotEncoder`)
3. The `RandomForestClassifier` outputs a churn probability
4. The dashboard displays the risk level, probability gauge, and the top features influencing that prediction

---

## 🚀 Future Improvements

- [ ] Compare against XGBoost / LightGBM / Logistic Regression
- [ ] Add SHAP values for per-prediction explainability
- [ ] Support batch predictions via CSV upload
- [ ] Add model monitoring / retraining pipeline
- [ ] Track experiments with MLflow

---

## 👤 Author

**Your Name**
📧 priyavardhanakula114433@gmail.com | 🔗 [LinkedIn](www.linkedin.com/in/priyavardhanakula) | 💻 [GitHub](https://github.com/Vardhan-501)

---


<p align="center">Built as a learning project to demonstrate end-to-end ML workflow — from raw data to a deployed, interactive prediction app.</p>
