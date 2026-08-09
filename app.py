"""
Customer Churn Prediction - Streamlit App (styled dashboard version)
----------------------------------------------------------------------
Works with a plain sklearn Pipeline saved via:
    joblib.dump(model, "churn_model.pkl")

Also ships a matching .streamlit/config.toml for the dark/green theme.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# CUSTOM CSS — card styling to match the dashboard look
# =============================================================
st.markdown("""
<style>
    .card {
        background-color: #161618;
        border: 1px solid #262629;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .card h3 {
        margin-top: 0;
    }
    .result-card {
        background-color: #10170f;
        border: 1px solid #1f3d20;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 20px;
    }
    .accent-text {
        color: #22c55e;
    }
    .risk-high {
        color: #22c55e;
        font-weight: 700;
    }
    .risk-low {
        color: #22c55e;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: #22c55e;
    }
    .stButton > button {
        background-color: #22c55e;
        color: #08130a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1em;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #16a34a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# LOAD MODEL
# =============================================================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

preprocessor = model.named_steps["preprocessor"]
numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])
onehot_encoder = preprocessor.transformers_[1][1]
category_values = dict(zip(categorical_cols, [list(c) for c in onehot_encoder.categories_]))
feature_order = numeric_cols + categorical_cols

# Raw pipeline doesn't store original min/max -> use known Telco dataset ranges.
NUMERIC_RANGES = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (18.25, 118.75),
    "TotalCharges": (18.80, 8684.80),
}

# =============================================================
# SIDEBAR — INPUT FORM
# =============================================================
with st.sidebar:
    st.markdown("### 📊 Telco Churn Predictor")
    st.markdown("#### Customer Information")
    st.caption("Enter the customer details below to predict churn risk.")
    st.divider()

    user_input = {}

    for col in categorical_cols:
        options = category_values[col]
        user_input[col] = st.selectbox(col.replace("_", " "), options)

    for col in numeric_cols:
        lo, hi = NUMERIC_RANGES.get(col, (0.0, 100.0))
        if col == "SeniorCitizen":
            choice = st.selectbox("Senior Citizen", ["No", "Yes"])
            user_input[col] = 1 if choice == "Yes" else 0
        else:
            label = {
                "tenure": "Tenure (months)",
                "MonthlyCharges": "Monthly Charges (USD)",
                "TotalCharges": "Total Charges (USD)",
            }.get(col, col)
            default = (lo + hi) / 2
            user_input[col] = st.slider(label, min_value=float(lo), max_value=float(hi), value=float(default))

    st.divider()
    predict_clicked = st.button("📈  Predict Churn")

# =============================================================
# MAIN HEADER
# =============================================================
header_col1, header_col2 = st.columns([0.06, 0.94])
with header_col1:
    st.markdown("### 📊")
with header_col2:
    st.markdown("## IBM Telco Customer Churn Prediction")
st.write("This app predicts whether a customer is likely to churn based on the provided information.")

# =============================================================
# PREDICTION
# =============================================================
if predict_clicked:
    input_df = pd.DataFrame([user_input])[feature_order]
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]
    risk_label = "High" if pred == 1 else "Low"
    risk_icon = "⚠️" if pred == 1 else "✅"
    risk_message = (
        "This customer is likely to churn."
        if pred == 1 else
        "This customer is likely to stay."
    )

    # ---------------- Result card with gauge ----------------
    result_left, result_right = st.columns([0.62, 0.38])

    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col_text, col_gauge = st.columns([0.6, 0.4])

        with col_text:
            st.markdown("#### Prediction Result")
            st.markdown(f"### Churn Risk: <span class='accent-text'>{risk_label}</span> {risk_icon}", unsafe_allow_html=True)
            st.write(risk_message)

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "#22c55e"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#666"},
                    "bar": {"color": "#22c55e"},
                    "bgcolor": "#0e0e10",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 100], "color": "#1a1a1c"},
                    ],
                },
                title={"text": "Churn Probability", "font": {"size": 14, "color": "#cccccc"}},
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f5f5f5"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Info card + Input summary ----------------
    info_col, summary_col = st.columns(2)

    with info_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ℹ️ What does this mean?")
        if pred == 1:
            st.write(
                "The model predicts that this customer has a high probability of "
                "churning. You may consider taking retention actions such as "
                "offering discounts or improving customer support."
            )
        else:
            st.write(
                "The model predicts that this customer has a low probability of "
                "churning. Standard engagement and service quality should be "
                "sufficient to retain them."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with summary_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🧾 Input Summary")
        summary_rows = [
            ("Tenure (months)", user_input.get("tenure")),
            ("Monthly Charges (USD)", user_input.get("MonthlyCharges")),
            ("Total Charges (USD)", user_input.get("TotalCharges")),
            ("Contract", user_input.get("Contract")),
            ("Internet Service", user_input.get("InternetService")),
            ("Payment Method", user_input.get("PaymentMethod")),
        ]
        summary_df = pd.DataFrame(summary_rows, columns=["Feature", "Value"])
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Top factors bar chart ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Top Factors Influencing Churn (According to Model)")

    importances = model.named_steps["classifier"].feature_importances_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df["feature"] = imp_df["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    imp_df = imp_df.sort_values("importance", ascending=True).tail(5)

    fig_bar = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["feature"],
        orientation="h",
        marker_color="#22c55e",
        text=[f"{v:.2f}" for v in imp_df["importance"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f5f5f5"},
        xaxis={"showgrid": False, "visible": False},
        yaxis={"showgrid": False},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Higher values indicate more influence on churn.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to see results.")

st.divider()
st.caption("© 2026 Telco Churn Prediction App | Built with Streamlit 💚")