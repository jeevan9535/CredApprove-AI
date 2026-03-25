import streamlit as st
import pandas as pd
import joblib
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(base_dir, "..", "models", "credit_approval_model.pkl"))
features = joblib.load(os.path.join(base_dir, "..", "models", "model_features.pkl"))

st.set_page_config(page_title="CredApprove AI", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'> CredApprove AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI Powered Credit Risk Decision System</h4>", unsafe_allow_html=True)

st.markdown("## 👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100)
    income = st.number_input("Annual Income", min_value=0.0)
    children = st.number_input("Number of Children", 0)
    family = st.number_input("Family Members", 1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    car = st.selectbox("Owns Car", ["Yes", "No"])

with col2:
    realty = st.selectbox("Owns House", ["Yes", "No"])
    income_type = st.selectbox(
        "Income Type",
        ["Working", "Commercial associate", "Pensioner", "State servant"]
    )
    education = st.selectbox(
        "Education",
        ["Higher education", "Secondary / secondary special", "Incomplete higher"]
    )
    family_status = st.selectbox(
        "Marital Status",
        ["Single / not married", "Married", "Civil marriage"]
    )
    housing = st.selectbox(
        "Housing Type",
        ["House / apartment", "Rented apartment"]
    )

st.markdown("<br>", unsafe_allow_html=True)

center_col = st.columns([1,2,1])
with center_col[1]:
    predict_btn = st.button("Predict Credit Approval")

if predict_btn:

    data = pd.DataFrame(columns=features)
    data.loc[0] = 0

    data["DAYS_BIRTH"] = -age * 365
    data["AMT_INCOME_TOTAL"] = income
    data["CNT_CHILDREN"] = children
    data["CNT_FAM_MEMBERS"] = family

    if "CODE_GENDER_M" in data.columns and gender == "Male":
        data["CODE_GENDER_M"] = 1

    if "FLAG_OWN_CAR_Y" in data.columns and car == "Yes":
        data["FLAG_OWN_CAR_Y"] = 1

    if "FLAG_OWN_REALTY_Y" in data.columns and realty == "Yes":
        data["FLAG_OWN_REALTY_Y"] = 1

    col = f"NAME_INCOME_TYPE_{income_type}"
    if col in data.columns:
        data[col] = 1

    col = f"NAME_EDUCATION_TYPE_{education}"
    if col in data.columns:
        data[col] = 1

    col = f"NAME_FAMILY_STATUS_{family_status}"
    if col in data.columns:
        data[col] = 1

    col = f"NAME_HOUSING_TYPE_{housing}"
    if col in data.columns:
        data[col] = 1

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    risk_percent = probability * 100
    credit_limit = income * (1 - probability) * 0.6

    st.markdown("---")
    st.markdown("## Credit Decision Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 0:
            st.success("✅ Approved")
        else:
            st.error("❌ Rejected")

    with col2:
        st.metric("Risk Score", f"{round(risk_percent,2)}%")

    with col3:
        st.metric("Credit Limit", f"₹ {int(credit_limit)}")

    st.progress(int(risk_percent))

    if risk_percent < 30:
        st.success("🟢 Low Risk")
    elif risk_percent < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    st.markdown("---")
    st.markdown("## Decision Insights")

    if income < 200000:
        st.write("• Low income reduces approval chances")

    if children > 3:
        st.write("• High number of dependents increases financial risk")

    if family > 4:
        st.write("• Larger family size impacts repayment capacity")

    if risk_percent < 30:
        st.write("• Strong financial profile detected")

    elif risk_percent < 70:
        st.write("• Moderate risk due to mixed factors")

    else:
        st.write("• High risk due to financial instability indicators")