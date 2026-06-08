import joblib as jb
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Bank Loan Project",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_model():
    return jb.load("load_model.pkl")

model = load_model()

st.title("🏦 Bank Loan Risk Predictor")
st.markdown("Enter an applicant's details below to check loan risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Enter Age", 18, 99, 45)
    income = st.slider("Monthly income [PKR]", 15000, 200000, 30000, step=5000)
    credit_use = st.slider("Credit Utilization (%)", 0, 100, 30)
    debt_ratio = st.slider("Debt Ratio (%)", 0, 100, 20)

with col2:
    late_30 = st.slider("Times paid 30 days late", 0, 25, 0)
    late_60 = st.slider("Times paid 60 days late", 0, 25, 0)
    late_90 = st.slider("Times paid 90 days late", 0, 25, 0)
    open_loans = st.slider("Number of open loans", 0, 20, 2)
    property_loans = st.slider("Number of property loans", 0, 10, 0)
    dependents = st.slider("Number of dependents", 0, 10, 0)

if st.button("🔍 Check Risk", use_container_width=True):

    applicant = np.array([[
        credit_use,
        age,
        late_30,
        debt_ratio,
        income,
        open_loans,
        late_90,
        property_loans,
        late_60,
        dependents
    ]])

    prob = model.predict_proba(applicant)[0][1]
    percent = prob * 100

    st.subheader("Result")

    if percent < 30:
        st.success(f"Low Risk: {percent:.2f}%")
    elif percent < 70:
        st.warning(f"Medium Risk: {percent:.2f}%")
    else:
        st.error(f"High Risk: {percent:.2f}%")