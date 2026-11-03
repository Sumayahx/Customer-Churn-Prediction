import streamlit as st
import joblib
import pandas as pd

model = joblib.load('churn_pipeline.pkl')

st.title("Telecom Customer Churn Predictor")
st.write("Enter the key customer details below to evaluate churn risk.")

# 1. Interactive Inputs for Key Features
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=800.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 2. Predict Button & DataFrame Assembly
if st.button("Predict Churn Risk"):
    
    input_dict = {
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [tenure],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': [internet_service],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': [contract],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }
    
    input_data = pd.DataFrame(input_dict)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    if prediction == 1:
        st.error(f"**High Risk!** This customer is likely to churn. (Probability: {probability:.2%})")
    else:
        st.success(f"**Low Risk.** This customer is likely to stay. (Churn Probability: {probability:.2%})")