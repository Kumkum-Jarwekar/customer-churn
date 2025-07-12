import streamlit as st
import pandas as pd
import pickle

# Load trained churn prediction model
with open('customer_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor")
st.title("Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn.")

# Input form
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=800.0)
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"])

# Feature encoding
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
    'InternetService': 0 if internet_service == 'DSL' else 1 if internet_service == 'Fiber Optic' else 2,
    'OnlineSecurity': 1 if online_security == 'Yes' else 0,
    'OnlineBackup': 1 if online_backup == 'Yes' else 0,
    'DeviceProtection': 1 if device_protection == 'Yes' else 0,
    'TechSupport': 1 if tech_support == 'Yes' else 0,
    'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
    'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
    'Contract': 0 if contract == 'Month-to-Month' else 1 if contract == 'One Year' else 2,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    'PaymentMethod': 0 if payment_method == 'Bank Transfer' else 1 if payment_method == 'Credit Card' else 2 if payment_method == 'Electronic Check' else 3
}

input_df = pd.DataFrame([input_data])

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is unlikely to churn (Probability of staying: {1 - probability:.2f})")