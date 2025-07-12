# app.py for Customer Churn Prediction using Streamlit and pickle

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model using pickle
with open('customer_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title='Customer Churn Prediction', layout='centered')
st.title('📊 Customer Churn Prediction App')
st.write('Input customer details below to predict churn using your trained Random Forest model.')

def user_input():
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0, value=70.0)
    TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=20000.0, value=800.0)
    SeniorCitizen = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    Partner = st.selectbox('Has Partner', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    Dependents = st.selectbox('Has Dependents', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    PhoneService = st.selectbox('Phone Service', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    MultipleLines = st.selectbox('Multiple Lines', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    InternetService = st.selectbox('Internet Service', [0, 1, 2], format_func=lambda x: 'DSL' if x==0 else 'Fiber' if x==1 else 'No')
    OnlineSecurity = st.selectbox('Online Security', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    OnlineBackup = st.selectbox('Online Backup', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    DeviceProtection = st.selectbox('Device Protection', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    TechSupport = st.selectbox('Tech Support', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    StreamingTV = st.selectbox('Streaming TV', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    StreamingMovies = st.selectbox('Streaming Movies', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    Contract = st.selectbox('Contract Type', [0, 1, 2], format_func=lambda x: 'Month-to-Month' if x==0 else 'One Year' if x==1 else 'Two Year')
    PaperlessBilling = st.selectbox('Paperless Billing', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    PaymentMethod = st.selectbox('Payment Method', [0, 1, 2, 3], format_func=lambda x: 'Bank Transfer' if x==0 else 'Credit Card' if x==1 else 'Electronic Check' if x==2 else 'Mailed Check')

    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }
    return pd.DataFrame(data, index=[0])

data_input = user_input()

if st.button('Predict Churn'):
    prediction = model.predict(data_input)[0]
    proba = model.predict_proba(data_input)[0][1]
    if prediction == 1:
        st.error(f'This customer is likely to churn (Probability: {proba:.2f})')
    else:
        st.success(f'This customer is unlikely to churn (Probability of staying: {1 - proba:.2f})')

if st.checkbox('Show Input Data'):
    st.write(data_input)
