#-*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
labelencoder = LabelEncoder()

# read model
dtc_model = pd.read_pickle('model/finalized_model.sav')

st.header("Predict Customer Churn")

# create 4 columns
fist_column, second_column ,third_column, fourth_column = st.columns(4)

#select option
internet_service_option = { 2: "DSL", 1: "Fiber optic", 0: "No"}
contract_option = { 2: "Month-to-month", 1: "One year", 0: "Two year"}
payment_method_option = { 3: "Electronic check", 2: "Mailed check", 1: "Bank transfer (automatic)",0: "Credit card (automatic)"}

#it makes more sense to catgorize custiomers wrt tenure, so let's convert tenure column to tenure range/buckets
def convert_to_buckets(tenure):
  if tenure < 24:
    return '0 - 24 months'
  elif tenure <= 36:
      return '25 - 36 months'
  elif tenure <= 48:
    return '36 - 48 months'
  elif tenure <= 60:
    return '48 - 60 months'
  else:
    return '> 60 months'

# Function to preprocess the user input
def preprocess_input(new_customer):
    # Convert the input to a dataframe
    input_df = pd.DataFrame([new_customer])

    # Convert tenure to buckets
    input_df['tenure'] = input_df['tenure'].map(convert_to_buckets)

    # Encode categorical features
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = labelencoder.fit_transform(input_df[col])

    return input_df


# Function to predict churn for a single user input
def predict_churn(new_customer):
    preprocessed_input = preprocess_input(new_customer)
    prediction = dtc_model.predict(preprocessed_input)
    return prediction[0]

with fist_column:
    gender = st.selectbox("Gender", ("Male", "Female"))
    senior_citizen_option = st.selectbox("Senior Citizen", ("Yes", "No"))
    partner = st.selectbox("Partner", ("Yes", "No"))
    dependents = st.selectbox("Dependents", ("Yes", "No"))
    tenure = st.number_input("Tenure", min_value=0, max_value=99, step=1)

with second_column:
    phon_service = st.selectbox("Phone Service", ("Yes", "No"))
    multiple_lines = st.selectbox("Multiple Lines", ("Yes", "No"))
    internet_service = st.selectbox("Internet Service", ( 2, 1, 0), format_func=lambda x: internet_service_option[x])
    online_security = st.selectbox("Online Security", ("Yes", "No"))
    online_backup = st.selectbox("Online Backup", ("Yes", "No"))

with third_column:
    device_protection = st.selectbox("Device Protection", ("Yes", "No"))
    tech_support = st.selectbox("Tech Support", ("Yes", "No"))
    streamingTV = st.selectbox("Streaming TV", ("Yes", "No"))
    streaming_movies = st.selectbox("Streaming Movies", ("Yes", "No"))
    contracts = st.selectbox("Contract", ( 2, 1, 0), format_func=lambda x: contract_option[x])
    
with fourth_column:
    paperless_billing = st.selectbox("Paperless Billing", ("Yes", "No"))
    payment_method = st.selectbox("Contract", ( 3, 2, 1,0), format_func=lambda x: payment_method_option[x])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=99.000)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=9999.000)

predict = st.button("Predict")

if predict:
    i_service = internet_service_option[internet_service]
    contract = contract_option[contracts]
    payment_m = payment_method_option[payment_method]
    senior_citizen = 0 if senior_citizen_option== "No" else 1

    new_customer = {'gender':gender,'SeniorCitizen': senior_citizen,'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
                'PhoneService': phon_service,'MultipleLines': multiple_lines,'InternetService':i_service,'OnlineSecurity': online_security,
                'OnlineBackup': online_backup, 'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV':streamingTV ,
                'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_m,
                'MonthlyCharges': monthly_charges, 'TotalCharges':total_charges}
    
    prediction = predict_churn(new_customer)
    print(new_customer)
    if prediction:
        st.success("Predicted results: Churn")
    else:
        st.warning("Predicted results: No churn")

