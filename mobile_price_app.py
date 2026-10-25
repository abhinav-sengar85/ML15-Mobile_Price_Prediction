import streamlit as st
import pandas as pd
import numpy as np
import joblib

## Load the trained model and scaler
mobile_price_model = joblib.load('mobile_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Mobile Price Prediction App")

ppi = st.number_input("Enter the PPI (Pixels Per Inch):", min_value=0)
cpu_core = st.number_input("Enter the number of CPU cores:", min_value=1)
cpu_freq = st.number_input("Enter the CPU frequency (in GHz):", min_value=0.0, format="%.2f")
internal_mem = st.number_input("Enter the internal memory (in GB):", min_value=0)
ram = st.number_input("Enter the RAM (in GB):", min_value=0)
RearCam = st.number_input("Enter the rear camera resolution (in MP):", min_value=0)
Front_Cam = st.number_input("Enter the front camera resolution (in MP):", min_value=0)
battery = st.number_input("Enter the battery power (in mAh):", min_value=0)
thickness = st.number_input("Enter the thickness (in mm):", min_value=0.0, format="%.2f")

if st.button("Predict Price Range"):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'ppi': [ppi],
        'cpu_core': [cpu_core],
        'cpu_freq': [cpu_freq],
        'internal_mem': [internal_mem],
        'ram': [ram],
        'RearCam': [RearCam],
        'Front_Cam': [Front_Cam],
        'battery': [battery],
        'thickness': [thickness]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = mobile_price_model.predict(input_data_scaled)

    # Display the predicted price range
    st.success(f"The predicted price range is: {prediction[0]}")

