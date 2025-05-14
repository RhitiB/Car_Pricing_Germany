import streamlit as st
import joblib
import pandas as pd
import time
import os

# Debugging helper
st.write("Files in directory:", os.listdir())

# Load model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("🚗 Car Price Estimator")

year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
power = st.number_input("Engine Power (PS)", min_value=10, max_value=1000, step=10)
fuel = st.number_input("Fuel Consumption (g/km)", min_value=0.0, max_value=500.0, step=1.0)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, step=1000)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'year': [year],
        'power_ps': [power],
        'fuel_consumption_g_km': [fuel],
        'mileage_in_km': [mileage]
    })

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Estimated Resale Price: €{prediction:,.2f}")
