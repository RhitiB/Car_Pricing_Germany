import streamlit as st
import pickle
import pandas as pd
import time
import os
# Load model and scaler
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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
    st.info("✅ Closing the app in 3 seconds... Please wait.")

    # Force a clean shutdown after prediction

    time.sleep(3)
    os._exit(0)
