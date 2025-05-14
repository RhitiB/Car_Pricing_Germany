import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("\U0001F697 Car Price Estimator")

# --- Country & Static Currency Rates (Streamlit Cloud Friendly) ---
currency_rates = {
    "Germany": ("EUR", "€", 1.0),
    "India": ("INR", "₹", 90.0),
    "USA": ("USD", "$", 1.1),
    "UK": ("GBP", "£", 0.85),
    "Japan": ("JPY", "¥", 150.0),
    "Australia": ("AUD", "A$", 1.65),
    "Canada": ("CAD", "C$", 1.45),
    "France": ("EUR", "€", 1.0),
    "China": ("CNY", "¥", 7.8),
    "Brazil": ("BRL", "R$", 5.4)
}

country = st.selectbox("\U0001F30D Select Country", list(currency_rates.keys()))
currency_code, symbol, rate = currency_rates[country]

# --- Brand Logos and Premium Multipliers ---
premium_multipliers = {
    "bmw": 1.10,
    "audi": 1.08,
    "mercedes-benz": 1.12,
    "porsche": 1.15,
    "ferrari": 1.25,
    "bentley": 1.3,
    "lamborghini": 1.35
}

brand_logos = {
    "bmw": "https://1000logos.net/wp-content/uploads/2018/02/BMW-Logo-768x432.png",
    "audi": "https://1000logos.net/wp-content/uploads/2018/02/Audi-Logo-768x432.png",
    "mercedes-benz": "https://1000logos.net/wp-content/uploads/2018/02/Mercedes-Benz-Logo-768x432.png",
    "porsche": "https://1000logos.net/wp-content/uploads/2018/02/Porsche-Logo-768x432.png",
    "ferrari": "https://1000logos.net/wp-content/uploads/2018/02/Ferrari-Logo-768x432.png",
    "lamborghini": "https://1000logos.net/wp-content/uploads/2018/02/Lamborghini-Logo-768x432.png"
}

# --- Inputs ---
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)

power_unit = st.selectbox("Select Power Unit", ["PS", "kW"])
power_value = st.number_input(f"Engine Power ({power_unit})", min_value=1.0)
power_ps = power_value * 1.36 if power_unit == "kW" else power_value

mileage_unit = st.selectbox("Select Mileage Unit", ["km", "miles"])
mileage_value = st.number_input(f"Mileage ({mileage_unit})", min_value=1.0)
mileage_km = mileage_value * 1.60934 if mileage_unit == "miles" else mileage_value

fuel_unit = st.selectbox("Fuel Consumption Unit", ["g/km", "km/L", "MPG"])
fuel_value = st.number_input(f"Fuel Consumption ({fuel_unit})", min_value=0.1)
if fuel_unit == "km/L":
    fuel_g_km = 235 / fuel_value
elif fuel_unit == "MPG":
    fuel_g_km = 235.2 / fuel_value
else:
    fuel_g_km = fuel_value

# --- Prediction ---
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'year': [year],
        'power_ps': [power_ps],
        'fuel_consumption_g_km': [fuel_g_km],
        'mileage_in_km': [mileage_km]
    })

    input_scaled = scaler.transform(input_df)
    base_price_eur = model.predict(input_scaled)[0]

    # Choose default brand logic (optional static mapping or skip)
    predicted_brand = "bmw" if base_price_eur > 40000 else "hyundai"
    brand_multiplier = premium_multipliers.get(predicted_brand.lower(), 1.0)
    final_price = base_price_eur * brand_multiplier * rate

    if predicted_brand.lower() in brand_logos:
        st.image(brand_logos[predicted_brand.lower()], width=120)

    st.success(f"\U0001F4B0 Estimated Price for **{predicted_brand.title()}** in **{country}**: {symbol}{final_price:,.2f}")
    st.caption(f"\U0001F50E Base price in EUR: €{base_price_eur:,.2f} (Brand: {predicted_brand.title()} × {brand_multiplier})")
