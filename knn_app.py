import pickle
import pandas as pd

# Load trained model and scaler
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("🚗 Welcome to Car Price Estimator (KNN Model)")

while True:
    try:
        # Take user inputs
        year = int(input("Enter Year of Manufacture: "))
        power = float(input("Enter Power (PS): "))
        fuel = float(input("Enter Fuel Consumption (g/km): "))
        mileage = float(input("Enter Mileage (in km): "))

        # Prepare DataFrame
        input_df = pd.DataFrame({
            'year': [year],
            'power_ps': [power],
            'fuel_consumption_g_km': [fuel],
            'mileage_in_km': [mileage]
        })

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        price = model.predict(input_scaled)[0]
        print(f"\n💰 Estimated Resale Price: €{price:,.2f}")

    except Exception as e:
        print("❌ Error:", e)

    # Ask to continue
    close = input("\nDo you want to close the program? (yes/no): ").strip().lower()
    if close == 'yes':
        print("✅ Program closed. Thank you!")
        break
    else:
        print("\n🔄 Let's estimate another car!\n")
