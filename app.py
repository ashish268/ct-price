import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("vot_reg.pkl")

st.set_page_config(page_title="Car Price Prediction")
st.title("🚗 Car Price Prediction")

# ---------------- Inputs ----------------
year = st.number_input("Year", 2000, 2025, step=1)
kms_driven = st.number_input("Kilometers Driven", 0, 300000, step=500)
engine = st.number_input("Engine (CC)", 500, 5000, step=100)
mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, step=0.5)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# ----------- Feature order (VERY IMPORTANT) -----------
FEATURE_COLUMNS = [
    "year",
    "kms_driven",
    "engine",
    "mileage",
    "fuel_type_Diesel",
    "fuel_type_Petrol",
    "fuel_type_CNG",
    "transmission_Manual",
    "transmission_Automatic"
]

if st.button("Predict Price"):
    
    input_data = pd.DataFrame([{
        "year": year,
        "kms_driven": kms_driven,
        "engine": engine,
        "mileage": mileage,
        "fuel_type": fuel_type,
        "transmission": transmission
    }])

    # One-hot encode
    input_encoded = pd.get_dummies(input_data)

    # Add missing columns
    for col in FEATURE_COLUMNS:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Keep correct order
    input_encoded = input_encoded[FEATURE_COLUMNS]

    # Predict
    prediction = model.predict(input_encoded)[0]

    st.success(f"💰 Predicted Car Price: ₹ {round(prediction, 2)}")
