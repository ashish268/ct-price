import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and columns
model = joblib.load("car_price_model.pkl")
columns = joblib.load("car_columns.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict the selling price")

# -------------------------
# User Inputs
# -------------------------
company = st.selectbox("Company", ["Audi", "BMW", "Hyundai", "Maruti", "Honda"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First", "Second", "Third"])

year = st.number_input("Year of Purchase", 2000, 2025, step=1)
kms_driven = st.number_input("Kilometers Driven", 0, 300000, step=500)
engine = st.number_input("Engine (CC)", 500, 5000, step=100)
mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, step=0.5)

# -------------------------
# Predict Button
# -------------------------
if st.button("Predict Price"):
    
    # Create input dataframe
    input_data = {
        "company": company,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "owner": owner,
        "year": year,
        "kms_driven": kms_driven,
        "engine": engine,
        "mileage": mileage
    }

    df = pd.DataFrame([input_data])

    # One-hot encoding
    df_encoded = pd.get_dummies(df)

    # Align columns with training data
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df_encoded)[0]

    st.success(f"💰 Estimated Car Price: ₹ {round(prediction, 2)}")

