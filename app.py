import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("vot_reg.pkl")

st.set_page_config(page_title="Car Price Prediction")

st.title("🚗 Car Price Prediction")

# User Inputs (MATCH THESE WITH TRAINING FEATURES)
year = st.number_input("Year", 2000, 2025, step=1)
kms_driven = st.number_input("Kilometers Driven", 0, 300000, step=500)
engine = st.number_input("Engine (CC)", 500, 5000, step=100)
mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, step=0.5)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

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
    input_data = pd.get_dummies(input_data)

    # Fix missing columns (important!)
    model_features = model.feature_names_in_
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Car Price: ₹ {round(prediction, 2)}")
