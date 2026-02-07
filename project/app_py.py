# -*- coding: utf-8 -*-
# app_py.py

import streamlit as st
import pickle
import pandas as pd
import os

# ------------------ Load Models Safely ------------------
BASE_DIR = os.path.dirname(__file__)

severity_model_path = os.path.join(BASE_DIR, "accuracysevirity.pkl")
alert_model_path = os.path.join(BASE_DIR, "aelertgenerated.pkl")

with open(severity_model_path, "rb") as f:
    severity_model = pickle.load(f)

with open(alert_model_path, "rb") as f:
    alert_model = pickle.load(f)

# ------------------ UI ------------------
st.title("🚨 Accident Prediction System")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
vehicle_count = st.number_input("Vehicle Count")
avg_speed = st.number_input("Average Speed (km/h)")
visibility = st.number_input("Visibility (m)")
accident_occurred = st.selectbox("Accident Occurred", [0, 1])

# ------------------ Prediction ------------------
if st.button("Predict"):

    feature_names = severity_model.feature_names_in_

    # create empty input with all required features = 0
    input_dict = {feature: 0 for feature in feature_names}

    # fill only available inputs
    input_dict["temperature"] = temperature
    input_dict["humidity"] = humidity
    input_dict["vehicle_count"] = vehicle_count
    input_dict["avg_speed"] = avg_speed
    input_dict["visibility"] = visibility
    input_dict["accident_occurred"] = accident_occurred

    input_data = pd.DataFrame([input_dict])

    severity = severity_model.predict(input_data)
    alert = alert_model.predict(input_data)

    st.success(f"🚦 Accident Severity: {severity[0]}")
    st.warning(f"⚠️ Alert Status: {alert[0]}")
