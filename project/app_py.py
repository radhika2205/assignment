# -*- coding: utf-8 -*-
# app_py.py

import streamlit as st
import pickle
import pandas as pd
import os

# ------------------ Load Models Safely ------------------
BASE_DIR = os.path.dirname(__file__)

severity_model_path = os.path.join(BASE_DIR, "accuracy_sevirity.pkl")
alert_model_path = os.path.join(BASE_DIR, "aelert_generated.pkl")

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
    input_data = pd.DataFrame(
        [[temperature, humidity, vehicle_count, avg_speed, visibility, accident_occurred]],
        columns=[
            "temperature",
            "humidity",
            "Vehicle_Count",
            "Avg_Speed(km/h)",
            "Visibility(m)",
            "Accident_Occurred"
        ]
    )

    severity = severity_model.predict(input_data)
    alert = alert_model.predict(input_data)

    st.success(f"🚦 Accident Severity: {severity[0]}")
    st.warning(f"⚠️ Alert Status: {alert[0]}")
