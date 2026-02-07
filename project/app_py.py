# -*- coding: utf-8 -*-
# app_py.py

import streamlit as st
import pickle
import pandas as pd
import os

# ------------------ Load Models Safely ------------------
BASE_DIR = os.path.dirname(__file__)

severity_model_path = os.path.join(BASE_DIR, "accuracysevirity(1).pkl")
alert_model_path = os.path.join(BASE_DIR, "aelertgenerated(1).pkl")

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

    # -------- Severity Model Input --------
    sev_features = severity_model.feature_names_in_
    sev_dict = {f: 0 for f in sev_features}

    if "temperature" in sev_dict:
        sev_dict["temperature"] = temperature
    if "humidity" in sev_dict:
        sev_dict["humidity"] = humidity
    if "vehicle_count" in sev_dict:
        sev_dict["vehicle_count"] = vehicle_count
    if "avg_speed" in sev_dict:
        sev_dict["avg_speed"] = avg_speed
    if "visibility" in sev_dict:
        sev_dict["visibility"] = visibility
    if "accident_occurred" in sev_dict:
        sev_dict["accident_occurred"] = accident_occurred

    severity_input = pd.DataFrame([sev_dict])
    severity = severity_model.predict(severity_input)

    # -------- Alert Model Input --------
    alert_features = alert_model.feature_names_in_
    alert_dict = {f: 0 for f in alert_features}

    if "temperature" in alert_dict:
        alert_dict["temperature"] = temperature
    if "humidity" in alert_dict:
        alert_dict["humidity"] = humidity
    if "vehicle_count" in alert_dict:
        alert_dict["vehicle_count"] = vehicle_count
    if "avg_speed" in alert_dict:
        alert_dict["avg_speed"] = avg_speed
    if "visibility" in alert_dict:
        alert_dict["visibility"] = visibility
    if "accident_occurred" in alert_dict:
        alert_dict["accident_occurred"] = accident_occurred

    alert_input = pd.DataFrame([alert_dict])
    alert = alert_model.predict(alert_input)

    st.success(f"🚦 Accident Severity: {severity[0]}")
    st.warning(f"⚠️ Alert Status: {alert[0]}")
