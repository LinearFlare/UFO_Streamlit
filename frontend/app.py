"""Streamlit frontend for UFO sighting prediction (calls FastAPI backend)."""
import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "https://ufo-streamlit.onrender.com")

st.title("UFO Sighting Predictor")
st.write("Predict which country a UFO sighting is likely from based on duration, latitude, and longitude.")

seconds = st.number_input("Duration (seconds)", min_value=1, max_value=60, value=30)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.0, format="%.4f")
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-100.0, format="%.4f")

if st.button("Predict"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict/",
            json={"seconds": seconds, "latitude": latitude, "longitude": longitude}
        )
        data = response.json()
        st.success(f"Likely country: **{data['prediction']}**")

        st.write("### Probability Distribution")
        st.bar_chart(data['probabilities'])
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {BACKEND_URL}. Is the backend running?")