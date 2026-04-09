import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Daily Habits Mood Predictor",
    layout="centered"
)

st.title("🧠 Daily Habits Mood Predictor")
st.write("Enter your daily habits to predict your mood")

# LOAD TRAINED PIPELINE
pipeline = joblib.load("rf_pipeline.pkl")

st.subheader("📋 Daily Habits Input")

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
study_hours = st.slider("Study Hours", 0.0, 12.0, 4.0)
steps = st.number_input("Steps Walked", 0, 30000, 6000)
water_intake = st.slider("Water Intake (ml)", 0, 5000, 2000)

# EXACT SAME FEATURE COUNT AS TRAINING (4)
input_data = np.array([
    [sleep_hours, study_hours, steps, water_intake]
])

if st.button("Predict Mood"):
    prediction = pipeline.predict(input_data)
    st.success(f"😊 Predicted Mood: **{prediction[0]}**")

st.markdown("---")
st.caption("Daily Habits Tracker – Machine Learning Project")
