import streamlit as st
from helper import train_model, predict_sleep_disorder
import pandas as pd

st.set_page_config(page_title="REMinder", layout="centered")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 REMinder - Sleep Disorder Predictor")
st.write("Model is training on startup...")

model, preprocess_fn = train_model()

with st.form("prediction_form"):
    age = st.slider("Age", 18, 90, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    occupation = st.selectbox("Occupation", ["Nurse", "Doctor", "Engineer", "Teacher", "Lawyer", "Sales", "Driver", "Scientist", "Artist", "Other"])
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.5)
    physical_activity = st.slider("Physical Activity Level (0–10)", 0, 10, 5)
    stress_level = st.slider("Stress Level (0–10)", 0, 10, 5)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    blood_pressure = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])
    heart_rate = st.slider("Heart Rate", 50, 120, 70)
    daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    caffeine = st.slider("Caffeine Consumption (mg/day)", 0, 500, 100)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Occupation": occupation,
            "Sleep Duration": sleep_duration,
            "Physical Activity Level": physical_activity,
            "Stress Level": stress_level,
            "BMI Category": bmi_category,
            "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate,
            "Daily Steps": daily_steps,
            "Smoking": smoking,
            "Alcohol Consumption": alcohol,
            "Caffeine Consumption": caffeine
        }])

        prediction, advice = predict_sleep_disorder(model, preprocess_fn, input_data)
if submitted:
    input_data = {
        'Age': Age,
        'Gender': Gender,
        'Occupation': Occupation,
        'Sleep Duration': Sleep_Duration,
        'Quality of Sleep': Quality_of_Sleep,
        'Physical Activity Level': Physical_Activity_Level,
        'Stress Level': Stress_Level,
        'BMI Category': BMI_Category,
        'Blood Pressure': Blood_Pressure,
        'Heart Rate': Heart_Rate,
        'Daily Steps': Daily_Steps,
        'Smoking': Smoking,
        'Alcohol Consumption': Alcohol_Consumption,
        'Occupation': Occupation,
        'Sleep Disorder': 'None',
        'Person ID': 99999
    }
    prediction, advice = predict_sleep_disorder(model, preprocess_fn, input_data)
    st.success(f"Predicted Sleep Disorder: {prediction}")
    st.info(advice)