import streamlit as st
import numpy as np
import joblib
import os

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "placement_model.pkl")

model = joblib.load(model_path)

st.set_page_config(page_title="Placement Prediction System")

st.title("🎓 Placement Prediction System")
st.write("Enter student details to predict placement outcome")

# Input fields
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
backlogs = st.number_input("Number of Backlogs", min_value=0, step=1)
internships = st.number_input("Internships Completed", min_value=0, step=1)
projects = st.number_input("Projects Completed", min_value=0, step=1)
aptitude = st.slider("Aptitude Score", 0, 100)
coding = st.slider("Coding Skills (1-10)", 1, 10)
communication = st.slider("Communication Skills (1-10)", 1, 10)
attendance = st.slider("Attendance Percentage", 0, 100)
certifications = st.number_input("Certifications", min_value=0, step=1)

# Prediction button
if st.button("Predict Placement"):
    input_data = np.array([[cgpa, backlogs, internships, projects,
                             aptitude, coding, communication,
                             attendance, certifications]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Student is likely to be PLACED (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Student is likely NOT to be placed (Probability: {probability:.2f})")