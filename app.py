# ================== IMPORTS ==================
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ================== PAGE CONFIG (MUST BE FIRST STREAMLIT CALL) ==================
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="centered"
)

# ================== LOAD MODEL ==================
model = joblib.load("model/placement_model.pkl")

# ================== SIDEBAR INPUTS ==================
st.sidebar.header("📋 Student Profile")

# 🎓 Grade Type Selection
grade_type = st.sidebar.radio(
    "🎓 Grade Type",
    ["CGPA", "SGPA"]
)

grade_value = st.sidebar.number_input(
    f"📘 Enter {grade_type}",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.01,
    format="%.2f"
)

# Convert SGPA to CGPA (approx)
cgpa = grade_value * 0.95 if grade_type == "SGPA" else grade_value
backlogs = st.sidebar.number_input("📉 Backlogs", min_value=0, step=1)
internships = st.sidebar.number_input("💼 Internships", min_value=0, step=1)
projects = st.sidebar.number_input("🛠 Projects", min_value=0, step=1)
aptitude = st.sidebar.slider("🧠 Aptitude Score", 0, 100, 60)
communication = st.sidebar.slider("🗣 Communication Skills", 0, 100, 70)
attendance = st.sidebar.slider("📊 Attendance (%)", 0, 100, 75)
certifications = st.sidebar.number_input("📜 Certifications", min_value=0, step=1)
coding = st.sidebar.slider("💻 Coding Skills", 0, 10, 7)

# ================== MAIN UI ==================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🎓 Placement Prediction System</h1>
    <p style='text-align: center;'>Predict your placement chances with smart ML insights</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================== PREDICTION ==================
if st.button("🚀 Predict Placement"):

    input_data = pd.DataFrame([{
        "cgpa": cgpa,
        "backlogs": backlogs,
        "internships": internships,
        "projects": projects,
        "aptitude": aptitude,
        "coding": coding,
        "communication": communication,
        "attendance": attendance,
        "certifications": certifications
    }])

    probability = model.predict_proba(input_data)[0][1]
    chance = int(probability * 100)

    # 🎯 Chance Meter
    if chance >= 80:
        st.success(f"🟢 Excellent! Your placement chance is **{chance}%**")
    elif chance >= 60:
        st.warning(f"🟡 Good! Your placement chance is **{chance}%**")
    else:
        st.error(f"🔴 Low! Your placement chance is **{chance}%**")

    st.progress(chance)

    # 📊 Bar chart
    chart_data = pd.DataFrame({
        "Outcome": ["Not Placed", "Placed"],
        "Probability (%)": [100 - chance, chance]
    })
    st.bar_chart(chart_data.set_index("Outcome"))

    # 🧠 Improvement Tips
    st.subheader("🧠 Personalized Improvement Tips")

    if cgpa < 7:
        st.warning("📘 Improve CGPA by strengthening fundamentals.")
    elif cgpa < 8:
        st.info("📗 Try pushing CGPA above 8.")
    else:
        st.success("📘 Strong CGPA!")

    if backlogs > 0:
        st.warning("📉 Clear backlogs to improve chances.")
    else:
        st.success("✅ No backlogs!")

    if internships == 0:
        st.info("💼 Do at least one internship.")
    elif internships < 2:
        st.success("💼 Good internship exposure.")
    else:
        st.success("💼 Excellent internship profile.")

    if projects < 2:
        st.info("🛠 Build more projects.")
    else:
        st.success("🛠 Strong project experience.")

    if aptitude < 50:
        st.warning("🧠 Improve aptitude with daily practice.")
    elif aptitude < 70:
        st.info("🧠 Practice mock tests.")
    else:
        st.success("🧠 Strong aptitude skills!")