import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Success Predictor", layout="centered")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_student = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kd54v9z6.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model (4).pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st_lottie(lottie_student, height=200, key="coding")
st.title("🎓 Student Success Predictor")
st.write("Enter the student details below to predict their academic outcome.")

# --- INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=[0, 1], help="0 for Female, 1 for Male")
        attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
        study_hours = st.number_input("Study Hours Per Week", 0, 100, 15)
        prev_grade = st.slider("Previous Grade", 0, 100, 75)
        
    with col2:
        extra_curr = st.selectbox("Extracurricular Activities", options=[0, 1], help="0: No, 1: Yes")
        parent_support = st.selectbox("Parental Support", options=[0, 1, 2], help="0: Low, 1: Med, 2: High")
        final_grade = st.slider("Current Final Grade Mock", 0, 100, 70)
        study_hrs_alt = st.number_input("Study Hours (Daily Avg)", 0, 24, 3)
        att_perc_alt = st.number_input("Attendance (%) Detailed", 0.0, 100.0, 85.0)

# --- PREDICTION ---
if st.button("Analyze Performance"):
    # Create features array matching the model's 9 inputs
    features = np.array([[
        gender, attendance_rate, study_hours, prev_grade, 
        extra_curr, parent_support, final_grade, study_hrs_alt, att_perc_alt
    ]])
    
    prediction = model.predict(features)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.balloons()
        st.success(f"### Result: Positive Outcome Predicted!")
    else:
        st.warning(f"### Result: Improvement Needed.")
