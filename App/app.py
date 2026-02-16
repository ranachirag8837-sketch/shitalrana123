# ==========================================
# STUDENT RESULT PREDICTION WEB APP
# ==========================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="🎓 Student Result Prediction AI",
    layout="wide"
)

# ==========================================
# Custom CSS Styling
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

.stApp {
    background-color: #F8FAFC;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] label {
    color: white !important;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stSidebar"] h2 {
    color: white !important;
}

/* Title */
.main-title {
    background-color: #BDBCB8;
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    border: 1px solid #E2E8F0;
    text-align: center;
    margin-bottom: 2rem;
}

.main-title h1 {
    color: #1E293B;
    font-weight: 800;
    font-size: 2.5rem;
    margin: 0;
}

/* Result Box */
.result-box {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    border: 1px solid #E2E8F0;
    border-top: 6px solid #6366F1;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    margin: 1.5rem 0;
}

.result-box span {
    font-size: 3rem;
    font-weight: 800;
    display: block;
    margin: 10px 0;
}

/* Recommendation Box */
.custom-recommend {
    font-family: 'Space Grotesk', sans-serif;
    background: #ffffff;
    padding: 25px;
    border-radius: 18px;
    border-left: 6px solid #6366F1;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    margin-top: 15px;
}

.custom-recommend:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.1);
}

.status-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 15px;
}

.suggest-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #334155;
}

.custom-recommend ul {
    padding-left: 20px;
    line-height: 1.8;
    color: #475569;
}

.goal-text {
    margin-top: 12px;
    font-weight: 700;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.75rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"><h1>🎓 Student Result Prediction System</h1></div>', unsafe_allow_html=True)

# ==========================================
# Load Models
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Model"

try:
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    models = joblib.load(MODEL_DIR / "models.pkl")
    top_models = joblib.load(MODEL_DIR / "top_models.pkl")
    scores = joblib.load(MODEL_DIR / "scores.pkl")
except:
    st.error("❌ Model files not found!")
    st.stop()

# ==========================================
# Sidebar Input
# ==========================================

st.sidebar.header("📌 Enter Student Details")

study_hours = st.sidebar.number_input("Study Hours (0-12)", 0.0, 12.0, value=5.0)
attendance = st.sidebar.number_input("Attendance % (0-100)", 0.0, 100.0, value=75.0)
internal = st.sidebar.number_input("Internal Marks (0-50)", 0, 50, value=30)
assignment = st.sidebar.number_input("Assignment Score (0-20)", 0, 20, value=15)

predict = st.sidebar.button("Predict Result")

# ==========================================
# Attendance Recommendation Function
# ==========================================

def generate_recommendation(attendance):

    attendance = float(attendance)

    if attendance < 30:
        status = "FAIL"
        color = "#e11d48"
        suggestions = [
            "Very low attendance",
            "Attend classes regularly",
            "Meet academic advisor"
        ]
        goal = "🎯 Immediate Action Required"

    elif attendance < 60:
        status = "AVERAGE"
        color = "#f59e0b"
        suggestions = [
            "Improve attendance consistency",
            "Avoid unnecessary leaves",
            "Maintain minimum 75%"
        ]
        goal = "🎯 Goal: Reach 70+"

    elif attendance < 80:
        status = "GOOD"
        color = "#2563eb"
        suggestions = [
            "Maintain regular class presence",
            "Participate in activities",
            "Stay consistent"
        ]
        goal = "🎯 Goal: Reach 85+"

    else:
        status = "EXCELLENT"
        color = "#059669"
        suggestions = [
            "Outstanding attendance record",
            "Keep up the discipline",
            "Be a role model"
        ]
        goal = "🎯 Goal: Maintain 100%"

    return f"""<div class="custom-recommend">
    <div class="status-title" style="color:{color};">
    ⭐ Status: {status}
    </div>

    <div class="suggest-title">📌 Suggestions</div>

    <ul>
    {''.join(f"<li>{item}</li>" for item in suggestions)}
    </ul>

    <div class="goal-text" style="color:{color};">
    {goal}
    </div>
    </div>"""

# ==========================================
# Prediction Section
# ==========================================

if predict:

    input_data = np.array([[study_hours, attendance, internal, assignment]])
    input_scaled = scaler.transform(input_data)

    st.subheader("📊 Individual Model Predictions")

    results_data = []
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        results_data.append({
            "Model": name,
            "Predicted Marks": round(pred, 2),
            "Result": "PASS ✅" if pred >= 40 else "FAIL ❌",
            "Accuracy": round(scores.get(name, 0), 4)
        })

    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True)

    hybrid_final = np.mean([models[name].predict(input_scaled)[0] for name in top_models])
    hybrid_final = np.clip(hybrid_final, 0, 100)
    hybrid_final = round(hybrid_final, 2)

    st.subheader("🤖 Hybrid Model Prediction (Top 3 Ensemble)")
    st.markdown(f"""
    <div class="result-box">
        <p style="color: #64748B; font-weight: 600;">Average Hybrid Score</p>
        <span style="color: #7c3aed;">{hybrid_final:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Algorithm Score Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_results["Model"], df_results["Predicted Marks"])
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("📘 Personalized Attendance Feedback")
    html(generate_recommendation(attendance), height=300)

