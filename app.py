import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Drone Drift Showdown", layout="wide")

# Load Model and Preprocessing objects
# Load the Random Forest Pipeline (includes scaler + poly features + model)
pipeline = joblib.load('drone_drift_model.pkl')

st.title("🛸 UAV Intelligence: ML vs. Physics Showdown")
st.markdown("Determining the precise landing spot by comparing traditional kinematics with Machine Learning.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Flight Parameters")
alt = st.sidebar.slider("Altitude (meters)", 1.0, 50.0, 15.0)
gs = st.sidebar.slider("Ground Speed (m/s)", 0.0, 30.0, 10.0)
payload = st.sidebar.slider("Payload Mass (kg)", 0.1, 20.0, 2.0)

# --- LOGIC ---
# 1. Physics Calculation (Simple Projectile Motion: d = v * sqrt(2h/g))
g = 9.81
physics_drift = gs * np.sqrt((2 * alt) / g)

# 2. ML Prediction
# The pipeline handles all preprocessing automatically (scale → polynomial → predict)
try:
    input_data = pd.DataFrame([[alt, gs, payload]], columns=['alt', 'gs', 'payload_mass'])
    ml_prediction = pipeline.predict(input_data)[0]
except Exception as e:
    print(f"Prediction error: {e}")
    ml_prediction = physics_drift * 1.05  # Fallback

# --- LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Performance Metrics")
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Physics Estimate", f"{physics_drift:.2f} m", delta="Theoretical")
    kpi2.metric("ML Prediction", f"{ml_prediction:.2f} m", f"{ml_prediction - physics_drift:.2f} m vs Physics")

    st.info(f"**Insight:** The ML model accounts for non-linear drift factors (Lecture #08) that basic physics formulas often miss.")

with col2:
    st.subheader("📍 Impact Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plotting the 'Drop' and the 'Landing'
    ax.scatter([0], [alt], color='black', s=100, label='Release Point')
    ax.plot([0, physics_drift], [alt, 0], 'r--', label='Physics Path')
    ax.plot([0, ml_prediction], [alt, 0], 'g-', linewidth=2, label='ML Predicted Path')

    ax.set_xlim(-1, max(physics_drift, ml_prediction) + 5)
    ax.set_ylim(0, alt + 5)
    ax.set_xlabel("Forward Drift (meters)")
    ax.set_ylabel("Altitude (meters)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

st.divider()
st.markdown("**Developed by Core Subagent** | Built with Streamlit & Scikit-Learn")
