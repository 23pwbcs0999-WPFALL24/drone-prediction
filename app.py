import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon, Circle

# Page Config
st.set_page_config(page_title="Drone Drop Simulator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Professional Styling
st.markdown("""
<style>
    /* Main App Background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .header-container h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .header-container p {
        font-size: 1.1em;
        margin-top: 10px;
        opacity: 0.95;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4em;
        font-weight: 700;
        color: #333;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info/Success/Warning Boxes */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    .success-box {
        border-left-color: #10b981;
    }
    
    .warning-box {
        border-left-color: #f59e0b;
    }
    
    /* Footer */
    .footer-container {
        text-align: center;
        padding: 30px 20px;
        margin-top: 40px;
        border-top: 2px solid #e0e0e0;
        color: #666;
        font-size: 0.95em;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        color: white;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    [data-testid="stSidebar"] .stSlider {
        padding: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load ML Components with Caching (efficient resource management)
@st.cache_resource
def load_ml_components():
    """Load model, scaler, and transformer into memory once at app startup."""
    m = joblib.load('drone_model.pkl')
    s = joblib.load('scaler.pkl')
    p = joblib.load('poly_transformer.pkl')
    return m, s, p

model, scaler, poly = load_ml_components()

# Professional Header
st.markdown("""
<div class="header-container">
    <h1>🛸 Autonomous Drone Supply Drop Simulator</h1>
    <p>AI-Powered Precision Targeting with Real-Time Physics Simulation</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown('<h3 style="color: white; margin-top: 20px;">✈️ Flight Configuration</h3>', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 20px 0; border: 1px solid rgba(255,255,255,0.2);">', unsafe_allow_html=True)
    
    st.markdown('<p style="color: rgba(255,255,255,0.9); font-weight: 500; margin-bottom: 10px;">📍 Altitude (meters)</p>', unsafe_allow_html=True)
    alt = st.slider("Altitude", 5.0, 100.0, 30.0, step=5.0, label_visibility="collapsed")
    
    st.markdown('<p style="color: rgba(255,255,255,0.9); font-weight: 500; margin-bottom: 10px;">💨 Ground Speed (m/s)</p>', unsafe_allow_html=True)
    gs = st.slider("Ground Speed", 5.0, 50.0, 15.0, step=2.0, label_visibility="collapsed")
    
    st.markdown('<p style="color: rgba(255,255,255,0.9); font-weight: 500; margin-bottom: 10px;">📦 Payload Mass (kg)</p>', unsafe_allow_html=True)
    payload = st.slider("Payload Mass", 0.5, 50.0, 5.0, step=1.0, label_visibility="collapsed")
    
    st.markdown('<p style="color: rgba(255,255,255,0.9); font-weight: 500; margin-bottom: 10px;">💨 Wind Factor</p>', unsafe_allow_html=True)
    wind_factor = st.slider("Wind Factor", 0.5, 2.0, 1.0, step=0.1, label_visibility="collapsed")
    
    st.markdown('<hr style="margin: 20px 0; border: 1px solid rgba(255,255,255,0.2);">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color: rgba(255,255,255,0.85); padding: 15px; background: rgba(0,0,0,0.1); border-radius: 8px; font-size: 0.9em;">
    <strong>📊 Current Config:</strong><br>
    Alt: <strong>{alt:.1f}m</strong> | Speed: <strong>{gs:.1f}m/s</strong><br>
    Mass: <strong>{payload:.1f}kg</strong> | Wind: <strong>{wind_factor:.1f}x</strong>
    </div>
    """, unsafe_allow_html=True)

# --- CALCULATIONS ---
g = 9.81
physics_drift = gs * np.sqrt((2 * alt) / g)

# ML Prediction (✅ SGD + Ridge with Polynomial Features)
try:
    # Step 1: Scale features (use DataFrame to avoid sklearn warnings)
    X_input = pd.DataFrame([[alt, gs, payload]], columns=['alt', 'gs', 'payload_mass'])
    X_scaled = scaler.transform(X_input)
    
    # Step 2: Apply polynomial features
    X_poly = poly.transform(X_scaled)
    
    # Step 3: Predict with Ridge model
    ml_drift = model.predict(X_poly)[0]
    ml_drift_adjusted = ml_drift * wind_factor
except Exception as e:
    # Fallback to physics if model fails
    ml_drift = physics_drift * 1.15
    ml_drift_adjusted = ml_drift * wind_factor
    st.warning(f"⚠️ Using physics fallback: {str(e)}")

# Fall time calculation
fall_time = np.sqrt((2 * alt) / g)

# --- MAIN VISUALIZATION ---
st.markdown('<div class="section-header">📡 Real-Time Trajectory Simulation</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2.5, 1])

with col1:
    fig, ax = plt.subplots(figsize=(13, 8), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    # --- GROUND ---
    ax.axhline(y=0, color='#8B4513', linewidth=4, label='Ground', zorder=1)
    ax.fill_between([-10, max(physics_drift, ml_drift_adjusted) + 20], -2, 0, 
                     color='#D2B48C', alpha=0.4)
    
    # --- DRONE START POSITION ---
    drone_x, drone_y = 0, alt
    
    # Draw drone at release point
    drone_size = 3
    drone = patches.Rectangle((drone_x - drone_size/2, drone_y - 1), drone_size, 2, 
                               color='#667eea', alpha=0.9, label='Drone')
    ax.add_patch(drone)
    
    # Draw release point marker
    ax.scatter([drone_x], [drone_y], color='#667eea', s=250, marker='^', 
               zorder=5, label='Release Point', edgecolors='#764ba2', linewidth=2.5)
    
    # --- PACKET TRAJECTORY (PARABOLIC) ---
    num_frames = 50
    t_array = np.linspace(0, fall_time, num_frames)
    
    # Physics-based packet motion
    x_physics = physics_drift * (t_array / fall_time)
    y_physics = alt - 0.5 * g * t_array**2
    
    # ML-predicted trajectory (with wind adjustment)
    x_ml = ml_drift_adjusted * (t_array / fall_time)
    y_ml = alt - 0.5 * g * t_array**2
    
    # Plot trajectories with improved styling
    ax.plot(x_physics, y_physics, 'o--', linewidth=2.5, alpha=0.8, 
            color='#f59e0b', label='Physics Formula', markersize=3)
    ax.plot(x_ml, y_ml, 'o-', linewidth=3.5, alpha=0.9, 
            color='#10b981', label='ML Prediction', markersize=3)
    
    # --- LANDING ZONES ---
    # Physics landing zone
    zone_width = 3
    physics_zone = Rectangle((physics_drift - zone_width/2, -1.5), zone_width, 1.5, 
                             color='#f59e0b', alpha=0.15, label='Physics Landing Zone', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(physics_zone)
    ax.scatter([physics_drift], [0], color='#f59e0b', s=400, marker='X', 
               edgecolors='#d97706', linewidth=2.5, zorder=10)
    
    # ML landing zone
    ml_zone = Rectangle((ml_drift_adjusted - zone_width/2, -1.5), zone_width, 1.5, 
                        color='#10b981', alpha=0.2, label='ML Landing Zone', edgecolor='#10b981', linewidth=2)
    ax.add_patch(ml_zone)
    ax.scatter([ml_drift_adjusted], [0], color='#10b981', s=400, marker='*', 
               edgecolors='#059669', linewidth=2.5, zorder=10)
    
    # --- PACKET AT LANDING ---
    packet_size = 1.5
    packet_physics = patches.Rectangle((physics_drift - packet_size/2, -1), packet_size, 1, 
                                       color='#f59e0b', alpha=0.7, edgecolor='#d97706', linewidth=1.5)
    ax.add_patch(packet_physics)
    
    packet_ml = patches.Rectangle((ml_drift_adjusted - packet_size/2, -1), packet_size, 1, 
                                  color='#10b981', alpha=0.7, edgecolor='#059669', linewidth=1.5)
    ax.add_patch(packet_ml)
    
    # --- FORMATTING ---
    ax.set_xlim(-5, max(physics_drift, ml_drift_adjusted) + 15)
    ax.set_ylim(-3, alt + 10)
    ax.set_xlabel("Horizontal Distance (meters)", fontsize=13, fontweight='bold', color='#333')
    ax.set_ylabel("Altitude (meters)", fontsize=13, fontweight='bold', color='#333')
    ax.set_title(f"🎯 Supply Drop Mission | Alt: {alt:.0f}m | Speed: {gs:.0f}m/s | Payload: {payload:.0f}kg", 
                fontsize=14, fontweight='bold', color='#333', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='#ccc', fancybox=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    # Dynamic aspect ratio for better visualization (altitude vs distance can vary greatly)
    # ax.set_aspect('equal', adjustable='box')  # Removed: causes vertical squishing on varied scales
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#ccc')
        spine.set_linewidth(1.5)
    
    st.pyplot(fig, use_container_width=True)

with col2:
    st.markdown('<div style="padding: 20px;"></div>', unsafe_allow_html=True)
    
    # Performance Metrics Cards
    st.markdown('<h4 style="text-align: center; color: #333;">⚡ Performance Metrics</h4>', unsafe_allow_html=True)
    
    # Physics Metric
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #FEF3C7 0%, #FED7AA 100%);">
        <div style="font-size: 0.9em; color: #92400e; font-weight: 600; margin-bottom: 5px;">📐 Physics Formula</div>
        <div style="font-size: 2em; font-weight: 700; color: #f59e0b;">{physics_drift:.1f}m</div>
        <div style="font-size: 0.85em; color: #b45309; margin-top: 5px;">Baseline Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    
    # ML Metric
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);">
        <div style="font-size: 0.9em; color: #166534; font-weight: 600; margin-bottom: 5px;">🤖 ML Prediction</div>
        <div style="font-size: 2em; font-weight: 700; color: #10b981;">{ml_drift_adjusted:.1f}m</div>
        <div style="font-size: 0.85em; color: #059669; margin-top: 5px;">Adjusted for Conditions</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    
    # Correction Analysis
    correction_pct = ((ml_drift_adjusted / physics_drift - 1) * 100)
    correction_color = "#10b981" if abs(correction_pct) < 30 else "#f59e0b"
    
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #E0E7FF 0%, #DDD6FE 100%);">
        <div style="font-size: 0.9em; color: #3730a3; font-weight: 600; margin-bottom: 5px;">📊 Correction Factor</div>
        <div style="font-size: 2em; font-weight: 700; color: {correction_color};">{correction_pct:+.1f}%</div>
        <div style="font-size: 0.85em; color: #6366f1; margin-top: 5px;">ML Adjustment Applied</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Parameters Section
    st.markdown('<div class="section-header">🔍 Mission Parameters & Analysis</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box success-box">
    <strong>📐 Physics-Based Calculation:</strong><br>
    Formula: <code>d = v × √(2h/g)</code><br><br>
    <table style="width: 100%; font-size: 0.9em;">
    <tr><td>Ground Speed (v)</td><td style="text-align: right;"><strong>{gs:.2f} m/s</strong></td></tr>
    <tr><td>Altitude (h)</td><td style="text-align: right;"><strong>{alt:.2f} m</strong></td></tr>
    <tr><td>Gravity (g)</td><td style="text-align: right;"><strong>{g:.2f} m/s²</strong></td></tr>
    <tr><td>Fall Time √(2h/g)</td><td style="text-align: right;"><strong>{fall_time:.3f}s</strong></td></tr>
    <tr style="border-top: 2px solid #d4d4d4;"><td><strong>Physics Drift</strong></td><td style="text-align: right;"><strong style="font-size: 1.1em; color: #f59e0b;">{physics_drift:.2f}m</strong></td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    # ML Analysis
    st.markdown(f"""
    <div class="info-box success-box" style="border-left-color: #10b981;">
    <strong>🤖 Machine Learning Refinement:</strong><br><br>
    <table style="width: 100%; font-size: 0.9em;">
    <tr><td>Raw ML Prediction</td><td style="text-align: right;"><strong>{ml_drift:.2f}m</strong></td></tr>
    <tr><td>Wind Factor Applied</td><td style="text-align: right;"><strong>{wind_factor:.2f}x</strong></td></tr>
    <tr><td style="color: #666; font-size: 0.85em;"><em>Wind multiplier accounts for real-world environmental conditions outside the core ML model</em></td></tr>
    <tr style="border-top: 2px solid #d4d4d4;"><td><strong>Final ML Prediction</strong></td><td style="text-align: right;"><strong style="font-size: 1.1em; color: #10b981;">{ml_drift_adjusted:.2f}m</strong></td></tr>
    <tr><td colspan="2"><em>← Recommended for targeting</em></td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison
    accuracy_score = min(100, 100 - abs((ml_drift_adjusted - physics_drift) / max(physics_drift, ml_drift_adjusted)) * 50)
    difference_m = abs(ml_drift_adjusted - physics_drift)
    difference_pct = ((ml_drift_adjusted / physics_drift - 1) * 100)
    
    col_cmp1, col_cmp2 = st.columns(2)
    
    with col_cmp1:
        st.markdown(f"""
        <div class="metric-card">
        <div style="font-size: 0.9em; color: #666; font-weight: 500;">📏 Absolute Difference</div>
        <div style="font-size: 1.8em; font-weight: 700; color: #667eea;">{difference_m:.2f}m</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_cmp2:
        st.markdown(f"""
        <div class="metric-card">
        <div style="font-size: 0.9em; color: #666; font-weight: 500;">📊 Percentage Adjustment</div>
        <div style="font-size: 1.8em; font-weight: 700; color: #667eea;">{difference_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# --- ADVANCED INSIGHTS ---
st.markdown('<div class="section-header">⚡ Advanced Insights & Recommendations</div>', unsafe_allow_html=True)

col_ins1, col_ins2, col_ins3 = st.columns(3)

with col_ins1:
    confidence = min(100, 100 - abs((ml_drift_adjusted - physics_drift) / max(physics_drift, ml_drift_adjusted)) * 50)
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #667eea;">
    <div style="font-size: 0.9em; color: #666; font-weight: 600;">🎯 Confidence Score</div>
    <div style="font-size: 2.2em; font-weight: 700; color: #667eea;">{confidence:.0f}%</div>
    <div style="margin-top: 10px; font-size: 0.85em; color: #667eea; font-weight: 500;">
    {"✅ Excellent" if confidence > 95 else "✅ Good" if confidence > 85 else "⚠️ Acceptable"}
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_ins2:
    status_color = "#10b981" if wind_factor < 1.3 else "#f59e0b" if wind_factor < 1.6 else "#ef4444"
    status_text = "🟢 Optimal" if wind_factor < 1.3 else "🟡 Moderate" if wind_factor < 1.6 else "🔴 High"
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {status_color};">
    <div style="font-size: 0.9em; color: #666; font-weight: 600;">💨 Wind Conditions</div>
    <div style="font-size: 2.2em; font-weight: 700; color: {status_color};">{wind_factor:.2f}x</div>
    <div style="margin-top: 10px; font-size: 0.85em; color: {status_color}; font-weight: 500;">
    {status_text}
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_ins3:
    model_accuracy = 99.96  # From your validation
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #10b981;">
    <div style="font-size: 0.9em; color: #666; font-weight: 600;">📈 Model Accuracy</div>
    <div style="font-size: 2.2em; font-weight: 700; color: #10b981;">{model_accuracy:.2f}%</div>
    <div style="margin-top: 10px; font-size: 0.85em; color: #10b981; font-weight: 500;">
    ✅ Production Ready
    </div>
    </div>
    """, unsafe_allow_html=True)

# Recommendation
st.markdown('<h4 style="color: #333; margin-top: 25px;">🎯 Mission Recommendation</h4>', unsafe_allow_html=True)

if wind_factor > 1.5:
    rec_color = "#ef4444"
    rec_text = "⚠️ **WARNING: High wind conditions detected!**"
    rec_detail = f"ML model adjusts prediction by {difference_pct:+.1f}% due to severe wind. Consider reviewing wind factor or delaying mission."
elif wind_factor > 1.2:
    rec_color = "#f59e0b"
    rec_text = "⚠️ **CAUTION: Moderate wind conditions**"
    rec_detail = f"ML model makes {abs(difference_pct):.1f}% adjustment. Use caution when applying corrections."
else:
    rec_color = "#10b981"
    rec_text = "✅ **OPTIMAL CONDITIONS FOR DEPLOYMENT**"
    rec_detail = f"Use ML prediction ({ml_drift_adjusted:.1f}m) for maximum accuracy. Confidence score: {confidence:.0f}%"

st.markdown(f"""
<div style="background: {rec_color}15; border-left: 5px solid {rec_color}; padding: 20px; border-radius: 8px;">
<div style="font-size: 1.1em; font-weight: 700; color: {rec_color}; margin-bottom: 10px;">{rec_text}</div>
<div style="color: #333; font-size: 0.95em; line-height: 1.6;">{rec_detail}</div>
</div>
""", unsafe_allow_html=True)

# --- TECHNICAL DETAILS EXPANDER ---
with st.expander("📚 Technical Details & Model Information"):
    st.markdown("""
    **🔬 How the Prediction Works:**
    
    1. **Physics Foundation**: Uses projectile motion equations to calculate baseline drift
    2. **Feature Engineering**: Inputs (altitude, ground speed, payload) are transformed using polynomial features
    3. **Standardization**: Features normalized using StandardScaler for numerical stability
    4. **ML Prediction**: Ridge Regression with regularization provides non-linear adjustments
    5. **Wind Adjustment**: Final prediction scaled by wind factor for environmental conditions
    
    **📊 Model Specifications:**
    - **Algorithm**: Ridge Regression (L2 regularization)
    - **Optimization**: SGD (Stochastic Gradient Descent) with 100 epochs
    - **Feature Count**: 9 (original 3 + polynomial combinations)
    - **Training Dataset**: 800,000 samples
    - **Test Accuracy**: R² = 99.96%
    - **Average Error**: ±0.11 meters
    
    **🎓 Based On**: ML Lectures 01-08 (Regularization, Gradient Descent, Polynomial Features)
    """)

st.divider()

# --- FOOTER ---
st.markdown("""
<div class="footer-container">
<p style="margin: 0; font-weight: 600; color: #333;">🛸 Autonomous Drone Supply Drop Simulator</p>
<p style="margin: 5px 0 0 0; color: #999; font-size: 0.9em;">Powered by Machine Learning | Physics-Based Validation | Real-Time Optimization</p>
<p style="margin: 10px 0 0 0; color: #bbb; font-size: 0.85em;">Built with Streamlit & Scikit-Learn | Production Model: SGD + Ridge Regression</p>
</div>
""", unsafe_allow_html=True)
