import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# -------- CONFIG --------
DEBUG = True 

# -------- Load model & scaler --------
try:
    model = joblib.load("random_forest_model.pkl")
except Exception:
    st.error("Could not load model file 'random_forest_model.pkl'. Make sure file exists.")
    raise

try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    st.error("Could not load scaler file 'scaler.pkl'. Make sure file exists.")
    raise

# -------- Page config --------
st.set_page_config(
    page_title="Solar Power Prediction App",
    layout="wide"
)

# -------- Header --------
page_style = """
<style>
/* Main App BG */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f7f9ff 0%, #eef2ff 100%);
}

/* Sidebar BG */
[data-testid="stSidebar"] {
    background: #dde4ff;
}

/* Gradient Header */
.gradient-header {
    padding: 18px 0px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
    border-radius: 10px;
    background: linear-gradient(90deg, #4b76ff, #6a9bff);
    margin-bottom: 15px;
}

/* Card Box */
.info-box {
    padding: 15px;
    border-radius: 10px;
    background: white;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border-left: 8px solid #4b76ff;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# -------- Header --------
st.markdown("<div class='gradient-header'> Solar Power Prediction App</div>", unsafe_allow_html=True)
st.markdown("Provide environmental inputs to estimate the solar power output (kWh).")

# -------- Sidebar Inputs --------
st.sidebar.header(" Input Features")

distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon", 0.0, 12.0, 6.0, format="%.3f")
temperature = st.sidebar.number_input("Temperature (°C)", -10, 60, 30)
wind_direction = st.sidebar.number_input("Wind Direction (°)", 0, 360, 180)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0, format="%.2f")
sky_cover = st.sidebar.number_input("Sky Cover (0–4)", 0.0, 4.0, 1.0, format="%.2f")
visibility = st.sidebar.number_input("Visibility (km)", 0.0, 50.0, 10.0, format="%.2f")
humidity = st.sidebar.number_input("Humidity (%)", 0, 100, 50)
avg_wind_speed_period = st.sidebar.number_input("Average Wind Speed (period)", 0.0, 40.0, 5.0, format="%.2f")
avg_pressure_period = st.sidebar.number_input("Average Pressure (hPa)", 900.0, 1100.0, 1013.0, format="%.2f")

# Input DF
input_df = pd.DataFrame({
    'distance-to-solar-noon': [distance_to_solar_noon],
    'temperature': [temperature],
    'wind-direction': [wind_direction],
    'wind-speed': [wind_speed],
    'sky-cover': [sky_cover],
    'visibility': [visibility],
    'humidity': [humidity],
    'average-wind-speed-(period)': [avg_wind_speed_period],
    'average-pressure-(period)': [avg_pressure_period]
})

# ------- Show summary box -------
st.subheader(" Input Summary")
st.markdown("<div class='info-box'>Below is a summary of the inputs you entered.</div>", unsafe_allow_html=True)
st.dataframe(input_df)

# -------- Predict Button --------
if st.button(" Predict Solar Power"):
    try:
        scaled_input = scaler.transform(input_df)
        pred = model.predict(scaled_input)[0]

        st.markdown(f"""
            <div class='result-box'>
            <h3> Predicted Power Output: <b>{pred:.2f} kWh</b></h3>
            </div>
        """, unsafe_allow_html=True)

        # -------- Level Categorization --------
        if pred <= 0:
            level = "Very Low"
            color = "red"
            desc = "Almost zero or very low solar power. Conditions are not suitable for generation."
            bar_max = 5000

        elif pred < 2000:
            level = "Low"
            color = "darkred"
            desc = "Low generation — conditions such as cloud cover or low sunlight."
            bar_max = 5000

        elif pred < 10000:
            level = "Moderate"
            color = "orange"
            desc = "Moderate generation — partial sunlight or average weather conditions."
            bar_max = 20000

        else:
            level = "High"
            color = "green"
            desc = "Excellent generation — clear sky and strong sunlight."
            bar_max = max(20000, pred * 1.2)

        # -------- Gauge Meter --------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(pred),
            title={'text': f" Power Level: {level}"},
            gauge={
                'axis': {'range': [0, bar_max]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, bar_max*0.2], 'color': "rgba(255,0,0,0.2)"},
                    {'range': [bar_max*0.2, bar_max*0.6], 'color': "rgba(255,165,0,0.2)"},
                    {'range': [bar_max*0.6, bar_max], 'color': "rgba(0,255,0,0.2)"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # -------- Explanation Box --------
        st.markdown(f"""
            <div class='info-box'>
            <h4> Meaning of this result</h4>
            <p><b>{level} Output:</b> {desc}</p>
            <p>Based on your inputs like temperature, humidity, sky cover and wind speed, 
            the system estimates how much sunlight is available for power generation.</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed. Check console for details.")
        print("Prediction error:", e)

# -------- Sidebar About --------
st.sidebar.markdown("---")
st.sidebar.info("""
###  About  
**Model:** Tuned Random Forest Regressor  
**Dataset:** Solar Power Generation  
**Output:** kWh Prediction  
""")