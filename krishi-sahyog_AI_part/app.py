import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Load trained machine learning model
@st.cache_resource
def load_model():
  return joblib.load("crop_model.pkl")

model = load_model()

# District Database with fixed coordinates and Soil Baselines
district_database = {
    "Madhya Pradesh": {
        "Bhopal": {"lat": 23.2599, "lon": 77.4126, "N": 60, "P": 52, "K": 40, "ph": 7.2},
        "Indore": {"lat": 22.7196, "lon": 75.8577, "N": 55, "P": 48, "K": 42, "ph": 6.8},
        "Jabalpur": {"lat": 23.1815, "lon": 79.9864, "N": 65, "P": 50, "K": 45, "ph": 7.0},
    },
    "Uttar Pradesh": {
        "Varanasi": {"lat": 25.3176, "lon": 82.9739, "N": 70, "P": 45, "K": 50, "ph": 7.5},
        "Lucknow": {"lat": 26.8467, "lon": 80.9462, "N": 62, "P": 48, "K": 44, "ph": 7.1},
    }
}

def fetch_live_weather_forecast(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&timezone=auto"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            daily = data.get("daily", {})
            temps = daily.get("temperature_2m_max", [25.0])
            rains = daily.get("precipitation_sum", [2.0])
            humidities = daily.get("relative_humidity_2m_mean", [70.0])
            
            avg_temp = sum(temps) / len(temps) if temps else 25.0
            total_rain_projection = (sum(rains) / len(rains)) * 30 if rains else 100.0
            avg_humidity = sum(humidities) / len(humidities) if humidities else 75.0
            return avg_temp, total_rain_projection, avg_humidity
    except Exception:
        pass
    return 26.0, 120.0, 75.0

# Streamlit UI Configuration
st.set_page_config(page_title="KRISHI Sahyog - Farm Advisor", page_icon="🌾", layout="centered")

st.title("कृषि सहयाग (KRISHI Sahyog)")
st.subheader("🌾 स्वचालित मौसम और बहु-फसल सलाहकार (Multi-Crop Predictor)")

# Location selection
selected_state = st.selectbox("अपना राज्य चुनें (Select State)", list(district_database.keys()))
available_districts = list(district_database[selected_state].keys())
selected_district = st.selectbox("अपना जिला चुनें (Select District)", available_districts)

if st.button("शीर्ष फसलें और मौसम सलाह देखें (Get Recommendations)"):
    district_info = district_database[selected_state][selected_district]
    
    with st.spinner("लाइव मौसम डेटा प्राप्त किया जा रहा है..."):
        forecast_temp, forecast_rainfall, forecast_humidity = fetch_live_weather_forecast(
            district_info["lat"], district_info["lon"]
        )
    
    input_features = [[
        district_info["N"], district_info["P"], district_info["K"],
        forecast_temp, forecast_humidity, district_info["ph"], forecast_rainfall
    ]]
    
    # Get probabilities for all crops using model.predict_proba
    probabilities = model.predict_proba(input_features)[0]
    classes = model.classes_
    
    # Get indices of top 3 crops with highest probabilities
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    st.markdown("---")
    st.markdown(f"### 📍 स्थान: **{selected_district}, {selected_state}**")
    st.success(f"📡 **ऑटो-फेच्ड मौसम स्थिति:** तापमान: {forecast_temp:.1f}°C | अनुमानित वर्षा: {forecast_rainfall:.1f} मिमी")
    
    st.markdown("### 🌱 आपके लिए शीर्ष 3 सुझाई गई फसलें (Top 3 Recommendations):")
    
    for i, idx in enumerate(top_indices):
        crop_name = classes[idx]
        confidence = probabilities[idx] * 100
        st.markdown(f"**{i+1}. {crop_name.upper()}** — उपयुक्तता स्कोर (Suitability): `{confidence:.1f}%`")
        st.progress(float(probabilities[idx]))

    if forecast_rainfall > 150:
        st.info("💧 **सिंचाई सलाह:** आगामी दिनों में अच्छी बारिश के संकेत हैं। खेतों से जल निकासी का ध्यान रखें और सिंचाई रोकें।")
    else:
        st.warning("💧 **सिंचाई सलाह:** सूखा या कम बारिश का अनुमान है। हल्की सिंचाई या ड्रिप पद्धति का उपयोग करें।")
        
    st.caption("ℹ️ *यह सलाह जिला मिट्टी प्रोफाइल और लाइव मौसम उपग्रह डेटा पर आधारित है।*")