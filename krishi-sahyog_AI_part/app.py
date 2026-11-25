import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Load trained ML artifact (supports both new bundle dict and legacy raw model)
@st.cache_resource
def load_artifact():
    loaded = joblib.load("crop_model.pkl")
    if isinstance(loaded, dict):
        return loaded
    return {
        "model": loaded,
        "model_name": "Random Forest",
        "feature_cols": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        "test_accuracy": None,
        "macro_f1": None,
        "crop_profiles": {},
    }

artifact = load_artifact()
model = artifact["model"]
feature_cols = artifact["feature_cols"]
crop_profiles = artifact.get("crop_profiles", {})

crop_profiles = artifact.get("crop_profiles", {})

PERENNIAL_CROPS = {
    "apple", "banana", "coconut", "coffee",
    "grapes", "mango", "orange", "papaya", "pomegranate"
}

# Expanded District Database with Soil Baselines & Seasonal Rainfall Norms (mm)
district_database = {
    "Madhya Pradesh": {
        "Bhopal": {
            "lat": 23.2599, "lon": 77.4126,
            "N": 60, "P": 52, "K": 40, "ph": 7.2,
            "seasonal_rain": {"Kharif (Monsoon)": 210.0, "Rabi (Winter)": 45.0, "Zaid (Summer)": 35.0},
        },
        "Indore": {
            "lat": 22.7196, "lon": 75.8577,
            "N": 55, "P": 48, "K": 42, "ph": 6.8,
            "seasonal_rain": {"Kharif (Monsoon)": 195.0, "Rabi (Winter)": 40.0, "Zaid (Summer)": 30.0},
        },
        "Jabalpur": {
            "lat": 23.1815, "lon": 79.9864,
            "N": 65, "P": 50, "K": 45, "ph": 7.0,
            "seasonal_rain": {"Kharif (Monsoon)": 240.0, "Rabi (Winter)": 55.0, "Zaid (Summer)": 40.0},
        },
        "Gwalior": {
            "lat": 26.2183, "lon": 78.1828,
            "N": 52, "P": 44, "K": 38, "ph": 7.4,
            "seasonal_rain": {"Kharif (Monsoon)": 175.0, "Rabi (Winter)": 42.0, "Zaid (Summer)": 32.0},
        },
        "Ujjain": {
            "lat": 23.1765, "lon": 75.7885,
            "N": 58, "P": 49, "K": 41, "ph": 7.1,
            "seasonal_rain": {"Kharif (Monsoon)": 190.0, "Rabi (Winter)": 38.0, "Zaid (Summer)": 30.0},
        },
    },
    "Uttar Pradesh": {
        "Lucknow": {
            "lat": 26.8467, "lon": 80.9462,
            "N": 62, "P": 48, "K": 44, "ph": 7.1,
            "seasonal_rain": {"Kharif (Monsoon)": 205.0, "Rabi (Winter)": 50.0, "Zaid (Summer)": 42.0},
        },
        "Varanasi": {
            "lat": 25.3176, "lon": 82.9739,
            "N": 70, "P": 45, "K": 50, "ph": 7.5,
            "seasonal_rain": {"Kharif (Monsoon)": 220.0, "Rabi (Winter)": 52.0, "Zaid (Summer)": 45.0},
        },
        "Kanpur": {
            "lat": 26.4499, "lon": 80.3319,
            "N": 64, "P": 47, "K": 43, "ph": 7.3,
            "seasonal_rain": {"Kharif (Monsoon)": 195.0, "Rabi (Winter)": 48.0, "Zaid (Summer)": 38.0},
        },
        "Prayagraj": {
            "lat": 25.4358, "lon": 81.8463,
            "N": 68, "P": 46, "K": 48, "ph": 7.4,
            "seasonal_rain": {"Kharif (Monsoon)": 215.0, "Rabi (Winter)": 50.0, "Zaid (Summer)": 40.0},
        },
        "Gorakhpur": {
            "lat": 26.7606, "lon": 83.3732,
            "N": 74, "P": 52, "K": 49, "ph": 6.9,
            "seasonal_rain": {"Kharif (Monsoon)": 255.0, "Rabi (Winter)": 60.0, "Zaid (Summer)": 55.0},
        },
    },
}

def fetch_live_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&timezone=auto"
    )
    try:
        response = requests.get(url, timeout=6)
        if response.status_code == 200:
            daily = response.json().get("daily", {})
            temps = daily.get("temperature_2m_max", [])
            rains = daily.get("precipitation_sum", [])
            humidities = daily.get("relative_humidity_2m_mean", [])

            avg_temp = sum(temps) / len(temps) if temps else 26.0
            live_monthly_rain = (sum(rains) / len(rains)) * 30 if rains else 80.0
            avg_humidity = sum(humidities) / len(humidities) if humidities else 70.0
            return avg_temp, live_monthly_rain, avg_humidity, True
    except Exception:
        pass
    return 26.0, 80.0, 70.0, False

def generate_crop_reasoning(crop_name, features_dict):
    profile = crop_profiles.get(crop_name)
    if not profile:
        return "जिला मिट्टी प्रोफाइल और मौसम के आधार पर उपयुक्त।"

    matches = []
    checks = [
        ("temperature", "तापमान (Temp)", "°C"),
        ("rainfall", "वर्षा (Rainfall)", "mm"),
        ("humidity", "नमी (Humidity)", "%"),
        ("ph", "मिट्टी pH", ""),
        ("N", "नाइट्रोजन (N)", ""),
    ]
    for key, label, unit in checks:
        val = features_dict[key]
        low, high = profile[key]["min"], profile[key]["max"]
        if low <= val <= high:
            matches.append(f"{label}: `{val:.1f}{unit}` (आदर्श: {low}–{high}{unit})")

    if matches:
        return "✅ **अनुकूल कारक:** " + " | ".join(matches[:3])
    return "ℹ️ वर्तमान मिट्टी और मौसमी संतुलन के लिए वैकल्पिक उपयुक्त फसल।"

# Streamlit UI Configuration
st.set_page_config(page_title="KRISHI Sahyog - AI Crop Predictor", page_icon="🌾", layout="centered")

st.title("कृषि सहयोग (KRISHI Sahyog)")
st.subheader("🌾 AI-आधारित मौसम और बहु-फसल सलाहकार (Multi-Crop Predictor)")

if artifact.get("test_accuracy"):
    st.caption(
        f"🤖 सक्रिय मॉडल: **{artifact['model_name']}** | "
        f"सटीकता (Test Accuracy): **{artifact['test_accuracy']}%** | "
        f"Macro F1: **{artifact['macro_f1']}%**"
    )

# Location & Season Selection
col1, col2 = st.columns(2)
with col1:
    selected_state = st.selectbox("अपना राज्य चुनें (Select State)", list(district_database.keys()))
with col2:
    selected_district = st.selectbox("अपना जिला चुनें (Select District)", list(district_database[selected_state].keys()))

season = st.selectbox(
    "बुवाई का मौसम चुनें (Select Sowing Season)",
    ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"],
)

district_info = district_database[selected_state][selected_district]

# Optional Soil Health Card Overrides
with st.expander("🧪 सॉइल हेल्थ कार्ड मान बदलें (Optional: Custom Soil NPK & pH)"):
    c1, c2, c3, c4 = st.columns(4)
    n_val = c1.number_input("Nitrogen (N)", min_value=0, max_value=200, value=int(district_info["N"]))
    p_val = c2.number_input("Phosphorus (P)", min_value=0, max_value=200, value=int(district_info["P"]))
    k_val = c3.number_input("Potassium (K)", min_value=0, max_value=250, value=int(district_info["K"]))
    ph_val = c4.number_input("Soil pH", min_value=3.5, max_value=10.0, value=float(district_info["ph"]), step=0.1)

if st.button("शीर्ष फसलें और मौसम सलाह देखें (Get AI Recommendations)", type="primary"):
    with st.spinner("लाइव मौसम डेटा और मौसमी औसत का विश्लेषण किया जा रहा है..."):
        live_temp, live_rain_proj, live_humidity, is_live = fetch_live_weather(
            district_info["lat"], district_info["lon"]
        )

    # Blend 35% live forecast projection with 65% seasonal historical rainfall baseline
    seasonal_base_rain = district_info["seasonal_rain"][season]
    effective_rainfall = (0.35 * live_rain_proj) + (0.65 * seasonal_base_rain)

    features_dict = {
        "N": n_val,
        "P": p_val,
        "K": k_val,
        "temperature": live_temp,
        "humidity": live_humidity,
        "ph": ph_val,
        "rainfall": effective_rainfall,
    }
    input_df = pd.DataFrame([features_dict])[feature_cols]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    sorted_indices = np.argsort(probabilities)[::-1]

    # Top 3 seasonal field crops
    top_indices = [idx for idx in sorted_indices if classes[idx] not in PERENNIAL_CROPS][:3]

    # Top perennial orchard option (if any)
    orchard_indices = [idx for idx in sorted_indices if classes[idx] in PERENNIAL_CROPS]

    st.markdown("---")
    st.markdown(f"### 📍 स्थान: **{selected_district}, {selected_state}** ({season})")

    weather_source = "लाइव उपग्रह डेटा" if is_live else "अनुमानित क्षेत्रीय डेटा (ऑफ़लाइन फ़ॉलबैक)"
    st.success(
        f"📡 **मौसम और वर्षा विश्लेषण ({weather_source}):** "
        f"तापमान: `{live_temp:.1f}°C` | नमी: `{live_humidity:.1f}%` | "
        f"प्रभावी मौसमी वर्षा: `{effective_rainfall:.1f} मिमी`"
    )

    st.markdown("### 🌱 आपके लिए शीर्ष 3 सुझाई गई फसलें (Top 3 AI Recommendations):")
    for i, idx in enumerate(top_indices):
        crop_name = classes[idx]
        confidence = probabilities[idx] * 100
        st.markdown(f"**{i+1}. {crop_name.upper()}** — उपयुक्तता स्कोर (Suitability): `{confidence:.1f}%`")
        st.progress(float(probabilities[idx]))
        st.markdown(generate_crop_reasoning(crop_name, features_dict))

    if orchard_indices:
        best_orchard = classes[orchard_indices[0]].upper()
        orchard_score = probabilities[orchard_indices[0]] * 100
        st.caption(f"🌳 **वार्षिक बागवानी विकल्प (Perennial Orchard Option):** {best_orchard} (`{orchard_score:.1f}%`)")

    st.markdown("---")
    if effective_rainfall > 150:
        st.info("💧 **सिंचाई सलाह:** पर्याप्त वर्षा/नमी के संकेत हैं। खेतों में जल निकासी (Drainage) सुनिश्चित करें।")
    else:
        st.warning("💧 **सिंचाई सलाह:** कम वर्षा का अनुमान है। स्प्रिंकलर या ड्रिप सिंचाई पद्धति का उपयोग करें।")
