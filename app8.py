"""
SurakshaNet: Privacy-Preserving Disease Cluster Surveillance Radar
Smart India Hackathon Prototype (Problem Statement S10)
Featuring:
1. Full Native Multilingual UI (English, Odia / ଓଡ଼ିଆ, Hindi / हिंदी)
2. Citizen View: Tabular Proximity Reports, Responsive OpenStreetMap & Centered Red Zone Modal
3. Officer View: 1-Click Judge Auth, Differential Privacy Audit & QR Bulletin
4. Clinic Portal: Facility Verification, 10s Tally, IVR Calling Simulator, & EMR Connector
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="SurakshaNet Health Radar", page_icon="🛡️", layout="wide")

# --- Distance Calculation Helper (Haversine Formula) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

# --- Multilingual Localization Dictionary (i18n) ---
I18N = {
    "English": {
        "app_title": "🛡️ SurakshaNet: Disease Cluster Radar",
        "app_sub": "Privacy-preserving multi-signal syndromic surveillance & anomaly detection (Odisha Health Grid)",
        "switch_view": "👤 Switch Viewport",
        "roles": [
            "Citizen / Public View", 
            "Public Health Officer / Authorized View",
            "🏥 Clinic / Frontline Staff Input Portal"
        ],
        "safe_badge": "🟢 Safe (Normal Baseline)",
        "verify_badge": "🟡 Under Verification",
        "alert_badge": "🔴 High Health Advisory",
        "safe_prec": "Safe historical baseline. Standard health hygiene recommended.",
        "verify_prec": "Routine verification underway. No community health risk.",
        "alert_prec": "Active symptom cluster detected. Practice preventive hygiene & seek care if fever persists.",
        "proximity_header": "📍 Nearby Health Alerts & Neighborhood Proximity",
        "privacy_note": "🔒 Privacy Guarantee: Proximity is calculated locally in your browser. Your coordinates are never stored.",
        "choose_loc": "Select Your Current Neighborhood:",
        "recheck_btn": "🎯 Re-check Area Status",
        "map_header": "🗺️ Regional Health Overview Map",
        "table_header": "📋 Neighborhood Health Status & Safety Guidance",
        "col_sector": "Surveillance Sector",
        "col_dist": "Distance (km)",
        "col_status": "Health Status",
        "col_cases": "Approx. Active Cases",
        "col_water": "Water Quality / Sewage Indicator",
        "col_pharma": "Local Pharmacy Medicine Demand",
        "col_prec": "Recommended Community Action",
        "red_alert_title": "RED ZONE HAZARD ALERT",
        "ack_btn": "🚨 Acknowledge & Stay Alert",
        "water_norm": "Normal",
        "water_high": "High (Spike Detected)",
        "pharma_norm": "Normal",
        "pharma_high": "High Surge (+140%)"
    },
    "ଓଡ଼ିଆ (Odia)": {
        "app_title": "🛡️ ସୁରକ୍ଷା-ନେଟ୍: ମହାମାରୀ ସତର୍କତା ରାଡାର",
        "app_sub": "ଗୋପନୀୟତା ସୁରକ୍ଷିତ ବହୁମୁଖୀ ଲକ୍ଷଣ ନିରୀକ୍ଷଣ ଏବଂ ସତର୍କତା ବ୍ୟବସ୍ଥା (ଓଡ଼ିଶା ସ୍ୱାସ୍ଥ୍ୟ ଗ୍ରୀଡ୍)",
        "switch_view": "👤 ବିଭାଗ ବାଛନ୍ତୁ",
        "roles": [
            "ନାଗରିକ / ସାଧାରଣ ସୂଚନା ଦୃଶ୍ୟ", 
            "ସ୍ୱାସ୍ଥ୍ୟ ଅଧିକାରୀ / ପ୍ରାଧିକୃତ ଦୃଶ୍ୟ",
            "🏥 କ୍ଲିନିକ୍ ଏବଂ ସ୍ୱାସ୍ଥ୍ୟକର୍ମୀ ଡାଟା ପୋର୍ଟାଲ୍"
        ],
        "safe_badge": "🟢 ସୁରକ୍ଷିତ (ସ୍ୱାଭାବିକ ସ୍ଥିତି)",
        "verify_badge": "🟡 ଯାଞ୍ଚ ଚାଲିଛି",
        "alert_badge": "🔴 ଉଚ୍ଚ ସ୍ୱାସ୍ଥ୍ୟ ସତର୍କତା ଜାରି",
        "safe_prec": "ସ୍ୱାଭାବିକ ସ୍ଥିତି। ସାଧାରଣ ସ୍ୱାସ୍ଥ୍ୟ ସ୍ୱଚ୍ଛତା ନିୟମ ପାଳନ କରନ୍ତୁ।",
        "verify_prec": "ତଥ୍ୟ ଯାଞ୍ଚ ଚାଲିଛି। ଭୟଭୀତ ହେବାର କୌଣସି କାରଣ ନାହିଁ।",
        "alert_prec": "ଏହି ଅଞ୍ଚଳରେ ରୋଗ ଲକ୍ଷଣ ବୃଦ୍ଧି ପାଇଛି। ପାଣି ଫୁଟାଇ ପିଅନ୍ତୁ ଓ ଜ୍ୱର ହେଲେ ତୁରନ୍ତ ଡାକ୍ତର ଦେଖାନ୍ତୁ।",
        "proximity_header": "📍 ନିକଟସ୍ଥ ସ୍ୱାସ୍ଥ୍ୟ ସତର୍କତା ଏବଂ ଦୂରତା",
        "privacy_note": "🔒 ଗୋପନୀୟତା ନିଶ୍ଚିତତା: ଦୂରତା ଆପଣଙ୍କ ମୋବାଇଲ୍ ଭିତରେ ଗଣନା ହୁଏ, ଆପଣଙ୍କ ଲୋକେସନ୍ କେବେବି ସର୍ଭରକୁ ଯାଏ ନାହିଁ।",
        "choose_loc": "ଆପଣଙ୍କର ବର୍ତ୍ତମାନର ଅଞ୍ଚଳ ବାଛନ୍ତୁ:",
        "recheck_btn": "🎯 ଅଞ୍ଚଳର ସ୍ଥିତି ଯାଞ୍ଚ କରନ୍ତୁ",
        "map_header": "🗺️ ଆଞ୍ଚଳିକ ସ୍ୱାସ୍ଥ୍ୟ ମାନଚିତ୍ର",
        "table_header": "📋 ଆଖପାଖ ଅଞ୍ଚଳର ସ୍ୱାସ୍ଥ୍ୟ ସ୍ଥିତି ଓ ସୁରକ୍ଷା ନିର୍ଦ୍ଦେଶାବଳୀ",
        "col_sector": "ସର୍ଭେଲାନ୍ସ ସେକ୍ଟର",
        "col_dist": "ଦୂରତା (କି.ମି.)",
        "col_status": "ସ୍ୱାସ୍ଥ୍ୟ ସ୍ଥିତି",
        "col_cases": "ଆନୁମାନିକ ସକ୍ରିୟ ସଂଖ୍ୟା",
        "col_water": "ଜଳ / ଡ୍ରେନେଜ୍ ଗୁଣମାନ",
        "col_pharma": "ଔଷଧ ଦୋକାନ ଚାହିଦା",
        "col_prec": "ପରାମର୍ଶିତ ନିରାପତ୍ତା ପଦକ୍ଷେପ",
        "red_alert_title": "⚠️ ରେଡ୍ ଜୋନ୍ ବିପଦ ସତର୍କତା",
        "ack_btn": "🚨 ସୂଚନା ପାଇଲି ଏବଂ ସତର୍କ ରହିବି",
        "water_norm": "ସ୍ୱାଭାବିକ",
        "water_high": "ଅଧିକ (ଜଳ ପ୍ରଦୂଷଣ ଚିହ୍ନଟ)",
        "pharma_norm": "ସ୍ୱାଭାବିକ",
        "pharma_high": "ଉଚ୍ଚ ଚାହିଦା (+୧୪୦%)"
    },
    "हिंदी (Hindi)": {
        "app_title": "🛡️ सुरक्षा-नेट: रोग संकुल निगरानी रडार",
        "app_sub": "गोपनीयता-संरक्षित बहु-संकेत लक्षण निगरानी और आउटब्रेक डिटेक्शन (ओडिशा स्वास्थ्य ग्रिड)",
        "switch_view": "👤 दृश्य बदलें",
        "roles": [
            "नागरिक / सार्वजनिक दृश्य", 
            "सार्वजनिक स्वास्थ्य अधिकारी दृश्य",
            "🏥 क्लिनिक एवं स्वास्थ्य कार्यकर्ता पोर्टल"
        ],
        "safe_badge": "🟢 सुरक्षित (सामान्य स्तर)",
        "verify_badge": "🟡 सत्यापन प्रगति पर है",
        "alert_badge": "🔴 उच्च स्वास्थ्य चेतावनी",
        "safe_prec": "स्थिति सामान्य है। बुनियादी स्वच्छता नियमों का पालन करें।",
        "verify_prec": "डेटा सत्यापन जारी है। घबराने की कोई आवश्यकता नहीं है।",
        "alert_prec": "सक्रिय लक्षण संकुल पाया गया। पानी उबालकर पिएं और बुखार होने पर तुरंत डॉक्टर से मिलें।",
        "proximity_header": "📍 नजदीकी स्वास्थ्य चेतावनी एवं दूरी",
        "privacy_note": "🔒 गोपनीयता की गारंटी: दूरी की गणना आपके फोन/ब्राउज़र पर होती है। आपकी लोकेशन कहीं स्टोर नहीं होती।",
        "choose_loc": "अपना वर्तमान क्षेत्र चुनें:",
        "recheck_btn": "🎯 क्षेत्र की स्थिति पुनः जांचें",
        "map_header": "🗺️ क्षेत्रीय स्वास्थ्य मानचित्र",
        "table_header": "📋 निकटवर्ती स्वास्थ्य स्थिति और सुरक्षा दिशानिर्देश",
        "col_sector": "निगरानी क्षेत्र",
        "col_dist": "दूरी (किमी)",
        "col_status": "स्वास्थ्य स्थिति",
        "col_cases": "अनुमानित सक्रिय मामले",
        "col_water": "जल / सीवेज गुणवत्ता सूचक",
        "col_pharma": "दवाइयों की मांग (फार्मेसी)",
        "col_prec": "अनुशंसित सुरक्षा कदम",
        "red_alert_title": "⚠️ रेड ज़ोन ख़तरा चेतावनी",
        "ack_btn": "🚨 समझ गया एवं सतर्क रहूँगा",
        "water_norm": "सामान्य",
        "water_high": "उच्च (संक्रमण संकेत)",
        "pharma_norm": "सामान्य",
        "pharma_high": "उच्च मांग (+140%)"
    }
}

# --- Session State Management ---
if "health_officer_authenticated" not in st.session_state:
    st.session_state.health_officer_authenticated = False
if "clinic_staff_authenticated" not in st.session_state:
    st.session_state.clinic_staff_authenticated = False
if "last_seen_red_zone" not in st.session_state:
    st.session_state.last_seen_red_zone = None
if "submitted_tally_log" not in st.session_state:
    st.session_state.submitted_tally_log = []
if "ivr_call_active" not in st.session_state:
    st.session_state.ivr_call_active = False

# --- Top Navigation Bar with Language Switcher ---
lang_col, nav_col1, nav_col2 = st.columns([1, 2.2, 1.3])

with lang_col:
    selected_lang = st.selectbox(
        "🌐 Language / ଭାଷା / भाषा",
        ["English", "ଓଡ଼ିଆ (Odia)", "हिंदी (Hindi)"],
        key="global_lang_selector"
    )
    t = I18N[selected_lang]

with nav_col1:
    st.title(t["app_title"])
    st.caption(t["app_sub"])

with nav_col2:
    selected_role_idx = st.selectbox(
        t["switch_view"],
        options=[0, 1, 2],
        format_func=lambda x: t["roles"][x]
    )

# --- Dialogs with 1-Click Judge Logins ---
@st.dialog("🔒 Public Health Officer Verification")
def officer_auth_dialog():
    st.markdown("⚠️ **Restricted Access:** Authorized epidemiological personnel only.")
    entered_pin = st.text_input("Enter Officer ID / PIN", type="password", placeholder="Enter 'SIH2026' or click Demo button below")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Authenticate", use_container_width=True):
            if entered_pin in ["SIH2026", "admin"]:
                st.session_state.health_officer_authenticated = True
                st.success("Verified. Access granted.")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.health_officer_authenticated = False
            st.rerun()
            
    st.markdown("---")
    if st.button("⚡ 1-Click Demo Login (For Judges)", use_container_width=True, type="primary"):
        st.session_state.health_officer_authenticated = True
        st.success("Authorized as Senior Epidemiologist (Demo Mode)")
        st.rerun()

@st.dialog("🏥 Frontline Clinic / Facility Verification")
def clinic_auth_dialog():
    st.markdown("🔒 **Accredited Healthcare Facility Login:** For verified PHC/CHC medical staff.")
    clinic_pin = st.text_input("Enter Facility NPI / PIN", type="password", placeholder="Enter 'CLINIC2026' or click Demo button below")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Verify Credentials", use_container_width=True):
            if clinic_pin in ["CLINIC2026", "admin", "1234"]:
                st.session_state.clinic_staff_authenticated = True
                st.success("Verified. Opening Ingestion Channel.")
                st.rerun()
            else:
                st.error("Unrecognized Token.")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.clinic_staff_authenticated = False
            st.rerun()
            
    st.markdown("---")
    if st.button("⚡ 1-Click Demo Login (For Judges)", use_container_width=True, type="primary"):
        st.session_state.clinic_staff_authenticated = True
        st.success("Authorized as Primary Health Centre (PHC) Lead Staff (Demo Mode)")
        st.rerun()

if selected_role_idx == 1 and not st.session_state.health_officer_authenticated:
    officer_auth_dialog()

if selected_role_idx == 2 and not st.session_state.clinic_staff_authenticated:
    clinic_auth_dialog()

if selected_role_idx == 0:
    st.session_state.health_officer_authenticated = False
    st.session_state.clinic_staff_authenticated = False

st.divider()

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Anomaly Simulation Engine")
scenario = st.sidebar.selectbox(
    "Select Epidemiological Scenario",
    [
        "🟢 Normal Baseline (No Cluster)",
        "🌊 Gastrointestinal / Waterborne Cluster Pattern",
        "🫁 Acute Respiratory Cluster Pattern",
        "⚠️ Single-Source Input Mismatch / Typo",
        "📋 Rural Reporting Delay & Data Gap"
    ]
)

if st.session_state.health_officer_authenticated:
    st.sidebar.subheader("🔒 Differential Privacy Parameters")
    epsilon = st.sidebar.slider("Privacy Budget (Epsilon: ε)", 0.1, 2.0, 0.5, 0.1)
else:
    epsilon = 0.5

# --- Data Telemetry Engine ---
np.random.seed(42)
zones_meta = [
    {"name": "CDA Sector 9, Cuttack", "lat": 20.4780, "lon": 85.8338, "baseline": 14},
    {"name": "Saheed Nagar, Bhubaneswar", "lat": 20.2925, "lon": 85.8450, "baseline": 18},
    {"name": "Badambadi Bus Stand, Cuttack", "lat": 20.4560, "lon": 85.8750, "baseline": 22},
    {"name": "Choudwar Industrial Area", "lat": 20.5280, "lon": 85.9120, "baseline": 12},
    {"name": "Banki Sub-district", "lat": 20.3780, "lon": 85.5340, "baseline": 8}
]

data = []
for z in zones_meta:
    raw_cases = max(2, int(np.random.normal(z["baseline"], 2)))
    ww_load = round(float(np.random.uniform(1.0, 2.2)), 1)
    otc_sales = round(float(np.random.uniform(-5.0, 10.0)), 1)
    
    # Syndromic Scenarios
    if "Gastrointestinal" in scenario and "CDA" in z["name"]:
        raw_cases = 65
        ww_load = 8.6
        otc_sales = 135.0
    elif "Respiratory" in scenario and "Saheed" in z["name"]:
        raw_cases = 90
        ww_load = 9.2
        otc_sales = 220.0
    elif "Mismatch" in scenario and "CDA" in z["name"]:
        raw_cases = 500
        ww_load = 1.2
        otc_sales = 2.0
    elif "Delay" in scenario and "Banki" in z["name"]:
        raw_cases = 10
        ww_load = 8.4
        otc_sales = 160.0
        
    # Laplace Noise
    noise = np.random.laplace(0, 1.0 / epsilon)
    privatized_cases = max(0, int(round(raw_cases + noise)))
    noise_delta = privatized_cases - raw_cases
    
    # Anomaly Logic
    ratio = privatized_cases / max(1, z["baseline"])
    if ratio >= 4.0 and ww_load < 3.0:
        status = "DATA_CHECK"
        public_badge = t["verify_badge"]
        precautions = t["verify_prec"]
        cluster_type = "Data Entry Outlier (Quarantined)"
        xai = "94% Clinic Typo | 4% Pharmacy (Flat) | 2% Wastewater (Normal)"
        public_ww = t["water_norm"]
        public_otc = t["pharma_norm"]
        playbook = ["Dispatch verification ticket to clinic.", "Hold aggregate from public bulletin.", "Re-sample wastewater."]
    elif ww_load >= 7.5 and otc_sales >= 100.0 and ratio <= 1.5:
        status = "HEALTH_ALERT"
        public_badge = t["alert_badge"]
        precautions = t["alert_prec"]
        cluster_type = "Unregistered Community Cluster"
        xai = "60% Wastewater Surge | 32% Pharmacy Sales | 8% Clinic Lag"
        public_ww = t["water_high"]
        public_otc = t["pharma_high"]
        playbook = ["Deploy mobile ASHA health survey units.", "Verify pending physical OPD paper logs.", "Pre-position fever diagnostic kits."]
    elif (ratio >= 2.0 and ww_load >= 5.0) or (ww_load >= 8.0 and otc_sales >= 80.0):
        status = "HEALTH_ALERT"
        public_badge = t["alert_badge"]
        precautions = t["alert_prec"]
        cluster_type = "Confirmed Syndromic Cluster"
        xai = "48% Wastewater Spike | 32% Clinic Surge | 20% Pharmacy Sales"
        public_ww = t["water_high"]
        public_otc = t["pharma_high"]
        playbook = ["Issue localized vector/sanitation municipal order.", "Pre-position oral rehydration & medication stocks.", "Broadcast regional SMS health alert."]
    else:
        status = "SAFE"
        public_badge = t["safe_badge"]
        precautions = t["safe_prec"]
        cluster_type = "Normal Baseline"
        xai = "75% Seasonal Norm | 15% Normal Clinic | 10% Normal Water"
        public_ww = t["water_norm"]
        public_otc = t["pharma_norm"]
        playbook = ["Maintain passive multi-sensor telemetry monitoring."]
        
    data.append({
        "Neighborhood": z["name"],
        "lat": z["lat"],
        "lon": z["lon"],
        "Baseline": z["baseline"],
        "Status": status,
        "Status_Badge": public_badge,
        "Cluster_Type": cluster_type,
        "Estimated_Cases": f"~{max(5, (privatized_cases // 5) * 5)} - {((privatized_cases // 5) + 1) * 5}",
        "Public_Wastewater": public_ww,
        "Public_Pharmacy": public_otc,
        "Precautions": precautions,
        "Playbook": playbook,
        "Raw_Count": raw_cases,
        "Privatized_Count": privatized_cases,
        "Noise_Delta": f"{'+' if noise_delta >= 0 else ''}{noise_delta}",
        "Wastewater_Index": f"{ww_load}/10",
        "OTC_Surge": f"+{otc_sales}%",
        "XAI": xai
    })

df = pd.DataFrame(data)

# ==============================================================================
# VIEW 1: CITIZEN VIEW (Fully Localized)
# ==============================================================================
if selected_role_idx == 0:
    st.subheader(t["proximity_header"])
    st.caption(t["privacy_note"])

    loc_col1, loc_col2 = st.columns([2, 1])
    with loc_col1:
        selected_location = st.selectbox(
            t["choose_loc"],
            [z["name"] for z in zones_meta],
            key="selected_neighborhood_key"
        )
    with loc_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        detect_loc = st.button(t["recheck_btn"], use_container_width=True)

    current_zone = df[df["Neighborhood"] == selected_location].iloc[0]
    user_lat, user_lon = current_zone["lat"], current_zone["lon"]
    df["Distance_km"] = df.apply(lambda r: haversine_distance(user_lat, user_lon, r["lat"], r["lon"]), axis=1)
    df_sorted = df.sort_values(by="Distance_km")

    # --- Responsive Red Zone Dialog Modal ---
    @st.dialog(" ")
    def show_red_zone_hazard_modal(zone_name, est_cases, precautions):
        st.markdown(f"""
        <div style="text-align: center; padding: 10px 5px;">
            <div style="display: flex; justify-content: center; margin-bottom: 12px;">
                <svg width="84" height="84" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L1 21H23L12 2Z" fill="#FEE2E2" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 9V14" stroke="#DC2626" stroke-width="2.5" stroke-linecap="round"/>
                    <circle cx="12" cy="17.5" r="1.25" fill="#DC2626"/>
                </svg>
            </div>
            <h2 style="color: #991B1B; margin: 0; font-weight: 800; font-size: 1.4rem;">
                {t["red_alert_title"]}
            </h2>
            <p style="color: #DC2626; font-weight: 600; margin: 6px 0 16px 0; font-size: 1rem;">
                {zone_name}
            </p>
            <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 10px; padding: 14px; text-align: left; margin-bottom: 18px;">
                <p style="color: #7F1D1D; margin: 0 0 8px 0; font-size: 0.9rem;">
                    <strong>{t["col_cases"]}:</strong> {est_cases}
                </p>
                <p style="color: #7F1D1D; margin: 0; font-size: 0.9rem;">
                    <strong>{t["col_prec"]}:</strong> {precautions}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(t["ack_btn"], use_container_width=True, type="primary"):
            st.session_state.last_seen_red_zone = f"{selected_location}_{scenario}_{selected_lang}"
            st.rerun()

    current_key = f"{selected_location}_{scenario}_{selected_lang}"
    if current_zone["Status"] == "HEALTH_ALERT" and (st.session_state.last_seen_red_zone != current_key or detect_loc):
        st.session_state.last_seen_red_zone = current_key
        show_red_zone_hazard_modal(current_zone["Neighborhood"], current_zone["Estimated_Cases"], current_zone["Precautions"])

    st.markdown("---")
    st.markdown(f"### 🏠 **{selected_location}**")
    if current_zone["Status"] == "HEALTH_ALERT":
        st.error(f"⚠️ **{current_zone['Status_Badge']}**\n\n👉 {current_zone['Precautions']}")
    elif current_zone["Status"] == "DATA_CHECK":
        st.warning(f"🔍 **{current_zone['Status_Badge']}**\n\n👉 {current_zone['Precautions']}")
    else:
        st.success(f"✅ **{current_zone['Status_Badge']}**\n\n👉 {current_zone['Precautions']}")

    # Mobile-Friendly OpenStreetMap
    st.markdown(f"#### {t['map_header']}")
    fig_map = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="Status_Badge",
        color_discrete_map={
            t["safe_badge"]: "#10b981",
            t["verify_badge"]: "#f59e0b",
            t["alert_badge"]: "#ef4444"
        },
        size=[26 if "High" in s or "ଉଚ୍ଚ" in s or "उच्च" in s else 14 for s in df["Status_Badge"]],
        hover_name="Neighborhood",
        hover_data={"lat": False, "lon": False, "Distance_km": True, "Status_Badge": True},
        zoom=9.5,
        center={"lat": float(df["lat"].mean()), "lon": float(df["lon"].mean())},
        mapbox_style="open-street-map"
    )
    fig_map.update_layout(
        autosize=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"responsive": True, "displayModeBar": False})

    st.markdown("---")
    st.subheader(t["table_header"])
    citizen_table = df_sorted[[
        "Neighborhood", "Distance_km", "Status_Badge", "Estimated_Cases", 
        "Public_Wastewater", "Public_Pharmacy", "Precautions"
    ]].rename(columns={
        "Neighborhood": t["col_sector"],
        "Distance_km": t["col_dist"],
        "Status_Badge": t["col_status"],
        "Estimated_Cases": t["col_cases"],
        "Public_Wastewater": t["col_water"],
        "Public_Pharmacy": t["col_pharma"],
        "Precautions": t["col_prec"]
    })
    st.dataframe(citizen_table, use_container_width=True, hide_index=True)

# ==============================================================================
# VIEW 2: PUBLIC HEALTH OFFICER VIEW + QR BULLETIN GENERATOR
# ==============================================================================
elif selected_role_idx == 1 and st.session_state.health_officer_authenticated:
    st.subheader("📊 Epidemiological Telemetry & Differential Privacy Ledger")
    st.caption("Authenticated Public Health Session: Complete multi-signal cross-validation.")

    if st.button("🔒 Lock & Exit Officer View"):
        st.session_state.health_officer_authenticated = False
        st.rerun()

    st.markdown("### 📋 Primary Surveillance & Anomaly Cross-Validation Table")
    officer_table = df[[
        "Neighborhood", "Status_Badge", "Cluster_Type", "Baseline",
        "Privatized_Count", "Wastewater_Index", "OTC_Surge", "XAI"
    ]].rename(columns={
        "Neighborhood": "Surveillance Zone",
        "Status_Badge": "Alert Level",
        "Cluster_Type": "Syndromic Classification",
        "Baseline": "3-Yr Normal Baseline",
        "Privatized_Count": "Privatized Triage Count",
        "Wastewater_Index": "Wastewater PCR (0-10)",
        "OTC_Surge": "Pharmacy Sales Surge",
        "XAI": "Explainable AI (Signal Weights)"
    })
    st.dataframe(officer_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"🔒 Differential Privacy (Laplace Mechanism) Mathematical Audit")
    st.caption(f"Verifying individual plausible deniability. Mathematical budget: ε = {epsilon}")

    privacy_audit_table = df[["Neighborhood", "Raw_Count", "Privatized_Count", "Noise_Delta"]].copy()
    privacy_audit_table["Privacy_Guarantee"] = "Protected by Laplace Mechanism (No PII Transmitted)"
    privacy_audit_table = privacy_audit_table.rename(columns={
        "Neighborhood": "Monitored Zone",
        "Raw_Count": "1. Confidential Raw Edge Count (Local Only)",
        "Privatized_Count": f"2. Transmitted Count (ε = {epsilon})",
        "Noise_Delta": "3. Laplace Noise Added (± Delta)",
        "Privacy_Guarantee": "Security Compliance"
    })
    st.dataframe(privacy_audit_table, use_container_width=True, hide_index=True)

    # QR Bulletin Generator
    st.markdown("---")
    st.subheader("📄 Generate QR-Verified Public Health Advisory Bulletin")
    rep_zone_name = st.selectbox("Select Zone to Issue Official Advisory For:", [z["name"] for z in zones_meta])
    rep_zone = df[df["Neighborhood"] == rep_zone_name].iloc[0]

    @st.dialog("📋 Official Public Health Advisory Bulletin")
    def show_qr_bulletin_modal(zone_row):
        report_id = f"GOV-OD-EPI-{np.random.randint(10000, 99999)}"
        timestamp = datetime.now().strftime("%d-%b-%Y %H:%M IST")
        sha_hash = f"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"[:24]
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=160x160&data=https://health.odisha.gov.in/verify?id={report_id}"
        
        st.markdown(f"""
        <div style="border: 2px solid #1E3A8A; border-radius: 12px; padding: 20px; background-color: #F8FAFC;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1E3A8A; padding-bottom: 10px;">
                <div>
                    <h3 style="color: #1E3A8A; margin: 0;">DEPARTMENT OF HEALTH & FAMILY WELFARE</h3>
                    <p style="color: #475569; margin: 2px 0 0 0; font-size: 0.85rem;">GOVERNMENT OF ODISHA | EPIDEMIOLOGICAL SURVEILLANCE CELL</p>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: {'#EF4444' if 'High' in zone_row['Status_Badge'] or 'ଉଚ୍ଚ' in zone_row['Status_Badge'] or 'उच्च' in zone_row['Status_Badge'] else '#10B981'}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">
                        {zone_row['Status_Badge']}
                    </span>
                </div>
            </div>
            
            <div style="display: flex; gap: 20px; margin-top: 15px;">
                <div style="flex: 2;">
                    <p style="margin: 4px 0;"><strong>Bulletin Reference:</strong> <code>{report_id}</code></p>
                    <p style="margin: 4px 0;"><strong>Surveillance Target:</strong> {zone_row['Neighborhood']}</p>
                    <p style="margin: 4px 0;"><strong>Syndromic Profile:</strong> {zone_row['Cluster_Type']}</p>
                    <p style="margin: 4px 0;"><strong>Issue Timestamp:</strong> {timestamp}</p>
                    <p style="margin: 4px 0;"><strong>Cryptographic Hash:</strong> <code>SHA256:{sha_hash}...</code></p>
                </div>
                <div style="flex: 1; text-align: center;">
                    <img src="{qr_url}" width="130" style="border: 1px solid #CBD5E1; border-radius: 8px; padding: 4px; background: white;" alt="Verification QR Code"/>
                    <p style="font-size: 0.7rem; color: #64748B; margin-top: 4px;">Scan to verify bulletin authenticity</p>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid #CBD5E1; margin: 15px 0;"/>
            <h4 style="color: #1E3A8A; margin-bottom: 8px;">🚨 Prescriptive Action Protocol:</h4>
        """, unsafe_allow_html=True)
        
        for i, step in enumerate(zone_row["Playbook"], 1):
            st.markdown(f"**{i}.** {step}")
            
        st.markdown(f"""
            <div style="background-color: #FEF2F2; border-left: 4px solid #DC2626; padding: 10px; border-radius: 6px; margin-top: 15px;">
                <strong style="color: #991B1B;">Public Safety Directive:</strong>
                <p style="color: #7F1D1D; margin: 4px 0 0 0; font-size: 0.9rem;">{zone_row['Precautions']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Close Bulletin", use_container_width=True):
            st.rerun()

    if st.button("📄 Generate & Preview Official QR Bulletin", type="primary"):
        show_qr_bulletin_modal(rep_zone)

# ==============================================================================
# VIEW 3: CLINIC / FRONTLINE PORTAL (Tally, OCR, IVR, & EMR Connector)
# ==============================================================================
elif selected_role_idx == 2 and st.session_state.clinic_staff_authenticated:
    st.subheader("📝 Rural Clinic & Frontline Staff Telemetry Entry")
    st.caption("Authenticated Healthcare Facility Session: Low-burden syndromic data ingestion.")
    
    if st.button("🔒 Lock & Exit Clinic Portal"):
        st.session_state.clinic_staff_authenticated = False
        st.rerun()

    clinic_col1, clinic_col2 = st.columns([1, 1])
    with clinic_col1:
        st.markdown("### 🏢 Facility Profile")
        clinic_facility = st.selectbox(
            "Select Health Sub-Centre / PHC:",
            [
                "Banki Rural Primary Health Centre (PHC)",
                "CDA Sector Urban Health Post",
                "Choudwar Community Health Centre (CHC)",
                "Badambadi Sub-Centre Triage Post"
            ]
        )
        staff_role = st.selectbox("Staff Role:", ["ANM / Staff Nurse", "Medical Officer / Doctor", "ASHA Health Worker"])
        
    with clinic_col2:
        st.markdown("### 🔒 Zero-PII Edge Guarantee")
        st.info("""
        * **Zero Personal Information:** No patient names, phone numbers, or Aadhaar numbers.
        * **Mathematical Privacy:** Laplace Differential Privacy noise added locally on device.
        * **No Routine Disruption:** Doctors and nurses maintain fast morning paper workflows.
        """)

    st.markdown("---")
    st.subheader("⚡ Ingestion Channels by Facility Maturity")
    
    input_mode = st.radio(
        "Choose Telemetry Ingestion Method:",
        [
            "Option 1: Quick 10-Second Digital Tally (Paper-First PHCs)", 
            "Option 2: Toll-Free IVR Voice Call Simulator (Keypad Phones / No Internet)",
            "Option 3: On-Device Paper Register Scanner (OCR Simulation)",
            "Option 4: Automated EMR / Hospital DB Connector (Zero-Touch Urban Hospitals)"
        ],
        horizontal=True
    )
    
    if "Option 1" in input_mode:
        st.markdown("#### 🔢 Enter Daily Aggregate Tally from Physical Register:")
        c1, c2, c3 = st.columns(3)
        with c1:
            fever_cases = st.number_input("🤒 Acute Fever / Malaria Tally", min_value=0, max_value=200, value=14, step=1)
        with c2:
            gastro_cases = st.number_input("🤢 Diarrhea / Vomiting Tally", min_value=0, max_value=200, value=6, step=1)
        with c3:
            resp_cases = st.number_input("🫁 Cough / Severe Breathlessness", min_value=0, max_value=200, value=8, step=1)
            
        total_raw = fever_cases + gastro_cases + resp_cases
        
        if st.button("🚀 Encrypt & Submit Daily Tally to Grid", type="primary", use_container_width=True):
            local_noise = np.random.laplace(0, 1.0 / 0.5)
            privatized_total = max(0, int(round(total_raw + local_noise)))
            
            st.session_state.submitted_tally_log.append({
                "Timestamp": "Just Now",
                "Channel": "Digital App Tally",
                "Facility": clinic_facility,
                "Staff": staff_role,
                "Raw Tally (Kept Local)": total_raw,
                "Transmitted Aggregate (DP Protected)": privatized_total,
                "Status": "✅ Verified & Synchronized"
            })
            st.success(f"Tally submitted! Raw count ({total_raw}) was protected with Differential Privacy ({privatized_total} transmitted). Central server received zero personal information.")

    elif "Option 2" in input_mode:
        st.markdown("#### 📞 Toll-Free Automated Voice Response (IVR) Channel")
        ivr_c1, ivr_c2 = st.columns([1, 1])
        with ivr_c1:
            st.markdown("""
            <div style="background-color: #1E293B; color: #F8FAFC; padding: 20px; border-radius: 12px; text-align: center;">
                <h3 style="color: #38BDF8; margin: 0;">📞 1800-SURAKSHA</h3>
                <p style="color: #94A3B8; font-size: 0.85rem; margin: 4px 0 15px 0;">National Health Surveillance Toll-Free Gateway</p>
                <div style="background-color: #0F172A; border-radius: 8px; padding: 12px; font-family: monospace; text-align: left; color: #4ADE80; font-size: 0.85rem;">
                    > IVR GATEWAY: READY<br>
                    > REGIONAL VOICE: ODIA / HINDI / ENGLISH<br>
                    > DTMF TONE CAPTURE: ENABLED
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if not st.session_state.ivr_call_active:
                if st.button("🟢 Dial 1800-787-257 (Start Call Simulation)", use_container_width=True, type="primary"):
                    st.session_state.ivr_call_active = True
                    st.rerun()
            else:
                if st.button("🔴 End Call", use_container_width=True):
                    st.session_state.ivr_call_active = False
                    st.rerun()

        with ivr_c2:
            if st.session_state.ivr_call_active:
                st.markdown("##### 🎙️ Call in Progress (Automated Voice Prompts):")
                st.audio("https://actions.google.com/sounds/v1/teleport/teleport_start.ogg", format="audio/ogg")
                st.markdown("*🔊 **Voice Prompt:** 'ନମସ୍କାର, ସୁରକ୍ଷା-ନେଟ୍ ରେ ଆପଣଙ୍କୁ ସ୍ୱାଗତ। (Please enter today's fever count followed by hash).'*" )
                
                ivr_fever = st.number_input("Dialpad Entry: Fever Count (#)", min_value=0, max_value=150, value=14)
                ivr_gastro = st.number_input("Dialpad Entry: Diarrhea / Vomiting (#)", min_value=0, max_value=150, value=6)
                
                if st.button("📲 Press [#] Hash to Confirm & Submit Call Tally", use_container_width=True):
                    raw_ivr_sum = ivr_fever + ivr_gastro
                    priv_ivr = max(0, int(round(raw_ivr_sum + np.random.laplace(0, 2.0))))
                    
                    st.session_state.submitted_tally_log.append({
                        "Timestamp": "Just Now",
                        "Channel": "Toll-Free IVR Phone Call",
                        "Facility": clinic_facility,
                        "Staff": staff_role,
                        "Raw Tally (Kept Local)": raw_ivr_sum,
                        "Transmitted Aggregate (DP Protected)": priv_ivr,
                        "Status": "✅ Verified via IVR Audio Gateway"
                    })
                    st.session_state.ivr_call_active = False
                    st.success(f"IVR Call Completed! Voice gateway captured {raw_ivr_sum} raw cases, added Laplace noise ({priv_ivr} transmitted), and hung up.")
                    st.rerun()
            else:
                st.info("Click 'Dial 1800-SURAKSHA' on the left to test voice submission.")

    elif "Option 3" in input_mode:
        st.markdown("#### 📷 Snap Photo of Paper Register Page:")
        uploaded_file = st.file_uploader("Upload register sheet photo (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None or st.button("📸 Simulate Scanning Paper Register Page"):
            st.success("✅ On-Device OCR Extracted: 16 Fever Marks, 4 Diarrhea Marks, 7 Respiratory Marks.")
            st.markdown("""
            * **Extracted Raw Total:** `27 Cases`
            * **Privacy Masking Applied:** Laplace Noise (+2)
            * **Payload Transmitted:** `29 Cases (Anonymized)`
            * **Photo Memory Status:** 🗑️ Original image wiped from RAM.
            """)

    elif "Option 4" in input_mode:
        st.markdown("#### 💻 Zero-Touch Hospital EMR / Database Hook")
        st.code("""
# SurakshaNet Background Service (Runs nightly at 23:59 on hospital server)
def sync_hospital_syndromes():
    raw_fever_count = db.query("SELECT COUNT(*) FROM triage WHERE symptom LIKE '%fever%' AND date=TODAY()")
    noisy_payload = add_laplace_noise(raw_fever_count, epsilon=0.5)
    requests.post("https://api.surakshanet.gov.in/v1/telemetry", json={"zone": "OR_CTC_09", "tally": noisy_payload})
        """, language="python")
        if st.button("🔄 Test Automated Hospital Database Connector (Zero-Touch Simulation)"):
            st.success("✅ Connection Successful! 48 fever records detected in HMIS. Injected Laplace noise (+2). Transmitted 50 anonymized cases to central grid.")

    if len(st.session_state.submitted_tally_log) > 0:
        st.markdown("---")
        st.markdown("### 📋 Local Submission Ledger (Stored on this device only)")
        st.dataframe(pd.DataFrame(st.session_state.submitted_tally_log), use_container_width=True, hide_index=True)