
import streamlit as st
import pandas as pd
import joblib
import re
import hashlib
from pypdf import PdfReader

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# Load saved model
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Disease Prediction by Owais ❤️")

# ==================================
# 1. PDF UPLOAD AND DATA EXTRACTION
# ==================================

st.header("📄 Upload Patient Medical Report")

uploaded_file = st.file_uploader(
    "Upload patient report",
    type=["pdf"]
)

if uploaded_file is not None:

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("applied_pdf_hash") != file_hash:

        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        clean_text = re.sub(r"\s+", " ", text)

        patterns = {
            "age": r"Age\s+(\d+)\s+years",
            "resting_bp": r"Resting blood pressure \(RestingBP\)\s+(\d+)",
            "cholesterol": r"Total cholesterol\s+(\d+)",
            "fasting_bs": r"Fasting blood sugar \(FastingBS\)\s+([01])",
            "max_hr": r"Maximum heart rate \(MaxHR\)\s+(\d+)",
            "oldpeak": r"Oldpeak\s+(\d+(?:\.\d+)?)",
            "sex": r"Sex\s+(M|F)",
            "chest_pain": r"Chest pain type\s+(ATA|NAP|TA|ASY)",
            "resting_ecg": r"Resting ECG\s+(Normal|ST|LVH)",
            "exercise_angina": r"Exercise-induced angina\s+(Y|N)",
            "st_slope": r"ST slope\s+(Up|Flat|Down)"
        }

        extracted = {}

        for key, pattern in patterns.items():

            match = re.search(pattern, clean_text, re.IGNORECASE)

            if match:
                extracted[key] = match.group(1)

        # Convert numerical values
        numeric_fields = [
            "age", "resting_bp", "cholesterol",
            "fasting_bs", "max_hr"
        ]

        for key in numeric_fields:
            if key in extracted:
                extracted[key] = int(extracted[key])

        if "oldpeak" in extracted:
            extracted["oldpeak"] = float(extracted["oldpeak"])

        # Update Streamlit session state
        # These keys MUST match the widget keys below

        for key, value in extracted.items():
            st.session_state[key] = value

        st.session_state["applied_pdf_hash"] = file_hash
        st.session_state["pdf_extracted_data"] = extracted

        st.success("PDF details extracted successfully!")

        # Refresh application
        st.rerun()

    st.success("Patient report loaded!")

    st.subheader("Extracted Patient Details")
    st.json(st.session_state.get("pdf_extracted_data", {}))


# ==================================
# 2. PATIENT INPUT FORM
# ==================================

st.header("👨‍⚕️ Patient Information")

st.markdown("Review the extracted details and correct them if necessary.")

age = st.slider(
    "Age", 18, 100,
    value=40,
    key="age"
)

sex = st.selectbox(
    "Sex", ["M", "F"],
    key="sex"
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"],
    key="chest_pain"
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80, 200,
    value=120,
    key="resting_bp"
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    100, 600,
    value=200,
    key="cholesterol"
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1],
    key="fasting_bs"
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"],
    key="resting_ecg"
)

max_hr = st.slider(
    "Max Heart Rate",
    60, 220,
    value=150,
    key="max_hr"
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"],
    key="exercise_angina"
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0, 6.0,
    value=1.0,
    step=0.1,
    key="oldpeak"
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"],
    key="st_slope"
)


# ==================================
# 3. KNN PREDICTION
# ==================================

if st.button("Predict"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ High Risk of Heart Disease")

    else:

        st.success("✅ Low Risk of Heart Disease")

    st.caption(
        "This is an experimental ML prediction, "
        "not a medical diagnosis. Consult a healthcare professional."
    )
