import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        rgba(20, 30, 48, 0.04),
        rgba(36, 59, 85, 0.08)
    );
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
    color: #e53935;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 30px;
    color: #666666;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    padding: 12px 15px;
    margin-top: 20px;
    margin-bottom: 20px;
    color: #ffffff !important;
    background: linear-gradient(90deg, #1d4ed8, #3b82f6);
    border-radius: 10px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
    font-weight: 600 !important;
    font-size: 16px !important;
    color: #e53935 !important;
}

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 10px;
    background: linear-gradient(90deg, #1d4ed8, #3b82f6);;
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #b71c1c, #d32f2f);
}

.info-card {
    padding: 18px;
    border-radius: 12px;
    background: rgba(128, 128, 128, 0.08);
    border-left: 5px solid #e53935;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

try:
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")

except Exception as e:
    st.error("Model files could not be loaded.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted heart disease risk prediction using a trained KNN model'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
<b>How it works:</b> Enter the patient's clinical information below.
The trained machine-learning model will analyze the information and
provide a prediction category.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider(
        "Age",
        18,
        100,
        40
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

with col3:
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )


# ---------------------------------------------------
# HEART TEST INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">🫀 Heart Test Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

with col2:
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

with col3:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )


# ---------------------------------------------------
# ECG INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">📊 ECG & Exercise Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

with col2:
    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

with col3:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )


# ---------------------------------------------------
# ST INFORMATION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">📈 ST Segment Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0,
        step=0.1
    )

with col2:
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Heart Disease Risk"):

    # ------------------------------------------------
    # CREATE RAW INPUT
    # ------------------------------------------------

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


    # ------------------------------------------------
    # CREATE DATAFRAME
    # ------------------------------------------------

    input_df = pd.DataFrame([raw_input])


    # ------------------------------------------------
    # ADD MISSING COLUMNS
    # ------------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0


    # ------------------------------------------------
    # REORDER COLUMNS
    # ------------------------------------------------

    input_df = input_df[expected_columns]


    # ------------------------------------------------
    # SCALE INPUT
    # ------------------------------------------------

    scaled_input = scaler.transform(input_df)


    # ------------------------------------------------
    # MAKE PREDICTION
    # ------------------------------------------------

    prediction = model.predict(scaled_input)[0]


    # ------------------------------------------------
    # GET PROBABILITY
    # ------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(scaled_input)

        probability = probabilities[0][1]


    # ------------------------------------------------
    # PATIENT CATEGORY
    # ------------------------------------------------

    if prediction == 1:

        category = "HIGH RISK"

        message = (
            "The model predicts a higher likelihood "
            "of heart disease."
        )

    else:

        category = "LOW RISK"

        message = (
            "The model predicts a lower likelihood "
            "of heart disease."
        )


    # ------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if category == "HIGH RISK":

        st.error(
            f"🔴 {category}\n\n"
            f"{message}"
        )

    else:

        st.success(
            f"🟢 {category}\n\n"
            f"{message}"
        )


    # ------------------------------------------------
    # PROBABILITY DISPLAY
    # ------------------------------------------------

    if probability is not None:

        probability_percent = probability * 100

        st.metric(
            label="Model Probability",
            value=f"{probability_percent:.2f}%"
        )

        st.progress(float(probability))


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        ❤️ Heart Disease Prediction System<br>
        Machine Learning Based Application
    </div>
    """,
    unsafe_allow_html=True
)