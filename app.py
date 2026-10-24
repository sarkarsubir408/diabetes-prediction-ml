"""Streamlit interface for an educational diabetes ML demo."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model.joblib"

st.set_page_config(page_title="Diabetes ML Demo", page_icon="🩺", layout="centered")

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` first."
        )
    # Only load model artifacts you created yourself. Joblib/pickle files can
    # execute code when loaded and must not be accepted from untrusted sources.
    return joblib.load(MODEL_PATH)

st.title("🩺 Diabetes Classification — Educational Demo")
st.warning(
    "Not for medical use. This model is not clinically validated and cannot "
    "diagnose diabetes or estimate an individual's clinical risk. Do not use "
    "its output to make healthcare decisions. Consult a qualified healthcare professional."
)
st.write(
    "Enter example feature values to see how a trained machine-learning model classifies "
    "the record. Zero values for selected measurements are treated as missing by the pipeline."
)

try:
    bundle = load_bundle()
    model = bundle["model"]
    features = bundle["features"]
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=110, step=1, help="Dataset units; 0 is treated as missing.")
        blood_pressure = st.number_input("Blood pressure", min_value=0, max_value=200, value=70, step=1, help="Dataset units; 0 is treated as missing.")
        skin_thickness = st.number_input("Skin thickness", min_value=0, max_value=100, value=20, step=1, help="Dataset units; 0 is treated as missing.")
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1, help="Dataset units; 0 is treated as missing.")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="0 is treated as missing.")
        pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.47, step=0.01, format="%.3f")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
    submitted = st.form_submit_button("Run demo prediction")

if submitted:
    row = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
        columns=features,
    )
    predicted_class = int(model.predict(row)[0])
    st.subheader("Model output")
    st.write(f"Predicted dataset class: **{predicted_class}**")
    st.caption("Class 0 and class 1 refer to labels in the training dataset, not a medical diagnosis.")
    st.info("This demo does not display a clinical risk estimate. KNN class probabilities are not automatically calibrated.")
