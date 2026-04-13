from pathlib import Path
import pandas as pd
import streamlit as st

from src.data import load_admission_data, preprocess_admission_data
from src.predict import predict_admission
from src.train import train_and_save_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="UCLA Admission Predictor", layout="wide")
st.title("UCLA Admission Prediction")

if st.button("Train / Refresh Model"):
    accuracy = train_and_save_model()
    st.success(f"Training complete. Accuracy: {accuracy:.2%}")

data_path = ROOT / "data" / "raw" / "Admission.csv"
if data_path.exists():
    raw_df = load_admission_data(data_path)
    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head())

    processed_df = preprocess_admission_data(raw_df)
    feature_cols = [c for c in processed_df.columns if c != "Admit_Chance"]

    st.subheader("Prediction Form")
    user_input = {}
    for col in feature_cols:
        user_input[col] = st.number_input(col, value=float(processed_df[col].median()))
    input_df = pd.DataFrame([user_input])

    if st.button("Predict Admission"):
        try:
            pred, proba = predict_admission(input_df)
            label = "Admit" if pred == 1 else "Not Admit"
            st.success(f"Prediction: {label}")
            st.write(f"Admission probability: {proba:.2%}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.warning("Place Admission.csv inside data/raw before using the app.")
