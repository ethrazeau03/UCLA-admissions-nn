from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def predict_admission(input_df: pd.DataFrame):
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    scaler = joblib.load(ROOT / "artifacts" / "scaler.joblib")
    feature_columns = joblib.load(ROOT / "artifacts" / "feature_columns.joblib")

    missing_cols = [col for col in feature_columns if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[feature_columns]
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]
    return int(pred), float(proba)
