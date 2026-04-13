from pathlib import Path
import pandas as pd


def load_admission_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def preprocess_admission_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Admit_Chance" not in df.columns:
        raise ValueError("Target column 'Admit_Chance' not found.")

    df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

    if "Serial_No" in df.columns:
        df = df.drop(columns=["Serial_No"])

    for col in ["University_Rating", "Research"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    dummy_cols = [col for col in ["University_Rating", "Research"] if col in df.columns]
    if dummy_cols:
        df = pd.get_dummies(df, columns=dummy_cols, dtype=int)

    return df
