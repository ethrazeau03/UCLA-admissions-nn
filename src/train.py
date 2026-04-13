from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from .data import load_admission_data, preprocess_admission_data
from .logger import setup_logger

LOGGER = setup_logger("ucla_train")
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def train_and_save_model():
    df = load_admission_data(ROOT / "data" / "raw" / "Admission.csv")
    df = preprocess_admission_data(df)

    X = df.drop(columns=["Admit_Chance"])
    y = df["Admit_Chance"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    model = MLPClassifier(hidden_layer_sizes=(3,), batch_size=50, max_iter=300, random_state=123)
    model.fit(xtrain_scaled, ytrain)

    preds = model.predict(xtest_scaled)
    accuracy = float(accuracy_score(ytest, preds))
    LOGGER.info("UCLA NN accuracy: %.4f", accuracy)

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), ARTIFACT_DIR / "feature_columns.joblib")
    joblib.dump({"accuracy": accuracy}, ARTIFACT_DIR / "metrics.joblib")

    return accuracy


if __name__ == "__main__":
    train_and_save_model()
