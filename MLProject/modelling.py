import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "processed_dataset.csv"
TARGET_COL = "Accident Severity"

DAGSHUB_OWNER = "virgiebeatrice"
DAGSHUB_REPO = "global-accident-mlflow.mlflow"       

def load_data(path: str):
    """
    Load the preprocessed dataset and split into
    feature matrix (X) and target vector (y).
    """
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y

def init_dagshub():
    """
    Initialize DagsHub repository as the remote MLflow tracking server.
    Make sure credentials (username/token) are configured locally.
    """
    dagshub.init(
        repo_owner=DAGSHUB_OWNER,
        repo_name=DAGSHUB_REPO,
        mlflow=True,
    )

    print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")


def train_model(data_path: str, mode: str = "local"):
    """
    Train a basic RandomForestClassifier using manual MLflow logging.
    Modes:
      - "local": log to local MLflow (mlruns/)
      - "dagshub": log to DagsHub (online)
    """

    # Select MLflow tracking mode
    if mode == "local":
        mlflow.set_tracking_uri("file:./mlruns")
        print("[INFO] Tracking locally to ./mlruns")
    elif mode == "dagshub":
        init_dagshub()
    else:
        raise ValueError("Mode must be either 'local' or 'dagshub'.")

    mlflow.set_experiment(f"GlobalAccident_BasicModel_{mode}")

    # Load data
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=f"RF_Basic_{mode}"):

        # Model Training
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predictions & Metrics
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # Manual Logging 
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)

        # Log Model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # Prediction Sample
        df_preds = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        })
        pred_path = "prediction_sample.csv"
        df_preds.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)

        # Console Output
        print("=== Basic Model Evaluation ===")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="local")
    args = parser.parse_args()

    train_model(DATA_PATH, mode=args.mode)
