import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

DATA_PATH = "processed_dataset.csv"
TARGET_COL = "Accident Severity"

DAGSHUB_OWNER = "virgiebeatrice"
DAGSHUB_REPO = "global-accident-mlflow.mlflow"      

# clean artifact directories for baseline model
ARTIFACT_DIR = "artifacts_basic"
PRED_DIR = os.path.join(ARTIFACT_DIR, "predictions")
PLOT_DIR = os.path.join(ARTIFACT_DIR, "plots")

for folder in [PRED_DIR, PLOT_DIR]:
    os.makedirs(folder, exist_ok=True)


def load_data(path: str):
    """
    Load dataset and return feature matrix X and target y.
    """
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' is missing. "
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
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
    )
    print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")


def train_basic_model(data_path: str = DATA_PATH):
    """
    Train a baseline RandomForest model with MLflow autolog.
    Artifacts stored neatly in artifacts_basic/.
    """

    # MLflow local tracking
    # mlflow.set_tracking_uri("file:./mlruns")

    # Dagshub
    init_dagshub()

    mlflow.set_experiment("GlobalAccident_BasicModel")

    # Load dataset
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Enable autolog 
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True
    )

    with mlflow.start_run(run_name="RandomForest_Baseline", nested=False):

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print("\n=== Baseline Model Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-weighted: {f1:.4f}")

        # Prediction Sample
        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
        })

        pred_path = os.path.join(PRED_DIR, "rf_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Baseline)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = os.path.join(PLOT_DIR, "confusion_matrix_baseline.png")
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(cm_path)


if __name__ == "__main__":
    train_basic_model()
