import os
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "processed_dataset.csv"
TARGET_COL = "Accident Severity"

DAGSHUB_OWNER = "virgiebeatrice"
DAGSHUB_REPO = "global-accident-mlflow.mlflow"      

ARTIFACT_DIR = "artifacts_basic"
PRED_DIR = os.path.join(ARTIFACT_DIR, "predictions")
PLOT_DIR = os.path.join(ARTIFACT_DIR, "plots")

for folder in [PRED_DIR, PLOT_DIR]:
    os.makedirs(folder, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def init_dagshub():
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow")
    print(f"[INFO] Tracking URI: {mlflow.get_tracking_uri()}")


def train_basic_model(data_path=DATA_PATH):
    init_dagshub()

    mlflow.set_experiment("GlobalAccident_BasicModel")

    mlflow.start_run(run_name="RandomForest_Baseline")

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_path = os.path.join(PRED_DIR, "rf_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    mlflow.log_artifact(pred_path)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Baseline)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(PLOT_DIR, "confusion_matrix_baseline.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    mlflow.log_artifact(cm_path)

    mlflow.end_run()


if __name__ == "__main__":
    train_basic_model()
