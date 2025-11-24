import os
import pandas as pd
import mlflow
import mlflow.sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


DATA_PATH = "processed_dataset.csv"
TARGET_COL = "Accident Severity"

ARTIFACT_DIR = "artifacts_ci"
PRED_DIR = os.path.join(ARTIFACT_DIR, "predictions")
PLOT_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train():
    mlflow.set_experiment("CI_Training")

    with mlflow.start_run(run_name="ci_run", nested=True):
        X, y = load_data(DATA_PATH)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        pred_path = os.path.join(PRED_DIR, "pred_ci.csv")
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("CM - CI Run")
        cm_path = os.path.join(PLOT_DIR, "cm_ci.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

if __name__ == "__main__":
    train()