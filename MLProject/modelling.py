import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

DATA_PATH = "processed_dataset.csv"      
TARGET_COL = "Accident Severity"        

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


def train_model(data_path: str = DATA_PATH):
    """
    Train a baseline RandomForest model using MLflow autologging.
    - Tracks parameters, metrics, artifacts automatically.
    - Saves results to a local MLflow tracking directory (mlruns/).
    """

    # Use local directory for MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("GlobalAccident_CI")

    # Load dataset
    X, y = load_data(data_path)

    # Train-test split (stratified for balanced class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Enable MLflow autologging
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True
    )

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Print summary
        print("=== Baseline Evaluation ===")
        print(f"Accuracy         : {acc:.4f}")
        print(f"F1-score weighted: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_model()
