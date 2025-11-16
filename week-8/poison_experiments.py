# poison_experiments.py
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

RNG = np.random.default_rng(42)


# ---------------------------
# Load IRIS dataset
# ---------------------------
def load_iris(csv_path):
    df = pd.read_csv(csv_path)
    return df


# ---------------------------
# Feature Noise Poisoning
# ---------------------------
def apply_feature_noise(df, fraction, noise_scale=1.0):
    df2 = df.copy(deep=True).reset_index(drop=True)  # FIXED
    n = len(df2)
    k = int(np.round(n * fraction))
    idx = RNG.choice(n, size=k, replace=False)

    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    for c in numeric_cols:
        col_mean = df2[c].mean()
        col_std = df2[c].std(ddof=0) or 1.0

        noise = RNG.normal(loc=col_mean, scale=col_std * noise_scale, size=k)
        df2.loc[idx, c] = noise

    return df2


# ---------------------------
# Label Flip Poisoning
# ---------------------------
def apply_label_flip(df, fraction):
    df2 = df.copy(deep=True).reset_index(drop=True)  # FIXED
    n = len(df2)
    k = int(np.round(n * fraction))
    idx = RNG.choice(n, size=k, replace=False)

    classes = sorted(df2["species"].unique())

    for i in idx:
        current = df2.at[i, "species"]
        others = [c for c in classes if c != current]
        df2.at[i, "species"] = RNG.choice(others)

    return df2


# ---------------------------
# Train & Evaluate + MLflow Logging
# ---------------------------
def train_and_eval(df_train, df_val, params, mlflow_exp="poison_experiments"):

    X_train = df_train[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y_train = df_train["species"]

    X_val = df_val[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y_val = df_val["species"]

    mlflow.set_experiment(mlflow_exp)

    with mlflow.start_run():
        mlflow.log_params(params)

        clf = DecisionTreeClassifier(
            max_depth=params.get("max_depth", 3),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
        )

        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        preds = clf.predict(X_val)
        acc = float(accuracy_score(y_val, preds))
        f1 = float(f1_score(y_val, preds, average="macro"))
        cm = confusion_matrix(y_val, preds).tolist()

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_macro", f1)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")

        # Log model with signature
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            signature=None,
            input_example=X_train.iloc[:1],  # example row
        )

        summary = {
            "params": params,
            "val_accuracy": acc,
            "val_f1_macro": f1,
            "train_time_sec": train_time,
            "confusion_matrix": cm,
        }

        mlflow.log_dict(summary, "summary.json")

        run_id = mlflow.active_run().info.run_id
        print(f"Logged run {run_id}: acc={acc:.4f}, f1={f1:.4f}")

        return summary


# ---------------------------
# Main Experiment Runner
# ---------------------------
def run_experiments(csv_path, out_dir, poison_types, fractions):
    os.makedirs(out_dir, exist_ok=True)

    df = load_iris(csv_path)

    # Clean validation set
    train_df, val_df = train_test_split(
        df, test_size=0.3, stratify=df["species"], random_state=42
    )

    # Baseline
    baseline_params = {"poison_type": "none", "poison_frac": 0.0, "max_depth": 3}
    baseline_summary = train_and_eval(train_df, val_df, baseline_params)
    with open(os.path.join(out_dir, "baseline_summary.json"), "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # Poison Experiments
    for ptype in poison_types:
        for frac in fractions:
            if ptype == "feature_noise":
                poisoned_train = apply_feature_noise(train_df, frac, noise_scale=1.0)
            elif ptype == "label_flip":
                poisoned_train = apply_label_flip(train_df, frac)
            else:
                raise ValueError("Unknown poisoning type")

            params = {"poison_type": ptype, "poison_frac": frac, "max_depth": 3}

            summary = train_and_eval(poisoned_train, val_df, params)

            fname = f"{ptype}_{int(frac*100)}pct_summary.json"
            with open(os.path.join(out_dir, fname), "w") as f:
                json.dump(summary, f, indent=2)


# ---------------------------
# CLI Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to original iris.csv")
    parser.add_argument("--out", type=str, default="poison_results", help="Output folder")
    args = parser.parse_args()

    run_experiments(
        args.csv,
        args.out,
        poison_types=["feature_noise", "label_flip"],
        fractions=[0.05, 0.10, 0.50],
    )
