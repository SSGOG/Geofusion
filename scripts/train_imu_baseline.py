"""Train and evaluate a reproducible IMU-only GeoFusion baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "imu_baseline_dataset.csv"
MODELS = ROOT / "models"
EDA = ROOT / "data" / "eda"
FEATURES = [
    "imu_sampling_hz",
    "linear_acc_rms_deviation_mps2",
    "linear_acc_peak_deviation_mps2",
    "linear_acc_jerk_rms_mps3",
    "gyro_peak_radps",
]


def metrics_for(frame: pd.DataFrame, probability) -> dict[str, float | int]:
    predicted = (probability >= 0.5).astype(int)
    target = frame["target_anomaly"].astype(int)
    values: dict[str, float | int] = {
        "rows": int(len(frame)),
        "positive_rows": int(target.sum()),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(target, probability)),
    }
    values["roc_auc"] = float(roc_auc_score(target, probability)) if target.nunique() == 2 else float("nan")
    return values


def main() -> None:
    data = pd.read_csv(DATA)
    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()
    test = data.loc[data["split"] == "test"].copy()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923)),
        ]
    )
    model.fit(train[FEATURES], train["target_anomaly"])
    metrics = {}
    predictions = {}
    for name, frame in {"train": train, "validation": validation, "test": test}.items():
        probability = model.predict_proba(frame[FEATURES])[:, 1]
        metrics[name] = metrics_for(frame, probability)
        predictions[name] = probability

    MODELS.mkdir(parents=True, exist_ok=True)
    EDA.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": FEATURES, "target": "annotated_event_vs_candidate_normal"}, MODELS / "imu_logistic_baseline.joblib")
    (MODELS / "imu_logistic_baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(test["target_anomaly"], (predictions["test"] >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Candidate normal", "Annotated event"], yticklabels=["Candidate normal", "Annotated event"], ax=ax)
    ax.set(title="IMU-only baseline: held-out sessions", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(EDA / "imu_baseline_test_confusion_matrix.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print("Important: negative rows are candidate-normal windows, not verified normal-road ground truth.")


if __name__ == "__main__":
    main()
