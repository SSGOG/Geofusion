"""Compare YOLO/clip visual representations with IMU on validation sessions only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
EDA = ROOT / "data" / "eda"
IMU_FEATURES = ["imu_sampling_hz", "linear_acc_rms_deviation_mps2", "linear_acc_peak_deviation_mps2", "linear_acc_jerk_rms_mps3", "gyro_peak_radps"]


def score(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(target)), "positive_rows": int(target.sum()), "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)), "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)), "average_precision": float(average_precision_score(target, probability)), "roc_auc": float(roc_auc_score(target, probability)),
    }


def main() -> None:
    base = pd.read_csv(PROCESSED / "imu_baseline_dataset.csv")
    clip_predictions = pd.read_csv(PROCESSED / "clip_validation_predictions.csv")
    yolo_predictions = pd.read_csv(PROCESSED / "yolo_validation_predictions.csv")
    archive = np.load(PROCESSED / "clip_encoder_embeddings.npz")
    embeddings = pd.DataFrame({"event_id": archive["event_id"].astype(str), "embedding_index": np.arange(len(archive["event_id"]))})
    base["event_id"] = base["event_id"].astype(str)
    data = base.merge(embeddings, on="event_id", how="inner", validate="one_to_one")
    # The clip prediction file contains validation only; its training embeddings are
    # still used for early fusion, while validation clip probabilities join here.
    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()
    validation = validation.merge(yolo_predictions[["event_id", "yolo_event_probability"]], on="event_id", how="inner", validate="one_to_one")
    validation = validation.merge(clip_predictions[["event_id", "clip_event_probability"]], on="event_id", how="inner", validate="one_to_one")
    y_train = train["target_anomaly"].to_numpy(dtype=int); y_validation = validation["target_anomaly"].to_numpy(dtype=int)
    imu_train = train[IMU_FEATURES].to_numpy(dtype=float); imu_validation = validation[IMU_FEATURES].to_numpy(dtype=float)
    visual_train = archive["embeddings"][train["embedding_index"].to_numpy()]; visual_validation = archive["embeddings"][validation["embedding_index"].to_numpy()]

    imu_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923))])
    imu_model.fit(imu_train, y_train)
    imu_probability = imu_model.predict_proba(imu_validation)[:, 1]
    early_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923))])
    early_model.fit(np.hstack([visual_train, imu_train]), y_train)
    early_probability = early_model.predict_proba(np.hstack([visual_validation, imu_validation]))[:, 1]
    yolo_probability = validation["yolo_event_probability"].to_numpy(dtype=float)
    clip_probability = validation["clip_event_probability"].to_numpy(dtype=float)
    # Equal-weight probability pooling is pre-specified, so validation labels are
    # not reused to fit a meta-classifier.
    late_yolo_probability = (yolo_probability + imu_probability) / 2
    late_clip_probability = (clip_probability + imu_probability) / 2
    results = {
        "selection_split": "validation_only",
        "imu_only_restricted_to_clip_coverage": score(y_validation, imu_probability),
        "yolo_only": score(y_validation, yolo_probability),
        "clip_only": score(y_validation, clip_probability),
        "early_clip_imu": score(y_validation, early_probability),
        "late_yolo_imu": score(y_validation, late_yolo_probability),
        "late_clip_imu": score(y_validation, late_clip_probability),
    }
    choices = ["early_clip_imu", "late_yolo_imu", "late_clip_imu"]
    selected = max(choices, key=lambda name: (results[name]["f1"], results[name]["average_precision"]))
    selection = {"selected_method": selected, "selection_metric": "validation_f1_then_average_precision", "test_set_status": "not_evaluated", "late_fusion_rule": "equal_weight_visual_and_imu_probabilities"}
    predictions = validation[["event_id", "session", "target_anomaly"]].copy()
    for name, values in {"imu_probability": imu_probability, "yolo_probability": yolo_probability, "clip_probability": clip_probability, "early_clip_imu_probability": early_probability, "late_yolo_imu_probability": late_yolo_probability, "late_clip_imu_probability": late_clip_probability}.items(): predictions[name] = values
    MODELS.mkdir(parents=True, exist_ok=True); EDA.mkdir(parents=True, exist_ok=True)
    (MODELS / "enhanced_fusion_validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (MODELS / "enhanced_fusion_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    predictions.to_csv(PROCESSED / "enhanced_fusion_validation_predictions.csv", index=False)
    import matplotlib.pyplot as plt
    names = ["early_clip_imu", "late_yolo_imu", "late_clip_imu"]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(names, [results[name]["f1"] for name in names], color=["#4C78A8", "#F58518", "#54A24B"]); ax.set(ylim=(0, 1), ylabel="Validation F1", title="Enhanced fusion comparison")
    for i, name in enumerate(names): ax.text(i, results[name]["f1"] + .02, f"{results[name]['f1']:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(EDA / "enhanced_fusion_validation_comparison.png", dpi=220); plt.close(fig)
    print(json.dumps(results, indent=2)); print(json.dumps(selection, indent=2))


if __name__ == "__main__": main()
