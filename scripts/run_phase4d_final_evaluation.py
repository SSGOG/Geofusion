"""Evaluate the validation-selected late YOLO plus IMU fusion once on held-out sessions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
EDA = ROOT / "data" / "eda"
SOURCE = ROOT / "data" / "derived" / "yolo_cls" / "source_manifest.csv"
FINAL_DATASET = ROOT / "data" / "derived" / "final_yolo_cls"
METRICS_PATH = MODELS / "phase4d_final_late_yolo_imu_metrics.json"
PREDICTIONS_PATH = PROCESSED / "phase4d_final_late_yolo_imu_test_predictions.csv"
IMU_FEATURES = [
    "imu_sampling_hz",
    "linear_acc_rms_deviation_mps2",
    "linear_acc_peak_deviation_mps2",
    "linear_acc_jerk_rms_mps3",
    "gyro_peak_radps",
]


def metric_values(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(target)),
        "positive_rows": int(target.sum()),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def save_test_frame(row: pd.Series) -> Path:
    output = FINAL_DATASET / "test" / row.class_name / f"{row.event_id}.jpg"
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(row.video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, float(row.event_seconds) * 1000)
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise RuntimeError(f"Could not decode held-out frame for {row.event_id}")
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Could not write {output}")
    return output


def prepare_final_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE)
    source["event_id"] = source["event_id"].astype(str)
    baseline = pd.read_csv(PROCESSED / "vision_baseline_manifest.csv")
    baseline["event_id"] = baseline["event_id"].astype(str)
    test = baseline.loc[baseline["split"].eq("test")].copy()
    test["class_name"] = test["target_anomaly"].map({1: "annotated_event", 0: "candidate_normal"})
    test["image_path"] = [str(save_test_frame(row)) for _, row in test.iterrows()]

    development = source.loc[source["split"].isin(["train", "validation"])].copy()
    for row in development.itertuples(index=False):
        destination = FINAL_DATASET / "train" / row.class_name / f"{row.event_id}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(Path(row.image_path), destination)
    development.to_csv(FINAL_DATASET / "development_manifest.csv", index=False)
    test.to_csv(FINAL_DATASET / "test_manifest.csv", index=False)
    return development, test


def main() -> None:
    if METRICS_PATH.exists() or PREDICTIONS_PATH.exists():
        raise RuntimeError("Phase 4D outputs already exist; the held-out test set must not be evaluated a second time.")
    selection = json.loads((MODELS / "enhanced_fusion_selection.json").read_text(encoding="utf-8"))
    if selection["selected_method"] != "late_yolo_imu":
        raise RuntimeError("Phase 4D only evaluates the validation-selected late_yolo_imu method.")

    MODELS.mkdir(parents=True, exist_ok=True)
    EDA.mkdir(parents=True, exist_ok=True)
    development, test = prepare_final_dataset()
    model = YOLO(str(MODELS / "yolo11n-cls.pt"))
    run = model.train(
        data=str(FINAL_DATASET), epochs=15, imgsz=224, batch=16, device=0,
        amp=False, workers=0, patience=5, pretrained=True, seed=20260923,
        val=False, project=str(MODELS / "runs"), name="geofusion_yolo_cls_final", exist_ok=True, verbose=False,
    )
    best_path = Path(run.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        best_path = Path(run.save_dir) / "weights" / "last.pt"
    yolo = YOLO(str(best_path))
    results = yolo.predict(test["image_path"].tolist(), imgsz=224, device=0, verbose=False)
    event_index = next(index for index, name in results[0].names.items() if name == "annotated_event")
    test["yolo_event_probability"] = [float(result.probs.data[event_index].cpu()) for result in results]

    data = pd.read_csv(PROCESSED / "imu_baseline_dataset.csv")
    data["event_id"] = data["event_id"].astype(str)
    train = data.loc[data["split"].isin(["train", "validation"])].copy()
    test_imu = data.loc[data["split"].eq("test")].copy()
    imu = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923)),
    ])
    imu.fit(train[IMU_FEATURES], train["target_anomaly"])
    test_imu["imu_event_probability"] = imu.predict_proba(test_imu[IMU_FEATURES])[:, 1]
    output = test[["event_id", "session", "target_anomaly", "yolo_event_probability"]].merge(
        test_imu[["event_id", "target_anomaly", "imu_event_probability"]], on="event_id", how="inner", validate="one_to_one", suffixes=("", "_imu")
    )
    if len(output) != len(test) or len(output) != len(test_imu):
        raise RuntimeError("The final visual and IMU test records do not align one-to-one.")
    if not output["target_anomaly"].equals(output["target_anomaly_imu"]):
        raise RuntimeError("The final visual and IMU test labels do not agree.")
    output = output.drop(columns="target_anomaly_imu")
    output["late_yolo_imu_probability"] = (output["yolo_event_probability"] + output["imu_event_probability"]) / 2
    target = output["target_anomaly"].to_numpy(dtype=int)
    probability = output["late_yolo_imu_probability"].to_numpy(dtype=float)
    validation = json.loads((MODELS / "enhanced_fusion_validation_metrics.json").read_text(encoding="utf-8"))
    payload = {
        "method": "late_yolo_imu",
        "selection": "selected_on_validation_before_test_evaluation",
        "validation_comparison": {
            "early_clip_imu": validation["early_clip_imu"],
            "late_yolo_imu": validation["late_yolo_imu"],
        },
        "test": metric_values(target, probability),
        "test_set_status": "evaluated_once",
        "test_definition": "annotated_road_event_vs_temporally_separated_candidate_normal_window",
        "yolo_weights": str(best_path),
    }
    output.to_csv(PREDICTIONS_PATH, index=False)
    joblib.dump({"pipeline": imu, "features": IMU_FEATURES, "late_fusion": "equal_weight_yolo_and_imu_probabilities"}, MODELS / "phase4d_final_imu_pipeline.joblib")
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    matrix = confusion_matrix(target, probability >= 0.5, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Candidate normal", "Annotated event"], yticklabels=["Candidate normal", "Annotated event"], ax=axes[0])
    axes[0].set(title="Selected late fusion: held-out test", xlabel="Predicted", ylabel="Actual")
    compared = ["Early clip + IMU\n(validation)", "Late YOLO + IMU\n(validation)", "Late YOLO + IMU\n(test)"]
    values = [validation["early_clip_imu"]["f1"], validation["late_yolo_imu"]["f1"], payload["test"]["f1"]]
    axes[1].bar(compared, values, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set(title="Early vs late fusion evidence", ylabel="F1", ylim=(0, 1))
    for index, value in enumerate(values): axes[1].text(index, value + .025, f"{value:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(EDA / "phase4d_fusion_comparison_and_test.png", dpi=220); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
