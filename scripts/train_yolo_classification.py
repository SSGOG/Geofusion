"""Train a compact YOLO classification model on GeoFusion aligned frames."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "derived" / "yolo_cls"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RUNS = MODELS / "runs"


def metrics(target, probability) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(target)),
        "positive_rows": int(target.sum()),
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def main() -> None:
    # Keep the downloaded official checkpoint in the project so training is
    # reproducible and does not depend on a later network connection.
    model = YOLO(str(MODELS / "yolo11n-cls.pt"))
    run = model.train(
        data=str(DATASET_ROOT),
        epochs=15,
        imgsz=224,
        batch=16,
        device=0,
        amp=False,
        workers=0,
        patience=5,
        pretrained=True,
        seed=20260923,
        project=str(RUNS),
        name="geofusion_yolo_cls",
        exist_ok=True,
        verbose=False,
    )
    best_path = Path(run.save_dir) / "weights" / "best.pt"
    best = YOLO(str(best_path))
    source = pd.read_csv(DATASET_ROOT / "source_manifest.csv")
    validation = source.loc[source["split"] == "validation"].copy().reset_index(drop=True)
    results = best.predict(validation["image_path"].tolist(), imgsz=224, device=0, verbose=False)
    event_index = next(index for index, name in results[0].names.items() if name == "annotated_event")
    probabilities = [float(result.probs.data[event_index].cpu()) for result in results]
    validation["yolo_event_probability"] = probabilities
    output = PROCESSED / "yolo_validation_predictions.csv"
    validation[["event_id", "session", "target_anomaly", "yolo_event_probability"]].to_csv(output, index=False)
    outcome = metrics(validation["target_anomaly"].to_numpy(dtype=int), validation["yolo_event_probability"].to_numpy())
    payload = {"model": "yolo11n-cls", "selection_split": "validation_only", **outcome, "weights": str(best_path)}
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "yolo_classification_validation_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Test images and test metrics were not used.")


if __name__ == "__main__":
    main()
