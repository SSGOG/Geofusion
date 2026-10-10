"""Validate Phase 4C1 YOLO classification inputs and outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "derived" / "yolo_cls" / "source_manifest.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    source = pd.read_csv(DATASET)
    prediction = pd.read_csv(PROCESSED / "yolo_validation_predictions.csv")
    metrics = json.loads((MODELS / "yolo_classification_validation_metrics.json").read_text(encoding="utf-8"))
    require(set(source["split"]) == {"train", "validation"}, "YOLO data excludes held-out test sessions")
    require(source["image_path"].map(lambda path: Path(path).exists()).all(), "every YOLO manifest image exists")
    require(set(source["class_name"]) == {"annotated_event", "candidate_normal"}, "both classification classes are present")
    require(len(prediction) == 109 and prediction["event_id"].is_unique, "one YOLO probability exists per validation row")
    require(prediction["yolo_event_probability"].between(0, 1).all(), "YOLO probabilities are bounded")
    require(0 <= metrics["f1"] <= 1 and 0 <= metrics["average_precision"] <= 1, "YOLO validation metrics are valid")
    print("\nYOLO classification validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
