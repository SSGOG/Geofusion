"""Validate GeoFusion Phase 3 split integrity and IMU baseline artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    data = pd.read_csv(PROCESSED / "imu_baseline_dataset.csv")
    splits = pd.read_csv(PROCESSED / "session_splits.csv")
    require(set(data["split"]) == {"train", "validation", "test"}, "all three splits exist")
    for split in ["train", "validation", "test"]:
        frame = data.loc[data["split"] == split]
        require(set(frame["target_anomaly"]) == {0, 1}, f"{split} contains both target classes")
    duplicated_sessions = splits.duplicated("session").sum()
    require(duplicated_sessions == 0, "each session has one split assignment")
    train_sessions = set(data.loc[data["split"] == "train", "session"])
    validation_sessions = set(data.loc[data["split"] == "validation", "session"])
    test_sessions = set(data.loc[data["split"] == "test", "session"])
    require(not (train_sessions & validation_sessions or train_sessions & test_sessions or validation_sessions & test_sessions), "no session occurs in more than one split")
    metrics = json.loads((MODELS / "imu_logistic_baseline_metrics.json").read_text(encoding="utf-8"))
    require(0 <= metrics["test"]["f1"] <= 1, "held-out F1 is a valid metric")
    require(0 <= metrics["test"]["average_precision"] <= 1, "held-out average precision is a valid metric")
    require((MODELS / "imu_logistic_baseline.joblib").exists(), "trained IMU baseline is saved")
    print("\nPhase 3 validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
