"""Validate the vision-baseline data alignment and artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    manifest = pd.read_csv(PROCESSED / "vision_baseline_manifest.csv")
    archive = np.load(PROCESSED / "vision_hog_features.npz")
    metrics = json.loads((MODELS / "vision_hog_baseline_metrics.json").read_text(encoding="utf-8"))
    require(len(manifest) == 729, "vision manifest matches the IMU baseline dataset")
    require(len(archive["event_id"]) == len(manifest), "one HOG vector exists per manifest row")
    require(archive["features"].shape[0] == len(manifest) and archive["features"].shape[1] > 1000, "HOG feature matrix has expected shape")
    require(set(manifest["split"]) == {"train", "validation", "test"}, "vision data uses all frozen session splits")
    require(0 <= metrics["validation"]["f1"] <= 1 and 0 <= metrics["test"]["f1"] <= 1, "vision metrics are valid")
    require((MODELS / "vision_hog_baseline.joblib").exists(), "trained vision baseline is saved")
    print("\nPhase 4A validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
