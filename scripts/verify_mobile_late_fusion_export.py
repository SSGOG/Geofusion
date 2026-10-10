"""Verify that portable IMU JSON reproduces the calibrated sklearn pipeline."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "phase4d_final_imu_pipeline.joblib"
EXPORT = ROOT / "mobile" / "assets" / "models" / "late_fusion_imu.json"
DATA = ROOT / "data" / "processed" / "imu_baseline_dataset.csv"
EARLY_ARTIFACT = ROOT / "models" / "early_fusion_validation_model.joblib"


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)
    print(f"PASS  {message}")


def portable_probability(model: dict[str, object], row: pd.Series) -> float:
    imu = model["imu"]
    features = imu["feature_order"]
    medians = imu["imputer_median"]
    means = imu["scaler_mean"]
    scales = imu["scaler_scale"]
    coefficients = imu["positive_class_coefficients"]
    overrides = imu["input_overrides"]
    values = []
    for index, feature in enumerate(features):
        value = overrides.get(feature, row[feature])
        values.append(medians[index] if pd.isna(value) else float(value))
    logit = float(imu["positive_class_intercept"]) + sum(coefficient * ((value - mean) / scale) for value, mean, scale, coefficient in zip(values, means, scales, coefficients, strict=True))
    return 1 / (1 + math.exp(-logit))


def main() -> None:
    exported = json.loads(EXPORT.read_text(encoding="utf-8"))
    artifact = joblib.load(SOURCE)
    pipeline = artifact["pipeline"]
    rows = pd.read_csv(DATA).head(20)
    features = artifact["features"]
    for _, row in rows.iterrows():
        calibrated = row[features].copy()
        calibrated["imu_sampling_hz"] = 100.0
        sklearn_probability = float(pipeline.predict_proba(pd.DataFrame([calibrated]))[:, 1][0])
        assert abs(portable_probability(exported, row) - sklearn_probability) < 1e-12
    require(exported["selected_method"] == "late_yolo_imu", "portable export identifies the selected late-fusion method")
    require(EXPORT.exists(), "portable selected-model artifact exists")
    require(EARLY_ARTIFACT.exists(), "early-fusion research comparator remains preserved")
    print("\nPortable late-fusion IMU export matches the calibrated sklearn pipeline.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, KeyError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
