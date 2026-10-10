"""Export the selected late-fusion IMU branch as a portable JSON artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "phase4d_final_imu_pipeline.joblib"
DESTINATION = ROOT / "mobile" / "assets" / "models" / "late_fusion_imu.json"
REFERENCE_SAMPLING_HZ = 100.0


def main() -> None:
    artifact = joblib.load(SOURCE)
    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    features = artifact["features"]
    payload = {
        "schema_version": 1,
        "selected_method": "late_yolo_imu",
        "fusion": {"rule": "equal_weight_yolo_and_imu_probabilities", "candidate_threshold": 0.5},
        "imu": {
            "feature_order": features,
            "imputer_median": imputer.statistics_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "positive_class_coefficients": classifier.coef_[0].tolist(),
            "positive_class_intercept": float(classifier.intercept_[0]),
            "input_overrides": {"imu_sampling_hz": REFERENCE_SAMPLING_HZ},
        },
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported selected late-fusion IMU artifact to {DESTINATION}")


if __name__ == "__main__":
    main()
