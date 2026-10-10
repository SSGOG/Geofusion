"""Validate the reproducible data-preparation outputs for GeoFusion Phase 1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EDA = ROOT / "data" / "eda"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    anomaly_files = list((PROCESSED / "anomaly_csv").glob("*.csv"))
    normal_files = list((PROCESSED / "normal_csv").glob("*.csv"))
    require(len(anomaly_files) >= 90, f"anomaly recordings available ({len(anomaly_files)})")
    require(len(normal_files) >= 5, f"normal recordings available ({len(normal_files)})")

    profile = json.loads((EDA / "dataset_profile.json").read_text(encoding="utf-8"))
    require(profile["recordings"] == len(anomaly_files) + len(normal_files), "EDA profile matches extracted recording count")
    require(profile["modalities_present"]["inertial"], "inertial fields are present")

    video = pd.read_csv(PROCESSED / "video_annotation_manifest.csv")
    require({"session", "video_path", "event_type", "seconds_elapsed", "target_frame"}.issubset(video.columns), "video manifest has alignment fields")
    require(len(video) >= 400 and video["session"].nunique() >= 80, "local video-event coverage is sufficient for Phase 1")
    require(video["frame_within_video"].astype(bool).all(), "annotated target frames are within local video bounds")

    fusion = pd.read_csv(PROCESSED / "fusion_event_windows.csv")
    feature_columns = {"linear_acc_rms_deviation_mps2", "linear_acc_peak_deviation_mps2", "linear_acc_jerk_rms_mps3", "gyro_peak_radps"}
    require(feature_columns.issubset(fusion.columns), "fusion manifest has IMU impact features")
    require(len(fusion) == len(video), "every local video event has a fusion-manifest row")
    require((fusion["window_start_seconds"] <= fusion["event_seconds"]).all(), "all windows start before their event")
    require((fusion["window_end_seconds"] >= fusion["event_seconds"]).all(), "all windows end after their event")
    missing_windows = int((fusion["imu_samples"] < 5).sum())
    print(f"INFO  {missing_windows} event windows have insufficient IMU samples and are explicitly flagged for exclusion during training.")
    print("\nPhase 1 validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
