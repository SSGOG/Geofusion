"""Create a transparent sensor-derived provisional impact priority for GeoFusion.

This is an operational prioritisation signal, not a ground-truth severity label.
It ranks the observed phone response relative to this dataset and is intended for
the initial GIS map and active validation workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "fusion_event_windows.csv"
OUTPUT = ROOT / "data" / "processed" / "provisional_event_priority.csv"
FEATURES = {
    "linear_acc_rms_deviation_mps2": 0.40,
    "linear_acc_peak_deviation_mps2": 0.30,
    "linear_acc_jerk_rms_mps3": 0.20,
    "gyro_peak_radps": 0.10,
}


def percentile_score(values: pd.Series) -> pd.Series:
    """Robust 0--1 score after log transform for positively skewed IMU values."""
    transformed = np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0))
    return transformed.rank(pct=True, method="average")


def main() -> None:
    events = pd.read_csv(INPUT)
    events["usable_imu_window"] = events["imu_samples"] >= 5
    usable = events.loc[events["usable_imu_window"]].copy()

    weighted = pd.Series(0.0, index=usable.index)
    for feature, weight in FEATURES.items():
        weighted += weight * percentile_score(usable[feature])
    # 0--100 describes relative sensor impact within this supplied dataset.
    events["provisional_impact_score_0_100"] = np.nan
    events.loc[usable.index, "provisional_impact_score_0_100"] = weighted * 100
    events["provisional_impact_priority"] = pd.NA
    events.loc[usable.index, "provisional_impact_priority"] = pd.qcut(
        weighted.rank(method="first"), 3, labels=["low_impact", "medium_impact", "high_impact"]
    ).astype(str).to_numpy()
    events["priority_method"] = "weighted_dataset_percentile_v1"
    events.loc[~events["usable_imu_window"], "priority_method"] = "unavailable_insufficient_imu"
    events["severity_ground_truth_available"] = False
    events["interpretation"] = "relative phone-sensor impact priority; not verified road-defect severity"

    columns = [
        "event_id", "session", "event_type", "event_latitude", "event_longitude", "event_seconds", "video_path",
        "linear_acc_rms_deviation_mps2", "linear_acc_peak_deviation_mps2", "linear_acc_jerk_rms_mps3", "gyro_peak_radps",
        "imu_samples", "usable_imu_window", "provisional_impact_score_0_100", "provisional_impact_priority",
        "priority_method", "severity_ground_truth_available", "interpretation",
    ]
    output = events.reindex(columns=columns)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(f"Priority events written: {len(output)}")
    print(f"Usable IMU windows: {int(output['usable_imu_window'].sum())}")
    print(f"Priority counts: {output['provisional_impact_priority'].value_counts(dropna=False).to_dict()}")
    print(f"GPS fixes available: {int(output.dropna(subset=['event_latitude', 'event_longitude']).shape[0])}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
