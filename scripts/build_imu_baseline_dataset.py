"""Create a session-split IMU baseline dataset for GeoFusion Phase 3.

Positive rows are manually annotated Bump/Pothole events. Negative rows are
temporally separated *candidate-normal* video windows and must be interpreted
as weak negatives until a later visual audit. Splits are assigned by session to
avoid recording-level leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_fusion_event_windows import window_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "imu_baseline_dataset.csv"
SPLITS = PROCESSED / "session_splits.csv"
RANDOM_SEED = 20260923
FEATURES = [
    "imu_sampling_hz",
    "linear_acc_rms_deviation_mps2",
    "linear_acc_peak_deviation_mps2",
    "linear_acc_jerk_rms_mps3",
    "gyro_peak_radps",
]


def make_splits(sessions: list[str]) -> pd.DataFrame:
    """Use a deterministic 70/15/15 session split."""
    ordered = np.array(sorted(set(sessions)), dtype=str)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(ordered)
    count = len(ordered)
    train_end = round(count * 0.70)
    validation_end = train_end + round(count * 0.15)
    labels = np.full(count, "test", dtype=object)
    labels[:train_end] = "train"
    labels[train_end:validation_end] = "validation"
    return pd.DataFrame({"session": ordered, "split": labels}).sort_values("session")


def main() -> None:
    positive = pd.read_csv(PROCESSED / "fusion_event_windows.csv")
    positive = positive.loc[positive["imu_samples"] >= 5].copy()
    positive["target_anomaly"] = 1
    positive["target_description"] = "annotated_road_event"
    positive["data_origin"] = "annotated_bump_or_pothole"

    candidate = pd.read_csv(PROCESSED / "candidate_normal_windows.csv")
    cache: dict[str, pd.DataFrame] = {}
    negative_rows: list[dict[str, object]] = []
    for _, row in candidate.iterrows():
        sensor_path = Path(row["sensor_csv_path"])
        key = str(sensor_path)
        if key not in cache:
            cache[key] = pd.read_csv(sensor_path, low_memory=False)
        values = window_features(cache[key], float(row["window_centre_seconds"]))
        if values.get("imu_samples", 0) < 5:
            continue
        negative_rows.append(
            {
                "event_id": row["normal_window_id"],
                "session": str(row["session"]),
                "event_type": "Candidate normal",
                "event_seconds": row["window_centre_seconds"],
                "target_anomaly": 0,
                "target_description": "candidate_normal_not_visually_verified",
                "data_origin": "temporally_separated_candidate_normal",
                **values,
            }
        )
    negative = pd.DataFrame(negative_rows)
    columns = ["event_id", "session", "event_type", "event_seconds", "target_anomaly", "target_description", "data_origin", *FEATURES]
    dataset = pd.concat([positive.reindex(columns=columns), negative.reindex(columns=columns)], ignore_index=True)
    dataset = dataset.dropna(subset=FEATURES).copy()
    # CSV inference can turn session 1 into an integer in one manifest and a
    # string in another. A canonical string ID is essential for leakage-safe
    # group splits and merges.
    dataset["session"] = dataset["session"].astype(str)
    splits = make_splits(dataset["session"].astype(str).tolist())
    dataset = dataset.merge(splits, on="session", how="left", validate="many_to_one")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT, index=False)
    splits.to_csv(SPLITS, index=False)
    print(f"Baseline rows: {len(dataset)}")
    print(f"Class counts: {dataset['target_anomaly'].value_counts().to_dict()}")
    print(f"Rows by split: {dataset['split'].value_counts().to_dict()}")
    print(f"Class counts by split: {dataset.groupby(['split', 'target_anomaly']).size().to_dict()}")


if __name__ == "__main__":
    main()
