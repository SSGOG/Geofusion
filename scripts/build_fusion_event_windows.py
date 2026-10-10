r"""Create aligned RGB--IMU event windows and sensor impact features for GeoFusion.

This script does not copy videos.  It references the source MP4 files in
E:\downloads_2\Raw Data and pairs every locally available labelled video event
with a three-second sensor window from the combined anomaly CSV files.

The resulting impact measures are *annotation priorities*, not severity labels.
RoadSens-4M labels event type (Bump/Pothole), not mild/moderate/severe damage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "video_annotation_manifest.csv"
ANOMALY_CSV = ROOT / "data" / "processed" / "anomaly_csv"
OUT = ROOT / "data" / "processed" / "fusion_event_windows.csv"
WINDOW_SECONDS = 1.5


def magnitude(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return np.sqrt((values**2).sum(axis=1))


def window_features(recording: pd.DataFrame, centre_seconds: float) -> dict[str, float | int]:
    time = pd.to_numeric(recording["seconds_elapsed"], errors="coerce")
    data = recording.loc[(time >= centre_seconds - WINDOW_SECONDS) & (time <= centre_seconds + WINDOW_SECONDS)].copy()
    if len(data) < 5:
        return {"imu_samples": len(data)}

    timestamps = pd.to_numeric(data["seconds_elapsed"], errors="coerce").to_numpy(dtype=float)
    linear_acc = magnitude(data, ["totalAcceleration_x", "totalAcceleration_y", "totalAcceleration_z"]).to_numpy(dtype=float)
    gyro = magnitude(data, ["gyroscope_x", "gyroscope_y", "gyroscope_z"]).to_numpy(dtype=float)
    baseline = float(np.nanmedian(linear_acc))
    deviation = np.abs(linear_acc - baseline)
    valid_steps = np.diff(timestamps)
    valid_steps = valid_steps[valid_steps > 0]
    median_step = float(np.nanmedian(valid_steps)) if len(valid_steps) else np.nan
    jerk = np.diff(linear_acc) / np.diff(timestamps)
    jerk = jerk[np.isfinite(jerk)]

    return {
        "imu_samples": int(len(data)),
        "imu_sampling_hz": float(1 / median_step) if np.isfinite(median_step) and median_step else np.nan,
        "linear_acc_baseline_mps2": baseline,
        "linear_acc_peak_deviation_mps2": float(np.nanmax(deviation)),
        "linear_acc_p95_deviation_mps2": float(np.nanpercentile(deviation, 95)),
        "linear_acc_rms_deviation_mps2": float(np.sqrt(np.nanmean(deviation**2))),
        "linear_acc_jerk_rms_mps3": float(np.sqrt(np.nanmean(jerk**2))) if len(jerk) else np.nan,
        "gyro_peak_radps": float(np.nanmax(gyro)),
    }


def event_location(recording: pd.DataFrame, centre_seconds: float) -> dict[str, float]:
    """Return the nearest available GPS fix for an event timestamp."""
    required = {"seconds_elapsed", "location_latitude", "location_longitude"}
    if not required.issubset(recording.columns):
        return {"event_latitude": np.nan, "event_longitude": np.nan}
    data = recording.loc[:, ["seconds_elapsed", "location_latitude", "location_longitude"]].copy()
    data["seconds_elapsed"] = pd.to_numeric(data["seconds_elapsed"], errors="coerce")
    data["location_latitude"] = pd.to_numeric(data["location_latitude"], errors="coerce")
    data["location_longitude"] = pd.to_numeric(data["location_longitude"], errors="coerce")
    data = data.dropna(subset=["seconds_elapsed", "location_latitude", "location_longitude"])
    if data.empty:
        return {"event_latitude": np.nan, "event_longitude": np.nan}
    nearest = data.iloc[(data["seconds_elapsed"] - centre_seconds).abs().argmin()]
    return {"event_latitude": float(nearest["location_latitude"]), "event_longitude": float(nearest["location_longitude"])}


def main() -> None:
    events = pd.read_csv(EVENTS)
    rows: list[dict[str, object]] = []
    cache: dict[str, pd.DataFrame] = {}

    for event_index, event in events.iterrows():
        session = str(event["session"])
        sensor_path = ANOMALY_CSV / f"{session}.csv"
        if not sensor_path.exists():
            continue
        if session not in cache:
            cache[session] = pd.read_csv(sensor_path, low_memory=False)
        seconds = float(event["seconds_elapsed"])
        features = window_features(cache[session], seconds)
        row = {
            "event_id": f"{session}_{event_index:04d}",
            "session": session,
            "event_type": event["event_type"],
            "video_path": event["video_path"],
            "sensor_csv_path": str(sensor_path),
            "event_seconds": seconds,
            "window_start_seconds": max(0.0, seconds - WINDOW_SECONDS),
            "window_end_seconds": min(float(event["duration_seconds"]), seconds + WINDOW_SECONDS),
            "target_frame": int(event["target_frame"]),
            "fps": event["fps"],
            "frame_within_video": event["frame_within_video"],
            **features,
            **event_location(cache[session], seconds),
        }
        rows.append(row)

    output = pd.DataFrame(rows)
    # Terciles identify a diverse set of events for human severity annotation.
    # They must never be treated as true mild/moderate/severe labels.
    score_column = "linear_acc_rms_deviation_mps2"
    if score_column in output and output[score_column].notna().sum() >= 3:
        ranks = output[score_column].rank(method="first")
        output["impact_priority_tercile"] = pd.qcut(ranks, 3, labels=["low", "medium", "high"])
    else:
        output["impact_priority_tercile"] = pd.NA

    OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    print(f"Aligned event windows: {len(output)}")
    print(f"Sessions: {output['session'].nunique()}")
    print(f"Event types: {output['event_type'].value_counts().to_dict()}")
    print(f"Priority terciles: {output['impact_priority_tercile'].value_counts().to_dict()}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
