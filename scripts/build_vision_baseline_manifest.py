"""Create aligned frame references for the Phase 4 vision baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "vision_baseline_manifest.csv"


def main() -> None:
    baseline = pd.read_csv(PROCESSED / "imu_baseline_dataset.csv")
    events = pd.read_csv(PROCESSED / "fusion_event_windows.csv")
    normal = pd.read_csv(PROCESSED / "candidate_normal_windows.csv")

    positives = baseline.loc[baseline["target_anomaly"] == 1, ["event_id", "session", "target_anomaly", "split"]].merge(
        events[["event_id", "video_path", "event_seconds", "fps", "target_frame"]], on="event_id", how="inner", validate="one_to_one"
    )
    negatives = baseline.loc[baseline["target_anomaly"] == 0, ["event_id", "session", "target_anomaly", "split"]].merge(
        normal[["normal_window_id", "video_path", "window_centre_seconds"]], left_on="event_id", right_on="normal_window_id", how="inner", validate="one_to_one"
    )
    negatives = negatives.rename(columns={"window_centre_seconds": "event_seconds"})
    negatives["fps"] = pd.NA
    negatives["target_frame"] = pd.NA
    negatives = negatives.drop(columns="normal_window_id")
    output = pd.concat([positives, negatives], ignore_index=True).sort_values("event_id")
    output["frame_source"] = "annotated_event" 
    output.loc[output["target_anomaly"] == 0, "frame_source"] = "candidate_normal"
    output.to_csv(OUTPUT, index=False)
    print(f"Vision rows: {len(output)}")
    print(f"Rows by split: {output['split'].value_counts().to_dict()}")
    print(f"Missing video references: {int(output['video_path'].isna().sum())}")


if __name__ == "__main__":
    main()
