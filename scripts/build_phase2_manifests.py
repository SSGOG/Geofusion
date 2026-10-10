r"""Build severity-review and candidate-normal manifests for GeoFusion Phase 2.

The source videos remain on E:\downloads_2\Raw Data.  This script writes only
CSV manifests.  Severity fields deliberately begin blank: reviewers label road
hazard severity from the clips using the rubric, not from an IMU threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FUSION = PROCESSED / "fusion_event_windows.csv"
VIDEO = PROCESSED / "video_annotation_manifest.csv"
QUEUE = PROCESSED / "severity_annotation_queue.csv"
NORMAL = PROCESSED / "candidate_normal_windows.csv"
RANDOM_SEED = 20260923
REVIEW_PER_STRATUM = 40
EVENT_EXCLUSION_SECONDS = 3.5
WINDOW_SECONDS = 1.5


def select_review_queue(events: pd.DataFrame) -> pd.DataFrame:
    """Select balanced, diverse candidates without exposing impact values to reviewers."""
    eligible = events.loc[events["imu_samples"] >= 5].copy()
    eligible = eligible.dropna(subset=["impact_priority_tercile"])
    chosen = []
    for (_, _), group in eligible.groupby(["event_type", "impact_priority_tercile"], observed=True):
        group = group.sample(frac=1, random_state=RANDOM_SEED)
        # Prefer session diversity: take one event per session before repeats.
        rounds: list[pd.DataFrame] = []
        for _, session_rows in group.groupby("session", sort=False):
            rounds.append(session_rows.reset_index(drop=True))
        picks = []
        depth = 0
        while len(picks) < REVIEW_PER_STRATUM:
            added = False
            for session_rows in rounds:
                if depth < len(session_rows):
                    picks.append(session_rows.iloc[depth])
                    added = True
                    if len(picks) == REVIEW_PER_STRATUM:
                        break
            if not added:
                break
            depth += 1
        chosen.extend(picks)

    queue = pd.DataFrame(chosen).copy()
    queue = queue.sort_values(["event_type", "session", "event_seconds"]).reset_index(drop=True)
    # Keep only information needed by a reviewer. Sampling stratum and IMU impact
    # values stay out of the file to prevent anchoring on an automated score.
    queue = queue[
        ["event_id", "session", "event_type", "video_path", "event_seconds", "window_start_seconds", "window_end_seconds", "target_frame"]
    ]
    queue["review_status"] = "pending"
    queue["reviewer_1_severity"] = ""
    queue["reviewer_2_severity"] = ""
    queue["adjudicated_severity"] = ""
    queue["visibility_quality"] = ""
    queue["review_notes"] = ""
    return queue


def candidate_centres(duration: float, event_times: np.ndarray) -> list[float]:
    # A 1.5-second stride gives a sufficiently broad review pool. Training later
    # uses a non-overlapping subset, selected only after visual rejection.
    centres = np.arange(WINDOW_SECONDS, duration - WINDOW_SECONDS + 0.001, 1.5)
    return [float(value) for value in centres if np.all(np.abs(event_times - value) >= EVENT_EXCLUSION_SECONDS)]


def build_candidate_normals(video: pd.DataFrame) -> pd.DataFrame:
    """Sample video windows separated from all known annotated events.

    These are candidate normal windows: unannotated does not prove that a surface
    is pristine, so they require a lightweight visual rejection pass before use
    as final negatives.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for session, group in video.groupby("session", sort=True):
        duration = float(group["duration_seconds"].iloc[0])
        event_times = group["seconds_elapsed"].to_numpy(dtype=float)
        centres = candidate_centres(duration, event_times)
        rng.shuffle(centres)
        for index, centre in enumerate(centres[:8]):
            rows.append(
                {
                    "normal_window_id": f"normal_{session}_{index:02d}",
                    "session": session,
                    "video_path": group["video_path"].iloc[0],
                    "sensor_csv_path": group["sensor_csv_path"].iloc[0] if "sensor_csv_path" in group else str(PROCESSED / "anomaly_csv" / f"{session}.csv"),
                    "window_centre_seconds": centre,
                    "window_start_seconds": max(0.0, centre - WINDOW_SECONDS),
                    "window_end_seconds": min(duration, centre + WINDOW_SECONDS),
                    "nearest_known_event_seconds": float(np.min(np.abs(event_times - centre))),
                    "candidate_label": "candidate_normal",
                    "visual_review_status": "pending",
                    "visual_reviewer_notes": "",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    events = pd.read_csv(FUSION)
    video = pd.read_csv(VIDEO)
    queue = select_review_queue(events)
    normals = build_candidate_normals(video)
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE, index=False)
    normals.to_csv(NORMAL, index=False)
    print(f"Severity review queue: {len(queue)} events")
    print(f"  By type: {queue['event_type'].value_counts().to_dict()}")
    print(f"  Sessions represented: {queue['session'].nunique()}")
    print(f"Candidate normal windows: {len(normals)}")
    print(f"  Sessions represented: {normals['session'].nunique()}")
    print(f"  Minimum distance from known event: {normals['nearest_known_event_seconds'].min():.2f} seconds")


if __name__ == "__main__":
    main()
