"""Validate GeoFusion Phase 2 review and candidate-normal manifests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    queue = pd.read_csv(PROCESSED / "severity_annotation_queue.csv", keep_default_na=False)
    normal = pd.read_csv(PROCESSED / "candidate_normal_windows.csv", keep_default_na=False)
    require(len(queue) == 240, f"balanced review queue contains 240 events ({len(queue)})")
    require(queue["event_id"].is_unique, "review queue has no duplicated events")
    require(set(queue["event_type"]) == {"Bump", "Pothole"}, "both anomaly types are represented")
    require((queue["review_status"] == "pending").all(), "review statuses begin pending")
    require((queue[["reviewer_1_severity", "reviewer_2_severity", "adjudicated_severity"]] == "").all().all(), "no severity labels were fabricated")
    require(len(normal) >= 300, f"candidate normal pool is adequate ({len(normal)} windows)")
    require((normal["nearest_known_event_seconds"].astype(float) >= 3.5).all(), "candidate normals are separated from known events")
    require((normal["visual_review_status"] == "pending").all(), "candidate normals are pending visual review")
    print("\nPhase 2 manifest validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
