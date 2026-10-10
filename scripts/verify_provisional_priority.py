"""Validate the automatic provisional-impact output for GeoFusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "processed" / "provisional_event_priority.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    data = pd.read_csv(PATH)
    usable = data["usable_imu_window"].astype(bool)
    tiers = {"low_impact", "medium_impact", "high_impact"}
    require(len(data) == 402, "all aligned events are represented")
    require(data.loc[usable, "provisional_impact_score_0_100"].between(0, 100).all(), "usable scores are bounded from 0 to 100")
    require(set(data.loc[usable, "provisional_impact_priority"].dropna()) == tiers, "all three provisional impact priorities are present")
    require((data["severity_ground_truth_available"] == False).all(), "output does not misrepresent provisional scores as ground truth")
    require(data.loc[usable, ["event_latitude", "event_longitude"]].notna().all(axis=1).mean() > 0.95, "GPS is available for more than 95% of usable events")
    print("\nAutomatic provisional-priority validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
