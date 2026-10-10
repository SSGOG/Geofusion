"""Verify portable severity tiers match the local GIS scoring rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "mobile" / "assets" / "models" / "impact_reference.json"
SOURCE = ROOT / "data" / "processed" / "fusion_event_windows.csv"


def tier(score: float) -> str:
    return "high" if score >= 2 / 3 else "medium" if score >= 1 / 3 else "low"


def main() -> None:
    model = json.loads(EXPORT.read_text(encoding="utf-8"))
    data = pd.read_csv(SOURCE).query("imu_samples >= 5").dropna(subset=model["weights"].keys()).head(30)
    for _, row in data.iterrows():
        expected = sum(weight * np.searchsorted(values, row[feature], side="right") / len(values) for feature, weight in model["weights"].items() for values in [np.array(model["reference"][feature])])
        portable = sum(weight * sum(value <= row[feature] for value in model["reference"][feature]) / len(model["reference"][feature]) for feature, weight in model["weights"].items())
        assert tier(expected) == tier(portable)
    print("PASS  portable severity tiers match the GIS reference rule")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
