"""Export the GIS severity reference distribution for on-device late fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "fusion_event_windows.csv"
DESTINATION = ROOT / "mobile" / "assets" / "models" / "impact_reference.json"
WEIGHTS = {
    "linear_acc_rms_deviation_mps2": 0.40,
    "linear_acc_peak_deviation_mps2": 0.30,
    "linear_acc_jerk_rms_mps3": 0.20,
    "gyro_peak_radps": 0.10,
}


def main() -> None:
    data = pd.read_csv(SOURCE)
    usable = data.loc[data["imu_samples"] >= 5]
    payload = {
        "schema_version": 1,
        "weights": WEIGHTS,
        "reference": {feature: np.sort(usable[feature].dropna().to_numpy(dtype=float)).tolist() for feature in WEIGHTS},
    }
    DESTINATION.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported mobile severity reference to {DESTINATION}")


if __name__ == "__main__":
    main()
