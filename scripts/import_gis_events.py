"""Import GPS-aligned GeoFusion events and available model confidences into PostGIS."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")
EVENTS = ROOT / "data" / "processed" / "fusion_event_windows.csv"
PREDICTIONS = ROOT / "data" / "processed" / "phase4d_final_late_yolo_imu_test_predictions.csv"


def optional_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def main() -> None:
    events = pd.read_csv(EVENTS)
    predictions = pd.read_csv(PREDICTIONS)[["event_id", "yolo_event_probability", "imu_event_probability", "late_yolo_imu_probability"]]
    events["event_id"] = events["event_id"].astype(str)
    predictions["event_id"] = predictions["event_id"].astype(str)
    events = events.merge(predictions, on="event_id", how="left", validate="one_to_one")
    rows = [
        (
            row.event_id, str(row.session), row.event_type, float(row.event_seconds), "annotated_bump_or_pothole",
            optional_float(row.yolo_event_probability), optional_float(row.imu_event_probability), optional_float(row.late_yolo_imu_probability),
            row.impact_priority_tercile if pd.notna(row.impact_priority_tercile) else "unknown",
            float(row.event_longitude), float(row.event_latitude),
        )
        for row in events.itertuples(index=False)
    ]
    query = """
        INSERT INTO road_events (
            event_id, session_id, event_type, event_seconds, source_label,
            yolo_probability, imu_probability, fused_probability, impact_priority, location
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)
        ON CONFLICT (event_id) DO UPDATE SET
            yolo_probability = EXCLUDED.yolo_probability,
            imu_probability = EXCLUDED.imu_probability,
            fused_probability = EXCLUDED.fused_probability,
            impact_priority = EXCLUDED.impact_priority,
            location = EXCLUDED.location
    """
    with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
        cursor.executemany(query, rows)
    print(f"Imported {len(rows)} GPS-aligned events; {events['late_yolo_imu_probability'].notna().sum()} have final model probabilities.")


if __name__ == "__main__":
    main()
