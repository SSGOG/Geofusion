"""Verify that a retried queued phone event creates one live map record."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gis_service import app


DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")
EVENT_ID = "phone_verify_retry_idempotency"


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    payload = app.LiveEventRequest(
        client_event_id=EVENT_ID,
        event_type="Road anomaly",
        latitude=18.6589,
        longitude=73.8119,
        image_base64="A" * 100,
        imu_sampling_hz=50,
        linear_acc_rms_deviation_mps2=1,
        linear_acc_peak_deviation_mps2=2,
        linear_acc_jerk_rms_mps3=3,
        gyro_peak_radps=1,
    )
    original_score = app.score_live_event
    try:
        with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
            cursor.execute("DELETE FROM live_events WHERE event_id = %s", (EVENT_ID,))
        app.score_live_event = lambda _: (0.9, 0.8, 0.85, "medium")
        first = app.create_live_event(payload)
        second = app.create_live_event(payload)
        with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM live_events WHERE event_id = %s", (EVENT_ID,))
            count = cursor.fetchone()[0]
        require(first["mapped"] and second["mapped"], "retried phone event remains mappable")
        require(first["event_id"] == second["event_id"] == EVENT_ID and count == 1, "retried phone event creates exactly one record")
    finally:
        app.score_live_event = original_score
        with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
            cursor.execute("DELETE FROM live_events WHERE event_id = %s", (EVENT_ID,))
            cursor.execute("SELECT COUNT(*) FROM live_events WHERE event_id = %s", (EVENT_ID,))
            require(cursor.fetchone()[0] == 0, "temporary retry test record was removed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, psycopg.Error) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
