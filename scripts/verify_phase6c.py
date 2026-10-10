"""Validate the live-event persistence table without creating an event."""

from __future__ import annotations

import os
import sys

import psycopg


DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
        cursor.execute("SELECT to_regclass('public.live_events')")
        table = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM live_events WHERE status NOT IN ('active', 'dismissed')")
        invalid_statuses = cursor.fetchone()[0]
    require(table == "live_events", "live-event table exists")
    require(invalid_statuses == 0, "stored live events have a valid review status")
    print("\nPhase 6C live-event storage validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, psycopg.Error) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
