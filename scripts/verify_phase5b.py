"""Validate OpenStreetMap import and event-to-road aggregation."""

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
        cursor.execute("SELECT COUNT(*) FROM road_segments WHERE source_name = 'openstreetmap'")
        road_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(event_count), 0), MAX(max_match_distance_m) FROM road_segment_priority")
        segments, matched_events, farthest_match = cursor.fetchone()
    require(road_count > 0, "OpenStreetMap roads were imported")
    require(segments > 0 and matched_events > 0, "GPS events were matched to nearby road segments")
    require(farthest_match <= 30, "all road matches are within the 30 m tolerance")
    print("\nPhase 5B road matching validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, psycopg.Error) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
