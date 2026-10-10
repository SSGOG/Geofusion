"""Verify the imported Phase 5A GeoFusion PostGIS event store."""

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
        cursor.execute("SELECT COUNT(*), COUNT(fused_probability), COUNT(*) FILTER (WHERE ST_SRID(location::geometry) = 4326) FROM road_events")
        events, scored, wgs84 = cursor.fetchone()
        cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'road_events_location_gix')")
        has_spatial_index = cursor.fetchone()[0]
    require(events == 402, "all 402 GPS-aligned events are stored")
    require(scored == 45, "45 GPS-annotated held-out events retain model probabilities")
    require(wgs84 == 402, "all event geometries use WGS84")
    require(has_spatial_index, "event geometry has a GiST spatial index")
    print("\nPhase 5A GIS event-store validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, psycopg.Error) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
