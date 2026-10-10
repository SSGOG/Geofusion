"""Import the downloaded OpenStreetMap road network into PostGIS."""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")
ROADS = ROOT / "data" / "processed" / "osm_roads.geojson"


def valid_coordinate(point: object) -> bool:
    return isinstance(point, list) and len(point) == 2 and all(isinstance(value, (int, float)) for value in point) and -180 <= point[0] <= 180 and -90 <= point[1] <= 90


def main() -> None:
    payload = json.loads(ROADS.read_text(encoding="utf-8"))
    rows = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        source_id = feature.get("properties", {}).get("osm_way_id")
        if geometry.get("type") != "LineString" or not source_id or len(coordinates) < 2 or not all(valid_coordinate(point) for point in coordinates):
            raise ValueError("Road GeoJSON contains an invalid LineString feature.")
        rows.append((str(source_id), json.dumps({"type": "LineString", "coordinates": coordinates})))
    if not rows:
        raise ValueError("Road GeoJSON contains no usable road segments.")

    query = """
        INSERT INTO road_segments (source_name, source_segment_id, geometry)
        VALUES ('openstreetmap', %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography)
        ON CONFLICT (source_name, source_segment_id) DO UPDATE SET geometry = EXCLUDED.geometry
    """
    with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
        cursor.executemany(query, rows)
    print(f"Imported {len(rows)} OpenStreetMap road segments.")


if __name__ == "__main__":
    main()
