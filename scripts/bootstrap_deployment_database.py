"""Initialize or refresh the GeoFusion deployment database from versioned project data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, env={**os.environ, "GEOFUSION_DATABASE_URL": DATABASE_URL})


def main() -> None:
    with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
        cursor.execute((ROOT / "sql" / "001_schema.sql").read_text(encoding="utf-8"))
    run("import_gis_events.py")
    run("import_road_network.py")
    run("apply_phase5b_schema.py")
    run("apply_phase6c_schema.py")
    print("GeoFusion deployment database is ready.")


if __name__ == "__main__":
    main()
