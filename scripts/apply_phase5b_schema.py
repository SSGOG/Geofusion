"""Apply the additive Phase 5B road-aggregation view."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")
SQL = ROOT / "sql" / "002_road_aggregation.sql"


def main() -> None:
    with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
        cursor.execute(SQL.read_text(encoding="utf-8"))
    print("Applied Phase 5B road aggregation view.")


if __name__ == "__main__":
    main()
