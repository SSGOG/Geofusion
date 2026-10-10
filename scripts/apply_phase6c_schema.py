"""Apply the additive live-event table used by the phone capture workflow."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")


def main() -> None:
    with psycopg.connect(DATABASE_URL) as database, database.cursor() as cursor:
        cursor.execute((ROOT / "sql" / "003_live_events.sql").read_text(encoding="utf-8"))
    print("Applied Phase 6C live-event schema.")


if __name__ == "__main__":
    main()
