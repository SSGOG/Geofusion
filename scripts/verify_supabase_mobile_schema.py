"""Check the required access-control and spatial clauses are present in the Supabase migration."""

from __future__ import annotations

import sys
from pathlib import Path


SQL = Path(__file__).resolve().parents[1] / "sql" / "004_supabase_mobile_events.sql"


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    text = SQL.read_text(encoding="utf-8").lower()
    require("extensions.geography(point, 4326)" in text, "Supabase migration stores a PostGIS point")
    require("enable row level security" in text, "Supabase migration enables row-level security")
    require("to authenticated" in text and "reporter_id = auth.uid()" in text, "Supabase policies restrict writes to the reporting device")
    require("status in ('active', 'dismissed')" in text, "Supabase migration preserves false-positive dismissal state")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
