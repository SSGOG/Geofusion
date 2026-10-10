"""Download drivable OpenStreetMap roads around the GeoFusion event extent."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "fusion_event_windows.csv"
OUTPUT = ROOT / "data" / "processed" / "osm_roads.geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MARGIN_DEGREES = 0.003


def event_bbox(events: pd.DataFrame) -> tuple[float, float, float, float]:
    latitude = pd.to_numeric(events["event_latitude"], errors="raise")
    longitude = pd.to_numeric(events["event_longitude"], errors="raise")
    return (
        float(latitude.min() - MARGIN_DEGREES),
        float(longitude.min() - MARGIN_DEGREES),
        float(latitude.max() + MARGIN_DEGREES),
        float(longitude.max() + MARGIN_DEGREES),
    )


def main() -> None:
    events = pd.read_csv(EVENTS)
    south, west, north, east = event_bbox(events)
    query = f'''[out:json][timeout:120];
way[highway~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)$"]({south},{west},{north},{east});
out geom;'''
    url = f"{OVERPASS_URL}?{urlencode({'data': query})}"
    try:
        request = Request(url, headers={"User-Agent": "GeoFusion research project contact: local-user"})
        with urlopen(request, timeout=180) as response:
            payload = json.load(response)
    except OSError as error:
        raise RuntimeError("OpenStreetMap road download failed. Check your internet connection and try again.") from error

    features = []
    for way in payload.get("elements", []):
        geometry = way.get("geometry", [])
        if len(geometry) < 2:
            continue
        coordinates = [[point["lon"], point["lat"]] for point in geometry]
        features.append({
            "type": "Feature",
            "properties": {"osm_way_id": str(way["id"]), **way.get("tags", {})},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        })

    if not features:
        raise RuntimeError("OpenStreetMap returned no drivable road segments for the recording area.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    print(f"Saved {len(features)} OpenStreetMap road ways to {OUTPUT}")


if __name__ == "__main__":
    main()
