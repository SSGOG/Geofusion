"""Read-only GeoJSON API for the GeoFusion PostGIS event store."""

from __future__ import annotations

import os
from base64 import b64decode
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
from json import loads
from math import asinh, floor, pi, radians, tan
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

import cv2
import joblib
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException, Query, Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from psycopg.rows import dict_row


DATABASE_URL = os.getenv("GEOFUSION_DATABASE_URL", "postgresql://geofusion:geofusion_dev_password@localhost:5433/geofusion")
ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="GeoFusion GIS API", version="0.1.0")
TILE_SIZE = 256
MAP_ZOOM = 12
IMU_FEATURES = ["imu_sampling_hz", "linear_acc_rms_deviation_mps2", "linear_acc_peak_deviation_mps2", "linear_acc_jerk_rms_mps3", "gyro_peak_radps"]
IMPACT_WEIGHTS = {"linear_acc_rms_deviation_mps2": 0.40, "linear_acc_peak_deviation_mps2": 0.30, "linear_acc_jerk_rms_mps3": 0.20, "gyro_peak_radps": 0.10}
MODEL_REFERENCE_SAMPLING_HZ = 100.0


class LiveEventRequest(BaseModel):
    client_event_id: str | None = Field(default=None, pattern=r"^phone_[A-Za-z0-9_-]{1,100}$")
    event_type: str = Field(pattern="^(Bump|Pothole|Road anomaly)$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    image_base64: str = Field(min_length=100, max_length=8_000_000)
    imu_sampling_hz: float = Field(gt=0, le=200)
    linear_acc_rms_deviation_mps2: float = Field(ge=0, le=100)
    linear_acc_peak_deviation_mps2: float = Field(ge=0, le=100)
    linear_acc_jerk_rms_mps3: float = Field(ge=0, le=5000)
    gyro_peak_radps: float = Field(ge=0, le=100)


@lru_cache(maxsize=1)
def live_models():
    from ultralytics import YOLO

    yolo = YOLO(str(ROOT / "models" / "runs" / "geofusion_yolo_cls_final" / "weights" / "best.pt"))
    imu = joblib.load(ROOT / "models" / "phase4d_final_imu_pipeline.joblib")["pipeline"]
    return yolo, imu


@lru_cache(maxsize=1)
def impact_reference() -> dict[str, np.ndarray]:
    data = pd.read_csv(ROOT / "data" / "processed" / "fusion_event_windows.csv")
    usable = data.loc[data["imu_samples"] >= 5]
    return {feature: np.sort(usable[feature].dropna().to_numpy(dtype=float)) for feature in IMPACT_WEIGHTS}


def relative_impact_tier(values: dict[str, float]) -> str:
    reference = impact_reference()
    score = sum(weight * np.searchsorted(reference[feature], values[feature], side="right") / len(reference[feature]) for feature, weight in IMPACT_WEIGHTS.items())
    return "high" if score >= 2 / 3 else "medium" if score >= 1 / 3 else "low"


def existing_live_event(event_id: str) -> dict[str, object] | None:
    query = """
        SELECT event_id, event_type, impact_priority, yolo_probability, imu_probability, fused_probability, status
        FROM live_events
        WHERE event_id = %s
    """
    with connection() as database, database.cursor() as cursor:
        cursor.execute(query, (event_id,))
        row = cursor.fetchone()
    if row is None:
        return None
    if row["status"] == "dismissed":
        return {"mapped": False, "event_id": event_id, "fused_probability": row["fused_probability"], "reason": "This phone event was previously dismissed."}
    return {"mapped": True, "event_id": event_id, "event_type": row["event_type"], "impact_priority": row["impact_priority"], "yolo_probability": row["yolo_probability"], "imu_probability": row["imu_probability"], "fused_probability": row["fused_probability"]}


def score_live_event(payload: LiveEventRequest) -> tuple[float, float, float, str]:
    try:
        image_bytes = b64decode(payload.image_base64.split(",")[-1], validate=True)
    except ValueError as error:
        raise ValueError("Camera image is not valid base64 data") from error
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Camera image cannot be decoded")
    yolo, imu = live_models()
    result = yolo.predict(image, imgsz=224, device="cpu", verbose=False)[0]
    event_index = next(index for index, name in result.names.items() if name == "annotated_event")
    yolo_probability = float(result.probs.data[event_index].cpu())
    feature_values = payload.model_dump(include=set(IMU_FEATURES))
    # shortcut: training sampling frequency is effectively constant at 100 Hz; calibrate before phone-specific retraining.
    model_feature_values = {**feature_values, "imu_sampling_hz": MODEL_REFERENCE_SAMPLING_HZ}
    imu_probability = float(imu.predict_proba(pd.DataFrame([model_feature_values]))[:, 1][0])
    fused_probability = (yolo_probability + imu_probability) / 2
    return yolo_probability, imu_probability, fused_probability, relative_impact_tier(feature_values)


@lru_cache(maxsize=256)
def osm_tile(z: int, x: int, y: int) -> bytes:
    if not 0 <= z <= 19 or not 0 <= x < 2 ** z or not 0 <= y < 2 ** z:
        raise ValueError("Invalid OpenStreetMap tile coordinate")
    request = Request(
        f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        headers={"User-Agent": "GeoFusion research prototype"},
    )
    with urlopen(request, timeout=15) as source:
        return source.read()


def world_pixel(longitude: float, latitude: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * 2 ** zoom
    x = (longitude + 180) / 360 * scale
    y = (1 - asinh(tan(radians(latitude))) / pi) / 2 * scale
    return x, y


def map_png(latitude: float | None = None, longitude: float | None = None) -> bytes:
    zoom = 14 if latitude is not None else MAP_ZOOM
    bounds_query = """
        SELECT MIN(ST_X(location::geometry)) AS west, MIN(ST_Y(location::geometry)) AS south,
               MAX(ST_X(location::geometry)) AS east, MAX(ST_Y(location::geometry)) AS north
        FROM road_events
    """
    segment_query = "SELECT maintenance_priority, ST_AsGeoJSON(geometry::geometry) AS geometry FROM road_segment_priority"
    event_query = """
        SELECT ST_X(location::geometry) AS longitude, ST_Y(location::geometry) AS latitude,
               CASE WHEN impact_priority = 'unknown' THEN 'sensor_data_unavailable' ELSE impact_priority END AS impact_priority
        FROM road_events
        UNION ALL
        SELECT ST_X(location::geometry) AS longitude, ST_Y(location::geometry) AS latitude, impact_priority
        FROM live_events
        WHERE status = 'active'
    """
    with connection() as database, database.cursor() as cursor:
        if latitude is None:
            cursor.execute(bounds_query)
            bounds = cursor.fetchone()
            west, south, east, north = bounds["west"], bounds["south"], bounds["east"], bounds["north"]
            if None in (west, south, east, north):
                raise ValueError("No GeoFusion event locations are available")
        else:
            west, south, east, north = longitude - 0.025, latitude - 0.018, longitude + 0.025, latitude + 0.018
        cursor.execute(segment_query)
        segments = cursor.fetchall()
        cursor.execute(event_query)
        events = cursor.fetchall()

    padding = 0.015
    left, bottom = world_pixel(float(west) - padding, float(south) - padding, zoom)
    right, top = world_pixel(float(east) + padding, float(north) + padding, zoom)
    tile_left, tile_top = floor(left / TILE_SIZE), floor(top / TILE_SIZE)
    tile_right, tile_bottom = floor(right / TILE_SIZE), floor(bottom / TILE_SIZE)
    width, height = (tile_right - tile_left + 1) * TILE_SIZE, (tile_bottom - tile_top + 1) * TILE_SIZE
    image = Image.new("RGB", (width, height), "white")
    for tile_x in range(tile_left, tile_right + 1):
        for tile_y in range(tile_top, tile_bottom + 1):
            tile = Image.open(BytesIO(osm_tile(zoom, tile_x, tile_y))).convert("RGB")
            image.paste(tile, ((tile_x - tile_left) * TILE_SIZE, (tile_y - tile_top) * TILE_SIZE))

    draw = ImageDraw.Draw(image)
    segment_colours = {"high": "#d32f2f", "medium": "#f57c00", "low": "#388e3c"}
    event_colours = {"high": "#b71c1c", "medium": "#ef6c00", "low": "#2e7d32", "sensor_data_unavailable": "#616161"}

    def canvas_point(longitude: float, latitude: float) -> tuple[int, int]:
        x, y = world_pixel(longitude, latitude, zoom)
        return round(x - tile_left * TILE_SIZE), round(y - tile_top * TILE_SIZE)

    for segment in segments:
        coordinates = loads(segment["geometry"])["coordinates"]
        draw.line([canvas_point(longitude, latitude) for longitude, latitude in coordinates], fill=segment_colours[segment["maintenance_priority"]], width=4)
    for event in events:
        x, y = canvas_point(event["longitude"], event["latitude"])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=event_colours[event["impact_priority"]], outline="white", width=1)
    if latitude is not None:
        x, y = canvas_point(longitude, latitude)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#1565c0", outline="white", width=2)

    crop = image.crop((round(left - tile_left * TILE_SIZE), round(top - tile_top * TILE_SIZE), round(right - tile_left * TILE_SIZE), round(bottom - tile_top * TILE_SIZE)))
    output = BytesIO()
    crop.save(output, format="PNG", optimize=True)
    return output.getvalue()


@contextmanager
def connection():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as database:
        yield database


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with connection() as database, database.cursor() as cursor:
            cursor.execute("SELECT PostGIS_Version() AS version")
            return {"status": "ok", "postgis_version": cursor.fetchone()["version"]}
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="PostGIS is unavailable") from error


@app.get("/osm-tiles/{z}/{x}/{y}.png", response_class=Response)
def osm_tiles(z: int, x: int, y: int) -> Response:
    try:
        return Response(content=osm_tile(z, x, y), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=502, detail="OpenStreetMap tile is unavailable") from error


@app.get("/map.png", response_class=Response)
def map_snapshot(
    latitude: float | None = Query(None, ge=-90, le=90),
    longitude: float | None = Query(None, ge=-180, le=180),
) -> Response:
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=422, detail="latitude and longitude must be supplied together")
    try:
        return Response(content=map_png(latitude, longitude), media_type="image/png", headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, psycopg.Error) as error:
        raise HTTPException(status_code=503, detail="GeoFusion map snapshot is unavailable") from error


@app.get("/events.geojson")
def events_geojson(
    minimum_confidence: float = Query(0, ge=0, le=1),
    limit: int = Query(500, ge=1, le=2000),
) -> dict[str, object]:
    query = """
        SELECT event_id, session_id, event_type, event_seconds, source_label,
               yolo_probability, imu_probability, fused_probability,
               CASE WHEN impact_priority = 'unknown' THEN 'sensor_data_unavailable' ELSE impact_priority END AS impact_priority,
               ST_AsGeoJSON(location::geometry)::jsonb AS geometry
        FROM road_events
        WHERE COALESCE(fused_probability, 0) >= %s
        ORDER BY fused_probability DESC NULLS LAST, event_id
        LIMIT %s
    """
    try:
        with connection() as database, database.cursor() as cursor:
            cursor.execute(query, (minimum_confidence, limit))
            rows = cursor.fetchall()
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="PostGIS is unavailable") from error
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": row.pop("event_id"), "geometry": row.pop("geometry"), "properties": row} for row in rows],
    }


@app.get("/events/summary")
def event_summary() -> list[dict[str, object]]:
    query = """
        SELECT event_type, impact_priority, COUNT(*) AS event_count,
               COUNT(fused_probability) AS scored_event_count,
               AVG(fused_probability) AS mean_fused_probability
        FROM road_events
        GROUP BY event_type, impact_priority
        ORDER BY event_type, impact_priority
    """
    try:
        with connection() as database, database.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="PostGIS is unavailable") from error


@app.get("/segments.geojson")
def segments_geojson(limit: int = Query(500, ge=1, le=2000)) -> dict[str, object]:
    query = """
        SELECT segment_id, source_name, source_segment_id, event_count, pothole_count,
               bump_count, scored_event_count, mean_fused_probability,
               max_match_distance_m, provisional_maintenance_score, maintenance_priority,
               ST_AsGeoJSON(geometry::geometry)::jsonb AS geometry
        FROM road_segment_priority
        ORDER BY provisional_maintenance_score DESC, segment_id
        LIMIT %s
    """
    try:
        with connection() as database, database.cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="PostGIS road aggregation is unavailable") from error
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": row.pop("segment_id"), "geometry": row.pop("geometry"), "properties": row} for row in rows],
    }


@app.post("/live-events")
def create_live_event(payload: LiveEventRequest) -> dict[str, object]:
    try:
        if payload.client_event_id:
            existing = existing_live_event(payload.client_event_id)
            if existing is not None:
                return existing
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Live event storage is unavailable") from error
    try:
        yolo_probability, imu_probability, fused_probability, impact_priority = score_live_event(payload)
    except (ValueError, StopIteration) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail="Live model inference is unavailable") from error
    if fused_probability < 0.5:
        return {"mapped": False, "fused_probability": fused_probability, "reason": "The camera and sensor fusion score did not reach the candidate threshold."}
    event_id = payload.client_event_id or f"live_{uuid4().hex}"
    query = """
        INSERT INTO live_events (event_id, event_type, yolo_probability, imu_probability, fused_probability, impact_priority, location)
        VALUES (%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)
        ON CONFLICT (event_id) DO NOTHING
        RETURNING event_id
    """
    try:
        with connection() as database, database.cursor() as cursor:
            cursor.execute(query, (event_id, payload.event_type, yolo_probability, imu_probability, fused_probability, impact_priority, payload.longitude, payload.latitude))
            inserted = cursor.fetchone()
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Live event storage is unavailable") from error
    if inserted is None:
        return existing_live_event(event_id) or {"mapped": False, "fused_probability": fused_probability, "reason": "The event could not be saved."}
    return {"mapped": True, "event_id": event_id, "event_type": payload.event_type, "impact_priority": impact_priority, "yolo_probability": yolo_probability, "imu_probability": imu_probability, "fused_probability": fused_probability}


@app.delete("/live-events/{event_id}")
def dismiss_live_event(event_id: str) -> dict[str, str]:
    query = "UPDATE live_events SET status = 'dismissed', dismissed_at = now() WHERE event_id = %s AND status = 'active' RETURNING event_id"
    try:
        with connection() as database, database.cursor() as cursor:
            cursor.execute(query, (event_id,))
            row = cursor.fetchone()
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Live event storage is unavailable") from error
    if row is None:
        raise HTTPException(status_code=404, detail="Active live event was not found")
    return {"event_id": row["event_id"], "status": "dismissed"}
