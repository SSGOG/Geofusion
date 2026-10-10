CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS road_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('Bump', 'Pothole')),
    event_seconds DOUBLE PRECISION NOT NULL CHECK (event_seconds >= 0),
    source_label TEXT NOT NULL,
    yolo_probability DOUBLE PRECISION CHECK (yolo_probability BETWEEN 0 AND 1),
    imu_probability DOUBLE PRECISION CHECK (imu_probability BETWEEN 0 AND 1),
    fused_probability DOUBLE PRECISION CHECK (fused_probability BETWEEN 0 AND 1),
    impact_priority TEXT NOT NULL DEFAULT 'unknown' CHECK (impact_priority IN ('low', 'medium', 'high', 'unknown')),
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS road_events_location_gix ON road_events USING GIST (location);
CREATE INDEX IF NOT EXISTS road_events_type_idx ON road_events (event_type);

CREATE TABLE IF NOT EXISTS road_segments (
    segment_id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_segment_id TEXT NOT NULL,
    geometry GEOGRAPHY(LINESTRING, 4326) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_name, source_segment_id)
);

CREATE INDEX IF NOT EXISTS road_segments_geometry_gix ON road_segments USING GIST (geometry);
