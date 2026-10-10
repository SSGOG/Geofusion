CREATE TABLE IF NOT EXISTS live_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (event_type IN ('Bump', 'Pothole', 'Road anomaly')),
    yolo_probability DOUBLE PRECISION NOT NULL CHECK (yolo_probability BETWEEN 0 AND 1),
    imu_probability DOUBLE PRECISION NOT NULL CHECK (imu_probability BETWEEN 0 AND 1),
    fused_probability DOUBLE PRECISION NOT NULL CHECK (fused_probability BETWEEN 0 AND 1),
    impact_priority TEXT NOT NULL CHECK (impact_priority IN ('low', 'medium', 'high')),
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dismissed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS live_events_location_gix ON live_events USING GIST (location);
CREATE INDEX IF NOT EXISTS live_events_status_idx ON live_events (status);

ALTER TABLE live_events DROP CONSTRAINT IF EXISTS live_events_event_type_check;
ALTER TABLE live_events ADD CONSTRAINT live_events_event_type_check
    CHECK (event_type IN ('Bump', 'Pothole', 'Road anomaly'));
