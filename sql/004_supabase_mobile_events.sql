-- Run once in the Supabase SQL Editor after enabling PostGIS in the extensions schema.

CREATE TABLE IF NOT EXISTS public.mobile_road_events (
    event_id TEXT PRIMARY KEY CHECK (event_id ~ '^phone_[A-Za-z0-9_-]{1,100}$'),
    reporter_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id),
    event_type TEXT NOT NULL CHECK (event_type = 'Road anomaly'),
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    location extensions.geography(POINT, 4326) GENERATED ALWAYS AS (
        extensions.st_setsrid(extensions.st_makepoint(longitude, latitude), 4326)::extensions.geography
    ) STORED,
    vision_probability DOUBLE PRECISION NOT NULL CHECK (vision_probability BETWEEN 0 AND 1),
    imu_probability DOUBLE PRECISION NOT NULL CHECK (imu_probability BETWEEN 0 AND 1),
    fused_probability DOUBLE PRECISION NOT NULL CHECK (fused_probability BETWEEN 0 AND 1),
    impact_priority TEXT NOT NULL CHECK (impact_priority IN ('low', 'medium', 'high')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dismissed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS mobile_road_events_location_gix ON public.mobile_road_events USING GIST (location);
CREATE INDEX IF NOT EXISTS mobile_road_events_active_idx ON public.mobile_road_events (status, created_at DESC);

ALTER TABLE public.mobile_road_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "authenticated users read active road events" ON public.mobile_road_events;
CREATE POLICY "authenticated users read active road events"
ON public.mobile_road_events FOR SELECT TO authenticated
USING (status = 'active' OR reporter_id = auth.uid());

DROP POLICY IF EXISTS "authenticated users create their own road events" ON public.mobile_road_events;
CREATE POLICY "authenticated users create their own road events"
ON public.mobile_road_events FOR INSERT TO authenticated
WITH CHECK (reporter_id = auth.uid());

DROP POLICY IF EXISTS "authenticated users dismiss their own road events" ON public.mobile_road_events;
CREATE POLICY "authenticated users dismiss their own road events"
ON public.mobile_road_events FOR UPDATE TO authenticated
USING (reporter_id = auth.uid())
WITH CHECK (reporter_id = auth.uid() AND status IN ('active', 'dismissed'));
