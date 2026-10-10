CREATE OR REPLACE VIEW road_segment_priority AS
WITH matched_events AS (
    SELECT
        event.event_id,
        event.event_type,
        event.fused_probability,
        event.impact_priority,
        segment.segment_id,
        ST_Distance(segment.geometry, event.location) AS match_distance_m
    FROM road_events AS event
    CROSS JOIN LATERAL (
        SELECT segment_id, geometry
        FROM road_segments
        WHERE ST_DWithin(geometry, event.location, 30)
        ORDER BY ST_Distance(geometry, event.location)
        LIMIT 1
    ) AS segment
), aggregated AS (
    SELECT
        segment_id,
        COUNT(*) AS event_count,
        COUNT(*) FILTER (WHERE event_type = 'Pothole') AS pothole_count,
        COUNT(*) FILTER (WHERE event_type = 'Bump') AS bump_count,
        COUNT(fused_probability) AS scored_event_count,
        AVG(fused_probability) AS mean_fused_probability,
        MAX(match_distance_m) AS max_match_distance_m,
        SUM(CASE impact_priority WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END * COALESCE(fused_probability, 0.5)) AS provisional_maintenance_score
    FROM matched_events
    GROUP BY segment_id
)
SELECT
    segment.segment_id,
    segment.source_name,
    segment.source_segment_id,
    segment.geometry,
    aggregated.event_count,
    aggregated.pothole_count,
    aggregated.bump_count,
    aggregated.scored_event_count,
    aggregated.mean_fused_probability,
    aggregated.max_match_distance_m,
    aggregated.provisional_maintenance_score,
    CASE
        WHEN aggregated.provisional_maintenance_score >= 6 THEN 'high'
        WHEN aggregated.provisional_maintenance_score >= 3 THEN 'medium'
        ELSE 'low'
    END AS maintenance_priority
FROM aggregated
JOIN road_segments AS segment USING (segment_id);
