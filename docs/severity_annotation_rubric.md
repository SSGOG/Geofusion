# GeoFusion severity annotation rubric — Phase 2

## Purpose

This rubric creates a reviewed three-level road-hazard severity label for a Bump or Pothole event. It does **not** estimate physical pothole depth. Review the 3-second road-facing video interval specified in `severity_annotation_queue.csv`; use the event type as context, but do not infer severity from the sampling group or an IMU score.

## Label definitions

| Label | Decision rule |
|---|---|
| `low` | A small, localised surface discontinuity is visible or plausible. It is unlikely to require an abrupt manoeuvre at ordinary urban speed and has limited apparent risk to a typical vehicle. |
| `medium` | The defect or bump is clearly visible and likely to cause noticeable discomfort or repeated vehicle impact. It merits scheduled maintenance but is not visibly an immediate safety hazard. |
| `high` | The defect appears extensive, deep, sharply edged, or likely to create substantial vehicle damage or safety risk. It should be marked for prompt inspection. |
| `ungradable` | The road feature cannot be judged because of blur, occlusion, night conditions, camera orientation, insufficient view, or ambiguity. Do not force a severity label. |

## Review procedure

1. Open the source MP4 at `video_path` and view from `window_start_seconds` to `window_end_seconds`.
2. Record one of `low`, `medium`, `high`, or `ungradable` in your reviewer column.
3. Record visibility as `good`, `limited`, or `poor`; briefly explain borderline cases in `review_notes`.
4. Two reviewers label independently. If their severity labels differ, a third reviewer or supervisor assigns `adjudicated_severity` and records the reason.
5. Only rows with an adjudicated `low`, `medium`, or `high` label become severity-model training data. `ungradable` rows remain useful for visual-quality analysis but are excluded from that target.

## Quality checks

- Do not use the annotation button-press duration as severity.
- Do not label by road location, session number, or prior knowledge of the data collector.
- Flag a clip rather than guessing when the road surface is not interpretable.
- Before modelling, calculate agreement between reviewers (weighted Cohen's kappa) and inspect the confusion between adjacent severity levels.
