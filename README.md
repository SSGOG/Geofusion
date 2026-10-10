# GeoFusion

GeoFusion is a mobile road-condition intelligence project. It combines road-facing video, phone inertial sensors and GPS/GIS context to detect road anomalies, estimate reviewed severity, and prioritise road segments on a map.

## Delivery phases

| Phase | Outcome | Status |
|---|---|---|
| 1. Data contract | Reproducible EDA, video/event index, RGB--IMU windows and validation checks | Complete |
| 2. Provisional priority | Automatic sensor-impact priority, GPS event locations and candidate-normal windows | Complete |
| 3. IMU baseline | Session-level split and IMU-only annotated-event vs candidate-normal model | Complete |
| 4A. Vision baseline | Aligned-frame HOG vision benchmark under the frozen session split | Complete |
| 4B. Fusion comparison | Early and late RGB--IMU fusion compared on validation sessions | Complete |
| 4C1. YOLO visual baseline | Fine-tuned YOLO11n classification on aligned frames | Complete |
| 4C2. Clip encoder + renewed fusion | Compact temporal visual encoder; enhanced fusion comparison | Complete |
| 4D. Final fusion evaluation | Refit selected method on train + validation and evaluate test once | Complete |
| 5A. GIS event store | PostGIS event database, GeoJSON API and GPS event import | Complete |
| 5B. Road matching and aggregation | OpenStreetMap road-network import, 30 m event matching and provisional segment priority | Complete |
| 6A. Mobile map client | React Native map, GeoJSON events/segments and foreground phone location | Ready for device test |
| 6B. Mobile sensing | Live phone accelerometer and gyroscope capture with baseline-aligned window features | Ready for device test |
| 6C. Mobile inference | Camera pipeline, calibrated fusion inference and live map event submission | Ready for controlled device test |
| 7C. Cloud mobile events | Anonymous Supabase sync, offline retry queue and keyless Leaflet/OpenStreetMap rendering | Ready for configured account/device test |

## Phase 1: run and verify

The raw videos remain at `E:\downloads_2\Raw Data`; the scripts only store paths and derived outputs in this workspace. Keep that drive connected while rebuilding video-derived files.

Open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_eda.py
python scripts/build_video_annotation_manifest.py
python scripts/build_fusion_event_windows.py
python scripts/verify_phase1.py
```

Expected final line:

```text
Phase 1 validation completed successfully.
```

The validation deliberately reports, rather than hides, events with insufficient sensor samples. These must be excluded from training but remain in the audit manifest.

## Important data rule

`impact_priority_tercile` and `provisional_impact_priority` are not mild/moderate/severe ground-truth labels. The dataset provides Bump/Pothole labels. The automatic priority is a transparent relative sensor-impact signal for the initial map and model workflow.
Two events have no aligned IMU samples and are shown on the API map as `sensor_data_unavailable`, rather than being assigned an unsupported impact tier.

## Phase 2: build and verify provisional priority

Run this only after Phase 1 succeeds:

```powershell
python scripts/build_phase2_manifests.py
python scripts/verify_phase2.py
python scripts/build_provisional_impact_priority.py
python scripts/verify_provisional_priority.py
```

This creates candidate-normal video windows and a sensor-derived provisional impact priority with GPS coordinates for map use. The severity-review queue and rubric remain available for later validation, but no manual labels are required for the current implementation phase.

## Phase 3: train and verify the IMU baseline

```powershell
python scripts/build_imu_baseline_dataset.py
python scripts/train_imu_baseline.py
python scripts/verify_phase3.py
```

This is a session-level 70/15/15 split: a whole recording can occur in only one split. The output is a baseline for **annotated road event versus candidate-normal window**, not a final road-condition classifier. Its held-out metrics and confusion matrix establish the benchmark that the camera and fusion models must beat.

## Phase 4A: train and verify the vision baseline

```powershell
python scripts/build_vision_baseline_manifest.py
python scripts/train_vision_hog_baseline.py
python scripts/verify_phase4a.py
```

This first camera baseline uses an aligned frame and HOG features. It is deliberately lightweight and reproducible; Phase 4B compares early and late fusion using the same frozen data split.

## Phase 4B: compare early and late fusion

```powershell
python scripts/run_fusion_comparison.py
python scripts/verify_phase4b.py
```

Early fusion concatenates visual and IMU features before classification. Late fusion combines independently trained visual and IMU probabilities through a meta-classifier trained on out-of-fold training predictions. The script selects the stronger method by validation F1, then average precision; it intentionally does **not** access the held-out test set.

## Phase 4C1: train and verify YOLO classification

```powershell
python scripts/prepare_yolo_classification_data.py
python scripts/train_yolo_classification.py
python scripts/verify_yolo_classification.py
```

YOLO is used in classification mode because the dataset has event timestamps and types but no bounding-box labels. The resulting event probability becomes the visual confidence for late fusion with IMU; no test images are prepared in this phase.

## Phase 4C2: compare learned visual signals with IMU

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" scripts\run_enhanced_fusion_comparison.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\verify_enhanced_fusion.py
```

The comparison evaluates three fusion options on validation sessions only: early clip-embedding plus IMU fusion, equal-weight YOLO plus IMU late fusion, and equal-weight clip plus IMU late fusion. The selected model is `late_yolo_imu`; test data remains untouched.

## Phase 4D: final selected-model evaluation

Phase 4D retrains only the validation-selected late YOLO plus IMU method on the combined development sessions, then evaluates the held-out test sessions once. The implementation refuses to run again after final metrics have been written.

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" scripts\run_phase4d_final_evaluation.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\verify_phase4d.py
```

The early-versus-late decision remains a validation comparison: early clip plus IMU F1 = 0.560 and late YOLO plus IMU F1 = 0.600. The final late-fusion held-out test F1 is 0.674 across 81 windows. Test negatives remain temporally separated candidate-normal windows, not inspected normal-road ground truth.

## Phase 5A: local GIS event store

```powershell
docker compose up -d
& "C:\Users\K8IN\anaconda3\python.exe" scripts\import_gis_events.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\verify_phase5a.py
& "C:\Users\K8IN\anaconda3\python.exe" -m uvicorn gis_service.app:app --reload
```

The API documents its endpoints at `http://127.0.0.1:8000/docs`. It exposes all 402 GPS-aligned annotated events as GeoJSON. Only 45 held-out annotated events carry final model probabilities because candidate-normal windows are not road hazards and have no GPS event record.

## Phase 5B: road matching and aggregation

The project downloads the OpenStreetMap drivable road network around the GPS event extent, imports it into PostGIS, and links each event to its nearest road segment within 30 m. Segment-level provisional maintenance priority combines event recurrence, the sensor-derived impact tier and available fused confidence. It must not be represented as a ground-truth defect-severity survey.

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" scripts\download_osm_roads.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\import_road_network.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\apply_phase5b_schema.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\verify_phase5b.py
```

After restarting the API, `http://127.0.0.1:8000/segments.geojson` returns the prioritised road segments for QGIS or the mobile map client.

## Phase 6A-6C: mobile map, sensor capture and live-event review

The Expo application in `mobile/` renders OpenStreetMap tiles with Leaflet inside Android WebView, without a Google Maps API key. It gathers a rolling phone accelerometer/gyroscope window, samples rear-camera frames after an IMU impact trigger, and computes the selected late-fusion score on-device. It can display local GIS event/segment layers from the API or active mobile events from Supabase; Supabase events can be retried from the bounded offline queue and dismissed without deleting their audit record. A local API URL is only reachable over the same Wi-Fi/hotspot, while cloud mode supports mobile data for live-event synchronisation. The map requires internet access for Leaflet assets and OpenStreetMap tiles. Automatic results are labelled Road anomaly until a separately evaluated bump-versus-pothole classifier is available. Follow `docs/phase6a_mobile_runbook.md` and `docs/phase7_zero_cost_runbook.md` for local and cloud setup.

## Main outputs

- `data/processed/video_annotation_manifest.csv`: video event locations and timing.
- `data/processed/fusion_event_windows.csv`: paired event windows and IMU features.
- `data/processed/osm_roads.geojson`: OpenStreetMap base-road network for the GPS recording extent.
- `data/eda/`: EDA plots, recording summary and dataset profile.
- `docs/literature_to_implementation.md`: research decisions and architecture rationale.
