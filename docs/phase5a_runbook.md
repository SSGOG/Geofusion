# Phase 5A GIS event store

Start Docker Desktop, then run the following in the project root.

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" -m pip install -r requirements.txt
docker compose up -d
& "C:\Users\K8IN\anaconda3\python.exe" scripts\import_gis_events.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\verify_phase5a.py
& "C:\Users\K8IN\anaconda3\python.exe" -m uvicorn gis_service.app:app --reload
```

Open `http://127.0.0.1:8000/docs` and test `GET /health`, `GET /events.geojson`, and `GET /events/summary`.

In QGIS, add `http://127.0.0.1:8000/events.geojson` as a GeoJSON layer after the API is running. The layer contains all GPS-aligned annotated events; final model probabilities exist for the 45 GPS-annotated held-out events. Candidate-normal test windows are intentionally not mapped as road hazards.

Phase 5B needs a road-network GeoJSON or Shapefile for the recording area. It will populate `road_segments`, map-match events, and calculate segment-level maintenance priority. Do not treat the current impact tier as verified road-defect severity.

## Phase 5B: road matching and provisional maintenance priority

Keep Docker Desktop running, then open a second PowerShell terminal in the project root. The API documentation page by itself does not verify the database; run the final command below before considering this phase complete.

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" scripts\download_osm_roads.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\import_road_network.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\apply_phase5b_schema.py
& "C:\Users\K8IN\anaconda3\python.exe" scripts\verify_phase5b.py
```

Restart the API if it is already running. `GET /segments.geojson` then returns road segments ordered by provisional maintenance score. Add `http://127.0.0.1:8000/segments.geojson` as a second GeoJSON layer in QGIS, style it by `maintenance_priority`, and retain the event layer for evidence.

OpenStreetMap data are used only as the base road network. The displayed priority combines repeated annotated events, the existing sensor-derived impact tier and any available final fused confidence; it is a prioritisation signal, not a surveyed defect-severity measurement.

Two events have no aligned IMU samples. Their API map category is `sensor_data_unavailable`; it must not be interpreted as a low-impact event.
