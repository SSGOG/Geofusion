FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY gis_service gis_service
COPY scripts/import_gis_events.py scripts/import_road_network.py scripts/apply_phase5b_schema.py scripts/apply_phase6c_schema.py scripts/bootstrap_deployment_database.py scripts/
COPY sql sql
COPY models/phase4d_final_imu_pipeline.joblib models/phase4d_final_imu_pipeline.joblib
COPY models/runs/geofusion_yolo_cls_final/weights/best.pt models/runs/geofusion_yolo_cls_final/weights/best.pt
COPY data/processed/fusion_event_windows.csv data/processed/fusion_event_windows.csv
COPY data/processed/phase4d_final_late_yolo_imu_test_predictions.csv data/processed/phase4d_final_late_yolo_imu_test_predictions.csv
COPY data/processed/osm_roads.geojson data/processed/osm_roads.geojson

CMD ["sh", "-c", "uvicorn gis_service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
