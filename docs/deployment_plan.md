# GeoFusion field deployment plan

## What is deployed

The deployment consists of a public FastAPI web service containing the selected YOLO classifier, IMU pipeline, GIS map endpoint and database bootstrap utility; and a managed PostgreSQL instance with PostGIS. The Android application is configured with the public API URL and distributed as an installable development or production build.

The deployment package deliberately includes only the model artefacts and three derived data files required for inference, severity reference and initial GIS layers. It excludes raw recordings, compact clips, Python virtual environments and mobile dependencies.

## First deployment sequence

1. Create a private source repository containing the deployment package.
2. Create a managed Postgres instance with PostGIS enabled.
3. Set the web service environment variable `GEOFUSION_DATABASE_URL` to the database connection string.
4. Run `python scripts/bootstrap_deployment_database.py` once against that database. It is safe to rerun because source events and roads are upserted.
5. Deploy the Docker web service and verify `/health`, `/events.geojson`, `/segments.geojson` and `/map.png` through its HTTPS URL.
6. Set `EXPO_PUBLIC_GEOFUSION_API_URL` to that HTTPS URL, build the Android application, then test a controlled road route.

## Updating after deployment

Backend code and model fixes are deployed from the source repository and take effect for all mobile clients. JavaScript-only mobile changes can use an Expo over-the-air update. Changes to native permissions, installed native packages or a Google Maps integration require a new Android build.

## Field constraints

An always-on public API is required for practical cellular use. A free service that sleeps after inactivity is unsuitable for roadside testing because a first detection may time out while it wakes. Automatic results remain labelled `Road anomaly` until a separately evaluated bump-versus-pothole classifier is deployed.
