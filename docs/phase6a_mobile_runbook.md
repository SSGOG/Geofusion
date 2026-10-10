# Phase 6A: mobile map client

The first mobile deliverable is a React Native (Expo) application that displays a refreshed GeoFusion event and prioritised road-segment map snapshot, then requests foreground GPS permission.

## Run it on an Android phone

1. Ensure the phone and this computer are on the same Wi-Fi network.
2. Keep PostGIS running, then start the GIS API so it accepts requests from the phone:

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" -m uvicorn gis_service.app:app --host 0.0.0.0 --port 8000
```

3. In a second terminal:

```powershell
cd mobile
Copy-Item .env.example .env
npm install
npx expo start
```

4. Install **Expo Go** on the Android phone, scan the QR code shown by Expo, and accept the foreground-location permission.

The supplied `.env.example` uses `192.168.1.5`, the current local Wi-Fi address. If the phone cannot load the map later, run `ipconfig`, replace that address in `.env`, stop Expo, and start it again. Allow private-network access if Windows Firewall prompts.

If Expo Go reports that it cannot connect to the development server, stop Expo and run `npx expo start --tunnel`. Scan the new QR code. The API still uses the local Wi-Fi address in `.env` for map data and tile delivery.

## Verify

The map should show 402 event markers and 72 coloured priority segments. Press **Refresh** to request a new API-rendered map snapshot; deny location permission once to confirm that map data still loads without it.

The Expo Go map uses an API-rendered snapshot because the current `react-native-maps` Android renderer is blank in Expo Go. A production development build can restore an interactive native map after its Android map-provider configuration is added.

## Phase 6B: phone sensor capture

The **Start sensing** button records accelerometer and gyroscope magnitudes at a 50 Hz target. Press **Stop sensing** to display the window statistics required by the IMU research baseline: sampling frequency, acceleration RMS deviation, peak deviation, jerk RMS and gyroscope peak.

## Phase 6C: camera, sensor fusion and live-map review

The current research prototype combines live road frames with a rolling phone sensor window and current GPS position. It runs the validation-selected late-fusion model: the YOLO road-event probability and IMU probability are averaged, then candidates at or above 0.50 are stored as active live events. Severity is a relative sensor-impact tier (low, medium, or high), calculated against the study-event feature distribution.

The visual model was trained for **road event versus candidate-normal road**, not for bump-versus-pothole classification. Automatic detections are therefore labelled **Road anomaly**. A separate bump-versus-pothole model must be trained and evaluated before the application can make that distinction automatically.

Before the first Phase 6C run, apply the additive schema and restart the API:

```powershell
& "C:\Users\K8IN\anaconda3\python.exe" scripts\apply_phase6c_schema.py
& "C:\Users\K8IN\anaconda3\python.exe" -m uvicorn gis_service.app:app --host 0.0.0.0 --port 8000
```

Then update the mobile dependencies and start Expo:

```powershell
cd mobile
npm install
npx expo start --tunnel -c
```

On the phone, follow this sequence:

1. Tap **Start live detection** with the phone securely mounted and its rear camera facing the road.
2. The app collects a rolling five-second IMU window. When an impact trigger is present, it samples a camera frame and submits the frame, sensor window, and current GPS automatically.
3. A candidate that passes fusion is marked at the current GPS position and reported with its relative sensor-impact tier. The map refreshes around the phone position; a blue point marks the live phone location.
4. Tap **Stop live detection** to end capture. If the latest mapped point is a false positive, tap **Remove latest**. The point disappears from the map while its record is retained as dismissed for later audit.

Do not operate the phone while driving. Capture evidence only with a passenger or when the vehicle is safely stationary. The resulting tier is a research prototype output, not a verified road-maintenance decision.
