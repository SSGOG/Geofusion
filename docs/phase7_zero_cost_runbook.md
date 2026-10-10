# GeoFusion zero-cost field path

## Selected deployment model

The mobile build uses the validation-selected late-fusion method. The Android package contains the exported YOLO visual classifier and the fitted IMU logistic-regression parameters. It calculates visual probability, IMU probability and their equal-weight fused probability on the phone. Early fusion remains in the repository as a research comparator and is not packaged for field inference.

The IMU training data used a fixed 100 Hz sampling feature. The mobile scorer therefore normalises that input to the 100 Hz training reference until a phone-specific calibration set has been collected. The measured phone sampling rate is still retained with each event for later analysis.

## Current free deployment boundary

- Android inference and the bounded offline queue are phone-side.
- The app renders OpenStreetMap tiles with Leaflet inside Android WebView and plots GeoFusion events without Google Maps or a laptop-rendered map image.
- Leaflet is loaded from the pinned unpkg CDN URL and map tiles come from the public OpenStreetMap tile service; mobile internet is required, and neither service provides offline tiles or a production SLA.
- The local Docker/PostGIS service remains suitable for development on the same Wi-Fi or hotspot network; its road-segment layer remains available in local-API mode.
- A phone on mobile data cannot reach the laptop's private `192.168.x.x` address. Supabase Free PostGIS is the intended public event-synchronisation target for Phase 7C.
- OpenStreetMap remains the map source. Keep normal interactive tile usage and attribution; the public tile service is best-effort, not a production SLA or an offline tile-download service.

## Build and share an Android tester APK

Expo Go cannot load the native ONNX runtime, so use the custom development build configured in `eas.json`.

1. Configure `ANDROID_HOME` and `ANDROID_SDK_ROOT` to point to the local Android SDK. This project uses Android Platform 36, Build Tools 36.0.0, NDK 27.1.12297006 and CMake 3.22.1.
2. Open PowerShell in `mobile\android` and build locally with `.\gradlew.bat :app:assembleRelease`.
3. Share `mobile\android\app\build\outputs\apk\release\GeoFusion-0.1.1-test.apk` and its `.sha256.txt` checksum with invited Android testers. The package targets `com.geofusion.mobile`, supports Android API 24 and newer, and includes the on-device model. The current APK is debug-signed and for trusted sideloaded testing only; configure a private release keystore before public distribution or publishing to an app store.
4. Testers should verify the downloaded file's SHA-256 checksum, allow installs from the app they use to open the APK, and install it. Do not uninstall an existing installation just to update it: uninstalling erases its local queued events. An update must use the same signing key; builds from a different debug keystore require uninstall/reinstall and can lose local data.
5. For JavaScript/UI development, run `npx expo start --dev-client` from `mobile` and open the installed build. Rebuild after adding/changing native dependencies or Android configuration.

Do not use a hosted build unless you explicitly approve sending the project source and model assets to that service.

## Controlled field test

1. Keep the phone securely fixed in the vehicle and have a passenger operate the application where possible.
2. Start live detection. It samples IMU data, takes a periodic road frame only after an impact trigger, computes late fusion locally and queues candidates if synchronisation fails.
3. Confirm the app shows a visual, IMU and combined probability. Do not interpret the current binary visual label as a verified pothole-versus-bump classification.
4. Use **Remove latest** to dismiss either the newest queued candidate or newest mapped record when it is a false positive.
5. Reconnect to the development API and refresh; queued records synchronise once and are removed from the phone queue.

## Phase 7C account handoff

Create one Supabase Free project, enable anonymous sign-ins, and enable PostGIS in the `extensions` schema. Copy the contents of `sql/004_supabase_mobile_events.sql` into the Supabase SQL Editor and run it. Then copy `mobile/.env.example` to `mobile/.env` and set the project URL and publishable key. Never put a Supabase `service_role` key in the mobile application. The client lists up to 500 active cloud events and may insert or dismiss events owned by its anonymous identity. Road-segment aggregation is not mirrored into the cloud mobile table.

The schema requires row-level security. A phone receives an anonymous authenticated identity, may create and dismiss only its own events, and may read active map events. This is the correct access boundary for the field prototype; public reporting and anti-abuse controls are a later production concern.
