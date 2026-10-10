# Implementation Plan: zero-cost GeoFusion field prototype

## Overview

Move the validation-selected late YOLO-plus-IMU fusion path toward on-device Android inference and free GIS synchronisation. Preserve early fusion, clip encoder and baseline artifacts unchanged as research comparators. Docker remains the reproducible local backend; Hugging Face compute is not selected because it is not a dependable zero-cost mobile backend.

## Architecture decisions

- Deploy only `late_yolo_imu`: validation F1 0.600 exceeds early clip-plus-IMU F1 0.560; its one-time held-out test F1 is 0.674.
- Keep the selected YOLO weights and a portable representation of the fitted IMU pipeline for mobile use. Do not retrain or re-evaluate the held-out test set.
- Use Supabase Free with PostGIS for event synchronisation after on-device inference is validated. The phone should queue unsent events when offline.
- Retain OpenStreetMap for the zero-cost map. Google Maps and a paid cloud inference API are out of scope for this path.

## Task list

### Phase 7A: portable selected-model artifacts

- [x] Task 1: Export the final fitted IMU scaler, medians and logistic-regression coefficients to a versioned JSON artifact. Phone sampling frequency is normalised to the fixed 100 Hz training reference because the fitted feature has near-zero training variance.
  - Acceptance: JSON has the five feature names, imputation values, scaling values, coefficients, intercept and equal-weight late-fusion rule.
  - Verification: a local evaluator reproduces the sklearn IMU and fusion probabilities within floating-point tolerance.
  - Dependencies: None.

- [x] Task 2: Export the final YOLO classifier to an Android-capable inference format without deleting `.pt` weights.
  - Acceptance: source weights stay unchanged; exported artifact loads and produces a valid two-class output on a known image.
  - Verification: focused export verification script passes.
  - Dependencies: Task 1.

### Checkpoint: portable-model evidence

- [x] Late-fusion probabilities reproduce locally.
- [x] No held-out evaluation is rerun.
- [x] Early-fusion and clip artifacts remain present.

### Phase 7B: phone inference and offline queue

- [ ] Task 3: Add a native Android inference runtime and run the exported visual model on periodic rear-camera frames.
  - Acceptance: phone displays visual, IMU and fused probabilities; automatic detection needs no capture tap.
  - Verification: Android development build runs a controlled on-road test.
  - Dependencies: Task 2.

- [ ] Task 4: Add a durable local event queue and false-positive dismissal state.
  - Acceptance: events survive loss of network and dismissed events are not resubmitted.
  - Verification: enable airplane mode, generate a controlled test event, reconnect and confirm one synchronised record.
  - Dependencies: Task 3.

### Checkpoint: device path

- [ ] Controlled road test records a late-fusion candidate using phone-only inference.
- [ ] Queue and dismissal behavior verified.

### Phase 7C: free GIS synchronisation and installable build

- [ ] Task 5: Configure Supabase Free PostGIS, restricted API access and event synchronisation.
  - Acceptance: a mobile event is stored with geometry, confidence, severity and dismissal status.
  - Verification: map reload shows a synchronised event from another client session.
  - Dependencies: Task 4 and a user-created Supabase project.

- [ ] Task 6: Create an installable Android APK and configure over-the-air JavaScript updates.
  - Acceptance: APK runs without Expo Go and targets the production synchronisation URL.
  - Verification: install on the user phone over mobile data and complete one controlled road test.
  - Dependencies: Task 5 and an Expo account.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Binary visual labels cannot distinguish bump from pothole | Keep automatic label `Road anomaly`; train a multiclass model only after a separate evaluation. |
| Phone camera runtime differs from Python YOLO runtime | Verify exported model output against a fixed local image before Android integration. |
| Free services have limits | Make inference and queue phone-side; use Supabase only for synchronisation. |
| Unsafe data collection while driving | Require fixed mounting and a passenger or stationary setup for controls. |
