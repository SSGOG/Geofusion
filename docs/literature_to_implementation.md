# GeoFusion literature review and implementation decisions

## Reading scope

This synthesis covers the 20 records supplied in `E:\downloads_2\scopus_part1.csv` through `scopus_part4.csv`. It is a design review, not a claim that every method has been reproduced. A limitation in the final column means the limitation that matters when transferring that method to GeoFusion; it is not necessarily a direct quotation from the authors.

## Finding that changes the severity plan

The selected RoadSens-4M data labels **Bump** and **Pothole**, but does not supply a validated mild/moderate/severe label. Accelerometer and gyroscope readings therefore support an **impact measure** and an efficient severity-annotation workflow; they do not, by themselves, establish true physical severity. Phone placement, vehicle suspension, speed and road approach all change the measured signal.

GeoFusion will use a three-stage approach:

1. Calculate transparent impact features from a synchronized 3-second IMU window: peak and RMS linear-acceleration deviation, jerk and gyroscope peak.
2. Sample events across low/medium/high impact values for double human review using a written visual/operational severity rubric.
3. Train and evaluate a three-level severity model only after those labels exist. Until then, the app exposes the number as a **provisional impact priority**, never as verified pothole severity.

`scripts/build_fusion_event_windows.py` implements stage 1. It produces `data/processed/fusion_event_windows.csv`: 402 video-aligned labelled events from 81 sessions, with source paths retained outside the workspace and no video duplication.

## Paper-by-paper synthesis

| # | Paper | Method or contribution | Transfer limitation | GeoFusion response |
|---|---|---|---|---|
| 1 | RoadSens-4M (2026), doi:10.1038/s41597-026-07072-y | Dataset of synchronized phone sensors, GPS/GIS/weather and 4K road video, with Bump/Pothole annotations. | Event type is available, but calibrated severity is not; not all recordings are locally present. | Use its aligned camera/IMU events for fusion, retain external video paths, and create a separate severity-label protocol. |
| 2 | Camera--IMU fusion for road-surface classification (2026), arXiv:2601.20847 | Lightweight bidirectional cross-attention and adaptive modality gating combine RGB and IMU. | Domain shift across weather, roads, phones and vehicles; limited real paired data. | Start with an interpretable late-fusion baseline, then test gated cross-attention with session-level splits and modality-ablation tests. |
| 3 | Multi-sensor fusion/deep learning survey (2026), doi:10.1007/s10462-026-11542-5 | Reviews data-, feature- and decision-level fusion for road scenes. | Labelling cost and cross-domain generalisation remain major barriers. | Keep raw modality-specific quality flags and plan self-supervised/pretraining only after a supervised baseline is measured. |
| 4 | CNN/LSTM multi-IMU road assessment (2025), doi:10.1038/s41598-025-10573-2 | CNN extracts local IMU patterns; LSTM captures temporal context across multiple IMUs. | Vehicle-specific sensor configuration and difficult classes reduce portability. | Use a short IMU temporal encoder, normalise by each recording/window baseline, and report results by session rather than random rows. |
| 5 | RDD2022 (2024), doi:10.1002/gdj3.260 | Large multi-country labelled road-damage image benchmark for object detection. | Vision-only labels/imagery cannot resolve vibration or route-level reliability. | Use it only later for visual pretraining/robustness tests; road event decisions remain camera plus IMU. |
| 6 | Improved YOLOv5 road-damage detection (2022), doi:10.1038/s41598-022-19674-8 | MobileNetV3, coordinate attention, revised anchors and deployment-oriented detection. | Detection boxes alone can be unstable under glare, shadows and small damage; it has no IMU confirmation. | Benchmark a compact detector for visual evidence, but require fusion confidence before publishing a map event. |
| 7 | Deep neural network road-damage detection (2024), doi:10.1007/s42452-024-06129-0 | Optimised visual detector for road damage. | Generalisation is tied to label definitions and camera/domain characteristics. | Keep type-label taxonomy explicit and assess performance on held-out sessions rather than one mixed image split. |
| 8 | Dashcam road-damage survey (2020), doi:10.1016/j.aei.2020.101182 | Surveys camera-based detection pipelines and datasets. | Image quality, occlusion, illumination and damage-scale variation remain unresolved in camera-only systems. | Use a pre/post event clip, orientation normalisation and IMU gating rather than a single frame alone. |
| 9 | Pothole disparity and road-surface modelling (2020), doi:10.1109/TIP.2019.2933750 | Uses stereo/disparity geometry to model surface deformation. | Requires stereo/depth geometry and calibration absent from ordinary one-phone deployments. | Do not make stereo a core dependency; consider it as a future validation modality only. |
| 10 | Unsupervised disparity segmentation (2020), doi:10.1109/TITS.2019.2947206 | Segments road-surface anomalies from disparity information without dense labels. | Depth/disparity quality depends on camera geometry and scene conditions. | Preserve a monocular-plus-IMU design for deployability; test depth only if device hardware makes it reliable. |
| 11 | Object detectors for Indian roads (2026), doi:10.1109/AIDE69088.2026.11544444 | Compares visual object detectors in an Indian-road context. | Country-specific visual data still does not account for vehicle dynamics or location precision. | Use it as regional motivation, while building local validation and sensor confirmation into the app. |
| 12 | Toward mobile crowdsensing for road assessment (2018), doi:10.1016/j.compenvurbsys.2017.12.005 | Positions crowdsensed mobile/GPS measurements for network-scale road assessment. | Crowdsourcing creates uneven coverage and heterogeneous measurements. | Aggregate repeated independent passes at road-segment level instead of trusting a single report. |
| 13 | Smart patrolling for road-surface monitoring (2017), doi:10.1016/j.pmcj.2017.06.002 | Mobile-phone sensing supports cost-effective patrol/monitoring. | Phone orientation, driving and vehicle factors confound vibration readings. | Record device/vehicle metadata where possible; use gravity/orientation features and calibration checks. |
| 14 | Smartphone crowdsourcing road diagnostics (2021), doi:10.1016/j.jtte.2020.09.004 | Combines temporal and spatial evidence, aggregates results on fixed 10-m road sections and compares with reference data. | A point-level alert can be noisy and a single pass is insufficient. | Store raw events, then map-match and aggregate to 10--25 m road segments with recurrence, recency and diversity. |
| 15 | Device- and vehicle-independent crowd profiling (2020), doi:10.1016/j.pmcj.2019.101103 | Studies making road profiles less dependent on sensing device and vehicle. | Full independence is difficult because hardware and suspension alter signal amplitude. | Use per-recording normalisation and evaluate leave-one-session/device/vehicle-out generalisation when metadata allows. |
| 16 | Smartphone-sensor road-anomaly review (2018), doi:10.3390/s18113845 | Reviews accelerometer, gyroscope, GPS and related phone approaches. | Lower phone sampling rates and undersampling can lower accuracy. | Record effective sampling frequency, reject inadequate windows and run sensor collection natively on Android. |
| 17 | Smartphone-app pavement-monitoring review (2024), doi:10.1016/j.conbuildmat.2023.134207 | Reviews app-based pavement monitoring and deployment concerns. | Signals reflect phone, vehicle, driving and environment as well as pavement. | Treat impact score as a calibrated model output with uncertainty, not a direct measurement of road depth. |
| 18 | GIS maintenance prioritisation in Nepal (2010), doi:10.1016/j.jtrangeo.2009.06.016 | Uses GIS to support road-maintenance planning. | Prioritisation depends on local cost, traffic, condition and policy inputs. | Separate event evidence from the authority-configurable maintenance score; do not hard-code a universal priority formula. |
| 19 | Pavement management system for local administrators (2023), doi:10.1016/j.trpro.2023.11.263 | GIS-enabled data management, analysis and map display for pavement decisions. | A useful dashboard needs maintained road inventory and local workflows. | Use PostGIS road segments, review status and audit fields, with QGIS for analyst validation rather than the mobile client. |
| 20 | Intelligent pavement management with ML (2026), doi:10.1038/s41598-026-49953-7 | Compares conventional ML and deep models using distress, geometry and traffic factors for priority classes. | Model outputs depend on the available agency/context features and class rules. | Keep detection/severity separate from maintenance priority; add traffic, road class and recurrence only when reliable GIS data is available. |

## Selected model and GIS pipeline

### Event model

**Input:** a 3-second video clip centred on the annotated event and the aligned 3-second IMU window.

**Stage A — robust baselines:**

- Vision-only compact detector/classifier on event clips.
- IMU-only temporal model using acceleration, gyroscope, sampling rate and derived features.
- Late-fusion logistic/gradient-boosting baseline over each modality's calibrated confidence and IMU features.

**Stage B — proposed GeoFusion model:** RGB clip encoder + IMU temporal encoder + gated cross-modal attention. The gate allows the system to lower the visual contribution under glare/blur or the sensor contribution when sampling/placement quality is poor. The model first predicts event/no-event and Bump/Pothole; severity is an additional head only after a reviewed severity set exists.

**Evaluation:** split by recording session before any frame or window extraction. Report precision, recall, F1, latency and ablation (RGB only, IMU only, fusion). For severity, report macro-F1, weighted kappa with reviewers, calibration and errors by speed/orientation/location cluster.

### GIS pipeline

1. App records an event with coordinates, direction, time, model confidence, type and provisional/validated severity.
2. Server map-matches the event to a road segment, retaining the original coordinate and match distance for audit.
3. Multiple events are clustered and aggregated over 10--25 m segments. A segment score uses validated severity, repeated independent passes, recency, confidence and later road-class/traffic context.
4. The mobile map displays the current segment status; a GIS analyst uses QGIS to validate road coverage and segment logic. Raw point detections remain reviewable.

## Mobile architecture decision: React Native is viable

React Native can be the main mobile-app layer. The recommended form is **React Native for interface, MapLibre map and workflow; Kotlin inside the Android project for high-frequency sensors, CameraX capture and TensorFlow Lite inference.** The Kotlin module exposes a small, stable event stream to React Native. This avoids asking JavaScript timers and generic sensor packages to guarantee synchronized 50--100 Hz capture or continuous on-device inference.

| Layer | Choice | Reason |
|---|---|---|
| Mobile UI and map | React Native + MapLibre React Native | One user interface codebase and an open map stack. |
| Sensor/camera/inference | Native Kotlin module, CameraX, Android sensor APIs, TensorFlow Lite | Reliable timing, background controls and on-device model execution. |
| API | FastAPI | Simple typed event API and model-serving option. |
| Spatial data | PostgreSQL + PostGIS | Map matching, clustering, segment aggregation and audit queries. |
| Research GIS | QGIS | Desktop validation, road-segment inspection and exported authority layers; not part of the end-user app. |

## Immediate next work

1. Define the severity rubric and label 200--250 diverse events, double-reviewing a subset.
2. Generate normal-road windows from the five normal sessions and sample non-event portions of anomaly sessions with temporal separation.
3. Build the session-level train/validation/test split before model selection.
4. Train IMU-only, vision-only and late-fusion baselines; then justify cross-modal attention only if it wins on the untouched test sessions.
5. Prototype the React Native map screen and native Android sensing module after the input-window contract is stable.
