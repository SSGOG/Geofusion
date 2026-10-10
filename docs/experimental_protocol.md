# GeoFusion experimental protocol

## Research question

Does combining a road-facing RGB frame with phone IMU features improve the detection of an annotated road event over either modality alone?

## Fixed evaluation rule

The session assignment in `data/processed/session_splits.csv` is frozen. Frames or sensor windows from one recording session must never cross train, validation or test splits. Models are selected using validation F1 and average precision. The test split is reported only after the modelling decision is fixed.

## Comparable models

| Model | Inputs | Purpose |
|---|---|---|
| M1: IMU-only | Five IMU summary features from a 3-second window | Existing physical-sensing baseline. |
| M2: Vision-only | HOG visual representation from one aligned road frame | Initial camera baseline. |
| M2b: YOLO classification | Fine-tuned YOLO11n classification probability from an event-centred frame | Learned visual baseline without bounding boxes. |
| M3: Early fusion | Concatenated, scaled visual and IMU features in one classifier | Tests whether joint feature learning improves detection. |
| M4: Late fusion | Calibrated IMU and visual probabilities combined by validation-trained logistic regression | Tests whether independent modality decisions combine more reliably. |

M3 and M4 will use exactly the same rows, session splits and candidate-normal limitation as M1/M2. The selected fusion method is the one with stronger validation F1; average precision, recall, F1, ROC-AUC, latency and test performance will all be reported.

## Scope statement

The initial target is **annotated Bump/Pothole event versus temporally separated candidate-normal window**. The automatic `high_impact` tier is a map-priority signal, not a training label for true defect severity. The later mobile model will use short clips and temporal IMU encoders; this first experiment establishes reproducible multimodal evidence before that higher-capacity step.
