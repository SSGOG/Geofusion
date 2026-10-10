"""Extract compact aligned frames for YOLO classification without touching test data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATASET_ROOT = ROOT / "data" / "derived" / "yolo_cls"
IMAGE_SIZE = 224


def save_frame(row: dict[str, object]) -> dict[str, object]:
    capture = cv2.VideoCapture(str(row["video_path"]))
    capture.set(cv2.CAP_PROP_POS_MSEC, float(row["event_seconds"]) * 1000)
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise RuntimeError(f"Could not decode {row['event_id']}")
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    output_path = Path(str(row["image_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Could not write {output_path}")
    return row


def main() -> None:
    manifest = pd.read_csv(PROCESSED / "vision_baseline_manifest.csv")
    # Test is deliberately excluded until a method is selected.
    manifest = manifest.loc[manifest["split"].isin(["train", "validation"])].copy()
    manifest["class_name"] = manifest["target_anomaly"].map({1: "annotated_event", 0: "candidate_normal"})
    manifest["image_path"] = [
        str(DATASET_ROOT / split / label / f"{event_id}.jpg")
        for event_id, split, label in zip(manifest["event_id"], manifest["split"], manifest["class_name"])
    ]
    rows = manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(save_frame, row) for row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 50 == 0 or completed == len(futures):
                print(f"Saved {completed}/{len(futures)} YOLO classification frames")
    output = DATASET_ROOT / "source_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Rows by split/class: {manifest.groupby(['split', 'class_name']).size().to_dict()}")


if __name__ == "__main__":
    main()
