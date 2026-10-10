"""Check Android-format YOLO output against the selected PyTorch classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "runs" / "geofusion_yolo_cls_final" / "weights" / "best.pt"
EXPORT = ROOT / "mobile" / "assets" / "models" / "geofusion_yolo_late.onnx"
IMAGE = ROOT / "data" / "derived" / "final_yolo_cls" / "train" / "annotated_event" / "1_0000.jpg"


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    require(SOURCE.exists(), "selected PyTorch YOLO weights remain preserved")
    require(EXPORT.exists() and EXPORT.stat().st_size > 1_000_000, "portable ONNX visual model exists")
    image = cv2.imread(str(IMAGE))
    require(image is not None, "development image is available for export verification")
    pytorch = YOLO(str(SOURCE)).predict(str(IMAGE), imgsz=224, device="cpu", verbose=False)[0].probs.data.cpu().numpy()
    rgb = cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255)
    session = ort.InferenceSession(str(EXPORT), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    require(output.shape == pytorch.shape, "ONNX output has the expected two-class shape")
    require(np.all(np.isfinite(output)) and abs(float(output.sum()) - 1) < 1e-4, "ONNX output is a valid probability vector")
    require(float(np.max(np.abs(output - pytorch))) < 1e-4, "ONNX probabilities match the selected PyTorch model")
    print("\nPortable YOLO export verification completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, RuntimeError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
