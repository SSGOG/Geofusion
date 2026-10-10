"""Export the validation-selected final YOLO classifier for Android inference."""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "runs" / "geofusion_yolo_cls_final" / "weights" / "best.pt"
DESTINATION = ROOT / "mobile" / "assets" / "models" / "geofusion_yolo_late.onnx"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Selected YOLO weights were not found: {SOURCE}")
    exported = Path(YOLO(str(SOURCE)).export(format="onnx", imgsz=224, opset=17, simplify=False, dynamic=False))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, DESTINATION)
    if exported.resolve() != DESTINATION.resolve():
        exported.unlink()
    print(f"Exported selected late-fusion YOLO model to {DESTINATION}")


if __name__ == "__main__":
    main()
