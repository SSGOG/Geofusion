"""Index local GeoFusion MP4 recordings and align raw annotations to video frames.

Videos remain on E:\\downloads_2\\Raw Data. The generated CSV is a lightweight
manifest that records the external path, annotation timestamp and target frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"E:\downloads_2\Raw Data")
OUT = ROOT / "data" / "processed" / "video_annotation_manifest.csv"


def main() -> None:
    rows = []
    for video in sorted(RAW.rglob("*.mp4")):
        session_dir = video.parent.parent
        annotation_path = session_dir / "Annotation.csv"
        if not annotation_path.exists():
            continue
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        duration = frames / fps if fps else None
        annotations = pd.read_csv(annotation_path)
        annotations = annotations[annotations["text"].astype(str).str.strip().isin(["Bump", "Pothole"])]
        for _, event in annotations.iterrows():
            seconds = float(event["seconds_elapsed"])
            frame_index = int(round(seconds * fps)) if fps else None
            rows.append(
                {
                    "session": session_dir.name,
                    "video_path": str(video),
                    "video_start_epoch_ms": int(video.stem) if video.stem.isdigit() else None,
                    "fps": fps,
                    "frame_count": frames,
                    "duration_seconds": duration,
                    "width_px": width,
                    "height_px": height,
                    "event_type": event["text"].strip(),
                    "annotation_time_ns": int(event["time"]),
                    "seconds_elapsed": seconds,
                    "target_frame": frame_index,
                    "frame_within_video": bool(frame_index is not None and 0 <= frame_index < frames),
                    "press_duration_ms": event["millisecond_press_duration"],
                }
            )
    manifest = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)
    print(f"Events indexed: {len(manifest)}")
    print(f"Sessions indexed: {manifest['session'].nunique()}")
    print(f"Events inside local video bounds: {manifest['frame_within_video'].sum()} of {len(manifest)}")
    print(f"Resolution counts: {manifest[['width_px', 'height_px']].value_counts().to_dict()}")
    print(f"Frame-rate range: {manifest['fps'].min():.4f} to {manifest['fps'].max():.4f} fps")


if __name__ == "__main__":
    main()
