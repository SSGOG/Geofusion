"""Create compact three-frame RGB clips for a learned GeoFusion temporal encoder.

Only train and validation sessions are materialised. Each source clip is three
96x96 frames at -1, 0 and +1 seconds around the aligned event/window centre.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "derived" / "clip_encoder"
SIZE = 96
OFFSETS = (-1.0, 0.0, 1.0)


def decode_clip(video_path: str, centre: float) -> np.ndarray:
    for _ in range(3):
        capture = cv2.VideoCapture(video_path)
        frames = []
        for offset in OFFSETS:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, centre + offset) * 1000)
            success, frame = capture.read()
            if not success or frame is None:
                frames = []
                break
            frame = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            frames.append(frame)
        capture.release()
        if len(frames) == len(OFFSETS):
            return np.stack(frames, axis=0)
    raise RuntimeError(f"Could not decode {video_path} at all requested clip timestamps around {centre:.3f}s")


def main() -> None:
    manifest = pd.read_csv(PROCESSED / "vision_baseline_manifest.csv")
    manifest = manifest.loc[manifest["split"].isin(["train", "validation"])].copy().reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(OUT / "clips_raw_uint8.npy", mode="w+", dtype=np.uint8, shape=(len(manifest), len(OFFSETS), SIZE, SIZE, 3))
    valid = np.zeros(len(manifest), dtype=bool)
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit only small batches so pending 4K decode results cannot exhaust RAM.
        for start in range(0, len(manifest), 16):
            batch = manifest.iloc[start : start + 16]
            futures = {executor.submit(decode_clip, str(row.video_path), float(row.event_seconds)): index for index, row in batch.iterrows()}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    raw[index] = future.result()
                    valid[index] = True
                except RuntimeError as error:
                    failures.append({"event_id": manifest.loc[index, "event_id"], "video_path": manifest.loc[index, "video_path"], "reason": str(error)})
            print(f"Processed {min(start + len(batch), len(manifest))}/{len(manifest)} compact clips")
    del raw
    good = manifest.loc[valid].copy().reset_index(drop=True)
    raw_read = np.load(OUT / "clips_raw_uint8.npy", mmap_mode="r")
    clips = np.lib.format.open_memmap(OUT / "clips_uint8.npy", mode="w+", dtype=np.uint8, shape=(len(good), len(OFFSETS), SIZE, SIZE, 3))
    clips[:] = raw_read[valid]
    del clips, raw_read
    good.to_csv(OUT / "source_manifest.csv", index=False)
    pd.DataFrame(failures, columns=["event_id", "video_path", "reason"]).to_csv(OUT / "decode_failures.csv", index=False)
    print(f"Clip tensor: {OUT / 'clips_uint8.npy'}")
    print(f"Rows by split: {good['split'].value_counts().to_dict()}")
    print(f"Decode failures: {len(failures)}")


if __name__ == "__main__":
    main()
