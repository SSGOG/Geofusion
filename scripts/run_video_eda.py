"""Video EDA for local GeoFusion MP4s referenced by the alignment manifest."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "video_annotation_manifest.csv"
OUT = ROOT / "data" / "eda"


def frame_metrics(video_path: str, frame_number: int) -> tuple[float, float] | tuple[None, None]:
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None, None
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_contact_sheet(data: pd.DataFrame) -> None:
    selected = []
    for event_type in ["Bump", "Pothole"]:
        selected.append(data[data["event_type"] == event_type].sample(4, random_state=42))
    selection = pd.concat(selected, ignore_index=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (_, row) in zip(axes.flat, selection.iterrows()):
        capture = cv2.VideoCapture(row["video_path"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["target_frame"]))
        ok, frame = capture.read()
        capture.release()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            axis.imshow(frame)
        axis.set_title(f"{row['event_type']} | session {row['session']} | {row['seconds_elapsed']:.2f} s", fontsize=10)
        axis.axis("off")
    fig.suptitle("Annotated video frames aligned with road-anomaly timestamps", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "video_event_contact_sheet.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(MANIFEST)
    data["orientation"] = np.where(data["width_px"] >= data["height_px"], "Landscape", "Portrait")
    quality_sample = pd.concat(
        [group.sample(min(len(group), 8), random_state=42) for _, group in data.groupby("event_type")],
        ignore_index=True,
    )
    metrics = [frame_metrics(path, frame) for path, frame in zip(quality_sample["video_path"], quality_sample["target_frame"])]
    quality_sample[["brightness_mean", "sharpness_laplacian_variance"]] = pd.DataFrame(metrics, index=quality_sample.index)
    quality_sample["quality_eda_sample"] = True
    quality_sample.to_csv(OUT / "video_event_frame_metrics.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=data, x="event_type", color="#4D4D4D", ax=axes[0])
    axes[0].set(title="Video-backed annotated events", xlabel="Event type", ylabel="Number of events")
    orientation_counts = data[["session", "orientation"]].drop_duplicates()["orientation"].value_counts().reindex(["Landscape", "Portrait"], fill_value=0)
    axes[1].bar(orientation_counts.index, orientation_counts.values, color=["#4D4D4D", "#A6A6A6"])
    axes[1].set(title="Video orientation by session", xlabel="", ylabel="Number of sessions")
    fig.tight_layout()
    fig.savefig(OUT / "video_inventory_eda.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=quality_sample, x="event_type", y="brightness_mean", color="#B3B3B3", ax=axes[0])
    axes[0].set(title="Frame brightness at annotated events", xlabel="Event type", ylabel="Mean grayscale intensity (0-255)")
    sns.boxplot(data=quality_sample, x="event_type", y="sharpness_laplacian_variance", color="#B3B3B3", ax=axes[1])
    axes[1].set(title="Frame sharpness at annotated events", xlabel="Event type", ylabel="Laplacian variance")
    fig.tight_layout()
    fig.savefig(OUT / "video_frame_quality_eda.png", dpi=220)
    plt.close(fig)

    make_contact_sheet(data)
    print(f"Aligned annotated events: {len(data)}")
    print(f"Quality-EDA sampled frames: {len(quality_sample)}")
    print(f"Frames successfully read: {quality_sample['brightness_mean'].notna().sum()}")
    print(f"Video sessions: {data['session'].nunique()}")
    print(quality_sample.groupby("event_type")[["brightness_mean", "sharpness_laplacian_variance"]].median().round(2).to_string())
    print(data[["session", "orientation"]].drop_duplicates()["orientation"].value_counts().to_string())


if __name__ == "__main__":
    main()
