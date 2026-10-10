"""Reproducible exploratory analysis for the GeoFusion road-sensing dataset.

The source archives are retained under data/raw. This script analyses the
extracted combined CSV recordings and writes only derived artefacts to
data/eda. A recording is labelled from its archive folder: road-anomaly or
normal-road. This avoids treating an empty annotation field as a negative label.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
ANOMALY_DIR = ROOT / "data" / "processed" / "anomaly_csv"
NORMAL_DIR = ROOT / "data" / "processed" / "normal_csv"
OUT = ROOT / "data" / "eda"
RAW_DATA_DIR = Path(r"E:\downloads_2\Raw Data")

ACC = ["totalAcceleration_x", "totalAcceleration_y", "totalAcceleration_z"]
GYRO = ["gyroscope_x", "gyroscope_y", "gyroscope_z"]
WEATHER = ["Temperature (°C)", "Humidity (%)", "Cloud Cover (%)", "Wind Speed (km/h)"]
GEO = ["location_latitude", "location_longitude", "GIS_Elevation"]


def safe_stat(frame: pd.DataFrame, column: str, method: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)
    return float(getattr(values, method)()) if values.notna().any() else float("nan")


def analyse_recording(path: Path, label: str) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    seconds = pd.to_numeric(frame["seconds_elapsed"], errors="coerce") if "seconds_elapsed" in frame else pd.Series(dtype=float)
    deltas = seconds.diff().dropna()
    acc = frame.reindex(columns=ACC).apply(pd.to_numeric, errors="coerce")
    gyro = frame.reindex(columns=GYRO).apply(pd.to_numeric, errors="coerce")
    acc_magnitude = np.sqrt((acc**2).sum(axis=1))
    gyro_magnitude = np.sqrt((gyro**2).sum(axis=1))
    duration = float(seconds.max() - seconds.min()) if seconds.notna().any() else float("nan")
    median_dt = float(deltas.median()) if not deltas.empty else float("nan")
    row = {
        "recording": path.name,
        "class": label,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duration_seconds": duration,
        "median_sample_interval_seconds": median_dt,
        "estimated_sampling_hz": (1 / median_dt) if median_dt > 0 else float("nan"),
        "acceleration_magnitude_mean": float(acc_magnitude.mean()),
        "acceleration_magnitude_std": float(acc_magnitude.std()),
        "acceleration_magnitude_max": float(acc_magnitude.max()),
        "gyroscope_magnitude_mean": float(gyro_magnitude.mean()),
        "gyroscope_magnitude_max": float(gyro_magnitude.max()),
        "latitude": safe_stat(frame, "location_latitude", "median"),
        "longitude": safe_stat(frame, "location_longitude", "median"),
        "gis_elevation_m": safe_stat(frame, "GIS_Elevation", "median"),
        "missing_cell_percent": float(frame.isna().mean().mean() * 100),
    }
    for column in WEATHER:
        row[column] = safe_stat(frame, column, "median")
    sample = pd.DataFrame(
        {
            "class": label,
            "acceleration_magnitude": acc_magnitude,
            "gyroscope_magnitude": gyro_magnitude,
            "latitude": pd.to_numeric(frame["location_latitude"], errors="coerce") if "location_latitude" in frame else np.nan,
            "longitude": pd.to_numeric(frame["location_longitude"], errors="coerce") if "location_longitude" in frame else np.nan,
        }
    )
    return row, sample


def plot_outputs(summary: pd.DataFrame, samples: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"Road anomaly": "#9B2226", "Normal road": "#1D3557"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=summary, x="class", hue="class", palette=palette, legend=False, ax=axes[0])
    axes[0].set(title="Recordings by class", xlabel="", ylabel="Number of recordings")
    sns.boxplot(data=summary, x="class", y="duration_seconds", hue="class", palette=palette, legend=False, ax=axes[1])
    axes[1].set(title="Recording duration", xlabel="", ylabel="Seconds")
    fig.tight_layout()
    fig.savefig(OUT / "recording_overview.png", dpi=220)
    plt.close(fig)

    sampled_groups = []
    for _, group in samples.dropna(subset=["acceleration_magnitude"]).groupby("class"):
        sampled_groups.append(group.sample(min(len(group), 15000), random_state=42))
    display = pd.concat(sampled_groups, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(data=display, x="class", y="acceleration_magnitude", hue="class", palette=palette, legend=False, cut=0, ax=ax)
    ax.set(title="Total-acceleration magnitude by class", xlabel="", ylabel="Magnitude (m/s²)")
    fig.tight_layout()
    fig.savefig(OUT / "acceleration_distribution.png", dpi=220)
    plt.close(fig)

    locations = summary.dropna(subset=["latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=locations, x="longitude", y="latitude", hue="class", palette=palette, s=55, ax=ax)
    ax.set(title="Median GPS position per recording", xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout()
    fig.savefig(OUT / "recording_locations.png", dpi=220)
    plt.close(fig)


def profile_raw_annotations() -> pd.DataFrame:
    """Read annotation files directly from the external Raw Data drive."""
    records = []
    if not RAW_DATA_DIR.exists():
        return pd.DataFrame(columns=["session", "seconds_elapsed", "event_type", "press_duration_ms"])
    for path in RAW_DATA_DIR.rglob("Annotation.csv"):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        if "text" not in frame:
            continue
        session = path.relative_to(RAW_DATA_DIR).parts[0]
        subset = pd.DataFrame(
            {
                "session": session,
                "seconds_elapsed": pd.to_numeric(frame.get("seconds_elapsed"), errors="coerce"),
                "event_type": frame["text"].astype(str).str.strip(),
                "press_duration_ms": pd.to_numeric(frame.get("millisecond_press_duration"), errors="coerce"),
            }
        )
        subset = subset[subset["event_type"].str.lower().isin(["bump", "pothole"])]
        records.append(subset)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["session", "seconds_elapsed", "event_type", "press_duration_ms"])


def plot_annotation_outputs(events: pd.DataFrame) -> None:
    if events.empty:
        return
    order = [value for value in ["Bump", "Pothole"] if value in events["event_type"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=events, x="event_type", order=order, color="#4D4D4D", ax=axes[0])
    axes[0].set(title="Annotated road events", xlabel="Event type", ylabel="Number of annotations")
    sns.boxplot(data=events, x="event_type", y="press_duration_ms", order=order, color="#B3B3B3", ax=axes[1])
    axes[1].set(title="Annotation button-press duration", xlabel="Event type", ylabel="Milliseconds")
    fig.tight_layout()
    fig.savefig(OUT / "annotation_event_eda.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    sns.countplot(data=events, x="event_type", order=order, color="#4D4D4D", ax=ax)
    ax.set(title="Annotated road events", xlabel="Event type", ylabel="Number of annotations")
    fig.tight_layout()
    fig.savefig(OUT / "annotation_event_types.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    sns.boxplot(data=events, x="event_type", y="press_duration_ms", order=order, color="#B3B3B3", ax=ax)
    ax.set(title="Annotation button-press duration", xlabel="Event type", ylabel="Milliseconds")
    fig.tight_layout()
    fig.savefig(OUT / "annotation_press_duration.png", dpi=220)
    plt.close(fig)


def create_video_link_index() -> pd.DataFrame:
    """Create a derived index; source video files remain on the external drive."""
    rows = []
    if not RAW_DATA_DIR.exists():
        return pd.DataFrame(columns=["session", "link_file", "google_drive_url"])
    for path in RAW_DATA_DIR.rglob("*.txt"):
        if "link" not in path.name.lower():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if "drive.google.com" in text:
            rows.append({"session": path.relative_to(RAW_DATA_DIR).parts[0], "link_file": str(path), "google_drive_url": text})
    index = pd.DataFrame(rows).sort_values("session") if rows else pd.DataFrame(columns=["session", "link_file", "google_drive_url"])
    index.to_csv(OUT / "video_link_index.csv", index=False)
    return index


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [(p, "Road anomaly") for p in sorted(ANOMALY_DIR.glob("*.csv"))]
    files += [(p, "Normal road") for p in sorted(NORMAL_DIR.glob("*.csv"))]
    if not files:
        raise FileNotFoundError("No extracted CSV recordings found. Extract the dataset first.")

    rows, samples = [], []
    for path, label in files:
        row, sample = analyse_recording(path, label)
        rows.append(row)
        samples.append(sample)
    summary = pd.DataFrame(rows)
    all_samples = pd.concat(samples, ignore_index=True)
    summary.to_csv(OUT / "recording_summary.csv", index=False)
    events = profile_raw_annotations()
    events.to_csv(OUT / "annotation_event_summary.csv", index=False)
    video_links = create_video_link_index()

    dataset_columns = pd.read_csv(files[0][0], nrows=0).columns.tolist()
    report = {
        "recordings": int(len(summary)),
        "rows": int(summary["rows"].sum()),
        "columns_per_recording": int(summary["columns"].iloc[0]),
        "class_counts": {str(k): int(v) for k, v in summary["class"].value_counts().items()},
        "median_sampling_hz": float(summary["estimated_sampling_hz"].median()),
        "median_duration_seconds": float(summary["duration_seconds"].median()),
        "coordinate_bounds": {
            "latitude_min": float(summary["latitude"].min()),
            "latitude_max": float(summary["latitude"].max()),
            "longitude_min": float(summary["longitude"].min()),
            "longitude_max": float(summary["longitude"].max()),
        },
        "available_columns": dataset_columns,
        "modalities_present": {
            "inertial": ACC + GYRO,
            "orientation_and_magnetometer": [c for c in dataset_columns if "orientation" in c or "magnetometer" in c],
            "geospatial": GEO,
            "weather": WEATHER,
            "image_or_video": False,
        },
        "raw_data_annotation_events": int(len(events)),
        "raw_data_event_counts": {str(k): int(v) for k, v in events["event_type"].value_counts().items()},
        "google_drive_video_link_files_found": int(len(video_links)),
        "raw_data_location": str(RAW_DATA_DIR),
        "class_labelling_note": "Class is derived from source archive membership; annotation_text is not used as a negative label.",
    }
    (OUT / "dataset_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_outputs(summary, all_samples)
    plot_annotation_outputs(events)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
