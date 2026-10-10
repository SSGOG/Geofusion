"""Extract aligned frame HOG features and train the GeoFusion vision baseline."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
EDA = ROOT / "data" / "eda"
FEATURES_PATH = PROCESSED / "vision_hog_features.npz"
METRICS_PATH = MODELS / "vision_hog_baseline_metrics.json"
HOG = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)


def read_hog(video_path: str, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise RuntimeError(f"Unable to decode frame at {seconds:.3f}s from {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return HOG.compute(resized).reshape(-1).astype(np.float32)


def metrics_for(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    target = frame["target_anomaly"].to_numpy(dtype=int)
    prediction = (probability >= 0.5).astype(int)
    result: dict[str, float | int] = {
        "rows": int(len(frame)), "positive_rows": int(target.sum()),
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(target, probability)),
    }
    result["roc_auc"] = float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else float("nan")
    return result


def main() -> None:
    manifest = pd.read_csv(PROCESSED / "vision_baseline_manifest.csv")
    # Parallel frame decoding keeps the initial CPU baseline practical with the
    # supplied 4K MP4 recordings. The returned feature order still matches the
    # manifest exactly.
    features: list[np.ndarray | None] = [None] * len(manifest)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(read_hog, str(row.video_path), float(row.event_seconds)): index
            for index, row in enumerate(manifest.itertuples(index=False))
        }
        completed = 0
        for future in as_completed(futures):
            features[futures[future]] = future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"Decoded {completed}/{len(futures)} frames")
    matrix = np.vstack(features)
    np.savez_compressed(FEATURES_PATH, event_id=manifest["event_id"].to_numpy(dtype=str), features=matrix)

    train_index = manifest["split"].eq("train").to_numpy()
    validation_index = manifest["split"].eq("validation").to_numpy()
    test_index = manifest["split"].eq("test").to_numpy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=20260923)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923)),
    ])
    model.fit(matrix[train_index], manifest.loc[train_index, "target_anomaly"])
    metrics, predictions = {}, {}
    for name, index in {"train": train_index, "validation": validation_index, "test": test_index}.items():
        probability = model.predict_proba(matrix[index])[:, 1]
        metrics[name] = metrics_for(manifest.loc[index], probability)
        predictions[name] = probability
    MODELS.mkdir(parents=True, exist_ok=True)
    EDA.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "feature_type": "hog_128x128", "target": "annotated_event_vs_candidate_normal"}, MODELS / "vision_hog_baseline.joblib")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(manifest.loc[test_index, "target_anomaly"], (predictions["test"] >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, xticklabels=["Candidate normal", "Annotated event"], yticklabels=["Candidate normal", "Annotated event"], ax=ax)
    ax.set(title="Vision HOG baseline: held-out sessions", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(EDA / "vision_hog_baseline_test_confusion_matrix.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print("Important: this is a single-frame classical vision baseline; the final model will use short clips.")


if __name__ == "__main__":
    main()
