"""Compare early and late RGB--IMU fusion on frozen GeoFusion validation sessions.

No held-out test metrics are generated here.  The preferred fusion method is
selected by validation performance, then evaluated once in Phase 4C.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
EDA = ROOT / "data" / "eda"
IMU_FEATURES = [
    "imu_sampling_hz",
    "linear_acc_rms_deviation_mps2",
    "linear_acc_peak_deviation_mps2",
    "linear_acc_jerk_rms_mps3",
    "gyro_peak_radps",
]


def vision_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=20260923)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923)),
    ])


def imu_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923)),
    ])


def early_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=20260923)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260923)),
    ])


def metric_values(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(target)),
        "positive_rows": int(target.sum()),
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(PROCESSED / "imu_baseline_dataset.csv")
    archive = np.load(PROCESSED / "vision_hog_features.npz")
    visual = pd.DataFrame({"event_id": archive["event_id"].astype(str), "_order": np.arange(len(archive["event_id"]))})
    data["event_id"] = data["event_id"].astype(str)
    data = data.merge(visual, on="event_id", how="inner", validate="one_to_one").sort_values("_order").reset_index(drop=True)
    return data, archive["features"], data[IMU_FEATURES].to_numpy(dtype=float)


def main() -> None:
    data, visual_features, imu_features = load_data()
    train_mask = data["split"].eq("train").to_numpy()
    validation_mask = data["split"].eq("validation").to_numpy()
    y_train = data.loc[train_mask, "target_anomaly"].to_numpy(dtype=int)
    y_validation = data.loc[validation_mask, "target_anomaly"].to_numpy(dtype=int)
    visual_train, visual_validation = visual_features[train_mask], visual_features[validation_mask]
    imu_train, imu_validation = imu_features[train_mask], imu_features[validation_mask]
    early_train = np.hstack([visual_train, imu_train])
    early_validation = np.hstack([visual_validation, imu_validation])

    # M3: early fusion of image and IMU feature vectors.
    early = early_pipeline()
    early.fit(early_train, y_train)
    early_probability = early.predict_proba(early_validation)[:, 1]

    # M4: late fusion. Meta-learner sees only out-of-fold base predictions.
    oof = np.zeros((len(y_train), 2), dtype=float)
    groups = data.loc[train_mask, "session"].astype(str).to_numpy()
    for fit_index, holdout_index in GroupKFold(n_splits=5).split(visual_train, y_train, groups):
        vision_fold = vision_pipeline().fit(visual_train[fit_index], y_train[fit_index])
        imu_fold = imu_pipeline().fit(imu_train[fit_index], y_train[fit_index])
        oof[holdout_index, 0] = vision_fold.predict_proba(visual_train[holdout_index])[:, 1]
        oof[holdout_index, 1] = imu_fold.predict_proba(imu_train[holdout_index])[:, 1]
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260923).fit(oof, y_train)
    vision_full = vision_pipeline().fit(visual_train, y_train)
    imu_full = imu_pipeline().fit(imu_train, y_train)
    validation_base = np.column_stack([
        vision_full.predict_proba(visual_validation)[:, 1],
        imu_full.predict_proba(imu_validation)[:, 1],
    ])
    late_probability = meta.predict_proba(validation_base)[:, 1]

    metrics = {
        "selection_split": "validation_only",
        "early_fusion": metric_values(y_validation, early_probability),
        "late_fusion": metric_values(y_validation, late_probability),
    }
    # F1 is the pre-specified selection metric; average precision breaks an exact tie.
    selected = max(("early_fusion", "late_fusion"), key=lambda name: (metrics[name]["f1"], metrics[name]["average_precision"]))
    selection = {
        "selected_method": selected,
        "selection_metric": "validation_f1_then_average_precision",
        "test_set_status": "not_evaluated_in_phase_4b",
        "candidate_normal_limitation": "negative rows are temporally separated candidate-normal windows, not verified pristine road labels",
    }
    predictions = pd.DataFrame({
        "event_id": data.loc[validation_mask, "event_id"].to_numpy(),
        "session": data.loc[validation_mask, "session"].to_numpy(),
        "target_anomaly": y_validation,
        "early_fusion_probability": early_probability,
        "late_fusion_probability": late_probability,
    })
    MODELS.mkdir(parents=True, exist_ok=True)
    EDA.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": early, "feature_order": "hog_then_imu", "imu_features": IMU_FEATURES}, MODELS / "early_fusion_validation_model.joblib")
    joblib.dump({"vision_pipeline": vision_full, "imu_pipeline": imu_full, "meta_pipeline": meta, "imu_features": IMU_FEATURES}, MODELS / "late_fusion_validation_model.joblib")
    (MODELS / "fusion_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS / "fusion_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    predictions.to_csv(PROCESSED / "fusion_validation_predictions.csv", index=False)

    chart = pd.DataFrame({
        "method": ["Early fusion", "Late fusion"],
        "F1": [metrics["early_fusion"]["f1"], metrics["late_fusion"]["f1"]],
        "Average precision": [metrics["early_fusion"]["average_precision"], metrics["late_fusion"]["average_precision"]],
    }).melt(id_vars="method", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=chart, x="method", y="score", hue="metric", palette=["#355C7D", "#C06C84"], ax=ax)
    ax.set(title="Fusion comparison on validation sessions", xlabel="", ylabel="Score", ylim=(0, 1))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(EDA / "fusion_validation_comparison.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
