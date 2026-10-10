"""Validate GeoFusion Phase 4B fusion comparison artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    metrics = json.loads((MODELS / "fusion_validation_metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((MODELS / "fusion_selection.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(PROCESSED / "fusion_validation_predictions.csv")
    require(metrics["selection_split"] == "validation_only", "fusion selection used validation sessions only")
    require(selection["test_set_status"] == "not_evaluated_in_phase_4b", "test set remains untouched in this comparison")
    require(selection["selected_method"] in {"early_fusion", "late_fusion"}, "one pre-defined fusion method is selected")
    for method in ["early_fusion", "late_fusion"]:
        require(0 <= metrics[method]["f1"] <= 1, f"{method} validation F1 is valid")
        require(0 <= metrics[method]["average_precision"] <= 1, f"{method} validation average precision is valid")
    require(len(predictions) == 109 and predictions["event_id"].is_unique, "one validation prediction exists for each aligned row")
    require((MODELS / "early_fusion_validation_model.joblib").exists() and (MODELS / "late_fusion_validation_model.joblib").exists(), "both fitted fusion models are saved")
    print("\nPhase 4B validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        raise SystemExit(1)
