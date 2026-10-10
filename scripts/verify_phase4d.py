"""Verify the one-time final late-fusion evaluation outputs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/'models'; PROCESSED=ROOT/'data'/'processed'; EDA=ROOT/'data'/'eda'
def require(value: bool, message: str) -> None:
    if not value: raise AssertionError(message)
    print(f'PASS  {message}')
def main() -> None:
    metrics=json.loads((MODELS/'phase4d_final_late_yolo_imu_metrics.json').read_text())
    predictions=pd.read_csv(PROCESSED/'phase4d_final_late_yolo_imu_test_predictions.csv')
    require(metrics['method']=='late_yolo_imu','validation-selected late fusion is evaluated')
    require(metrics['test_set_status']=='evaluated_once','held-out test status is recorded')
    require(metrics['validation_comparison']['late_yolo_imu']['f1'] > metrics['validation_comparison']['early_clip_imu']['f1'],'late fusion beat early fusion on validation')
    require(len(predictions)==81 and predictions.event_id.is_unique,'one aligned prediction exists per test event/window')
    require(predictions.filter(like='probability').apply(lambda c:c.between(0,1).all()).all(),'all final probabilities are bounded')
    require(0 <= metrics['test']['f1'] <= 1 and 0 <= metrics['test']['average_precision'] <= 1,'final test metrics are valid')
    require((EDA/'phase4d_fusion_comparison_and_test.png').exists(),'early-late comparison and final-test chart exists')
    print('\nPhase 4D final evaluation completed successfully.')
if __name__=='__main__':
    try: main()
    except (AssertionError,FileNotFoundError,KeyError,pd.errors.ParserError) as error: print(f'FAIL  {error}',file=sys.stderr); raise SystemExit(1)
