"""Validate enhanced GeoFusion fusion comparison outputs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; PROCESSED=ROOT/'data'/'processed'; MODELS=ROOT/'models'
def require(value: bool, message: str) -> None:
    if not value: raise AssertionError(message)
    print(f'PASS  {message}')
def main() -> None:
    metrics=json.loads((MODELS/'enhanced_fusion_validation_metrics.json').read_text()); selection=json.loads((MODELS/'enhanced_fusion_selection.json').read_text()); predictions=pd.read_csv(PROCESSED/'enhanced_fusion_validation_predictions.csv')
    require(metrics['selection_split']=='validation_only','comparison is validation-only')
    require(selection['test_set_status']=='not_evaluated','test set remains untouched')
    require(selection['selected_method'] in {'early_clip_imu','late_yolo_imu','late_clip_imu'},'one enhanced fusion method is selected')
    require(len(predictions)==109 and predictions.event_id.is_unique,'all usable validation clips have predictions')
    require(predictions.filter(like='probability').apply(lambda x:x.between(0,1).all()).all(),'all probabilities are bounded')
    print('\nEnhanced fusion validation completed successfully.')
if __name__=='__main__':
    try: main()
    except (AssertionError,FileNotFoundError,KeyError,pd.errors.ParserError) as error: print(f'FAIL  {error}',file=sys.stderr); raise SystemExit(1)
