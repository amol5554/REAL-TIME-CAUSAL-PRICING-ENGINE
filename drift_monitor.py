import pandas as pd, numpy as np, json, pickle
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping
import xgboost as xgb
 
def run_drift_check(reference_df, current_df, model, features, target='unit_price'):
    '''Compare current predictions vs reference distribution'''
 
    # Add predictions
    ref = reference_df.copy()
    cur = current_df.copy()
    ref['prediction'] = model.predict(ref[features])
    cur['prediction'] = model.predict(cur[features])
 
    column_mapping = ColumnMapping(
        target=target,
        prediction='prediction',
        numerical_features=features
    )
 
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(reference_data=ref, current_data=cur,
               column_mapping=column_mapping)
 
    results = report.as_dict()
    drift_detected = results['metrics'][0]['result']['dataset_drift']
    return drift_detected, results
 
def auto_retrain_if_needed(drift_detected, mae_threshold=15.0):
    if drift_detected:
        print('[ALERT] Data drift detected — triggering retrain...')
        # Reload fresh data and retrain
        df = pd.read_csv('data/retail_price.csv')
        # ... run engineer_features ...
        # ... retrain XGBoost ...
        # ... save new model ...
        print('[OK] Model retrained and saved')
        return True
    return False
 
if __name__ == '__main__':
    model    = pickle.load(open('models/xgb_model.pkl','rb'))
    meta     = json.load(open('models/model_meta.json'))
    features = meta['features']
 
    df = pd.read_csv('data/retail_price.csv')
    split = int(len(df)*0.8)
    ref   = df.iloc[:split]
    cur   = df.iloc[split:]
 
    drift, results = run_drift_check(ref, cur, model, features)
    print(f'Drift detected: {drift}')
    auto_retrain_if_needed(drift)
 
    # Save drift report
    import json
    with open('models/drift_report.json','w') as f:
        json.dump(results, f, default=str)
    print('Drift report saved to models/drift_report.json')
 
