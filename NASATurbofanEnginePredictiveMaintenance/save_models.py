"""
AeroSentinel — Model Export Script
====================================
Re-runs the FD001 training pipeline exactly as v1.py does,
then saves:
  - models/rul_rf_fd001.joblib          Random Forest RUL
  - models/rul_gb_fd001.joblib          Gradient Boosting RUL
  - models/failure_rf_fd001.joblib      Random Forest Failure Classifier
  - models/failure_gb_fd001.joblib      Gradient Boosting Failure Classifier
  - models/scaler_fd001.joblib          StandardScaler (fitted on train features)
  - models/feature_cols_fd001.json      Ordered list of feature column names
  - models/model_metrics_fd001.json     Evaluation metrics (FD001 run)

This script does NOT change v1.py.
Run once:  python save_models.py
"""

import os, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              precision_score, recall_score, f1_score, roc_auc_score)
import joblib

os.makedirs("models", exist_ok=True)

# ── 1. Load ─────────────────────────────────────────────────────────────────
index_names  = ['unit', 'cycle']
setting_names = ['setting1', 'setting2', 'setting3']
sensor_names  = [f'sensor{i}' for i in range(1, 22)]
col_names     = index_names + setting_names + sensor_names

train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
test  = pd.read_csv('test_FD001.txt',  sep=r'\s+', header=None, names=col_names)
rul   = pd.read_csv('RUL_FD001.txt',   sep=r'\s+', header=None, names=['RUL'])

print(f"Loaded FD001 — train: {train.shape}, test: {test.shape}")

# ── 2. Feature engineering (identical to v1.py) ──────────────────────────────

def add_rul_target(df):
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop('max_cycle', axis=1)

def add_rolling_features(df, window_sizes=[5, 10]):
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    for w in window_sizes:
        for s in sensor_cols:
            df[f'{s}_rolling_mean_{w}'] = (df.groupby('unit')[s]
                .transform(lambda x: x.rolling(w, min_periods=1).mean()))
            df[f'{s}_rolling_std_{w}']  = (df.groupby('unit')[s]
                .transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0))
    return df

def add_degradation_features(df):
    sensor_cols = [c for c in df.columns if 'sensor' in c and 'rolling' not in c]
    max_cycle_per_unit = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle_per_unit.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycle_per_unit, on='unit', how='left')
    for s in sensor_cols:
        df[f'{s}_diff_from_start'] = df.groupby('unit')[s].transform(lambda x: x - x.iloc[0])
        df[f'{s}_normalized']      = df[s] / (df['max_cycle'] + 1)
    return df.drop('max_cycle', axis=1)

def engineer_features(df, window_sizes=[5, 10]):
    df = add_rul_target(df)
    df = add_rolling_features(df, window_sizes)
    df = add_degradation_features(df)
    df['failure_within_30'] = (df['RUL'] <= 30).astype(int)
    return df

# ── 3. Prepare ───────────────────────────────────────────────────────────────
train_p = engineer_features(train.copy())
test_p  = engineer_features(test.copy())

# Test: last cycle per engine + true RUL
test_last = test_p.groupby('unit')['cycle'].max().reset_index()
test_last.columns = ['unit', 'last_cycle']
test_last = test_last.sort_values('unit').reset_index(drop=True)
test_last['RUL_true'] = rul['RUL'].values

test_final = test_p.merge(test_last[['unit','last_cycle']], on='unit', how='left')
test_final = test_final[test_final['cycle'] == test_final['last_cycle']].copy()
test_final = test_final.drop('last_cycle', axis=1)
test_final = test_final.sort_values('unit').reset_index(drop=True)
test_final['RUL_true'] = rul['RUL'].values

exclude_cols  = ['unit', 'cycle', 'RUL', 'failure_within_30']
feature_cols  = [c for c in train_p.columns if c not in exclude_cols]

X_train = train_p[feature_cols].values
y_rul   = train_p['RUL'].values
y_fail  = train_p['failure_within_30'].values

X_test      = test_final[feature_cols].values
y_rul_test  = test_final['RUL_true'].values
y_fail_test = (test_final['RUL_true'] <= 30).astype(int).values

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

# ── 4. Train and save ────────────────────────────────────────────────────────
print("Training Random Forest RUL …")
rul_rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rul_rf.fit(X_tr_sc, y_rul)

print("Training Gradient Boosting RUL …")
rul_gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
rul_gb.fit(X_tr_sc, y_rul)

print("Training Random Forest Classifier …")
fail_rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
fail_rf.fit(X_tr_sc, y_fail)

print("Training Gradient Boosting Classifier …")
fail_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
fail_gb.fit(X_tr_sc, y_fail)

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
def reg_metrics(model, Xte, yte):
    p = model.predict(Xte)
    return dict(
        mae  = float(mean_absolute_error(yte, p)),
        rmse = float(np.sqrt(mean_squared_error(yte, p))),
        r2   = float(r2_score(yte, p)),
        predictions = p.tolist()
    )

def cls_metrics(model, Xte, yte):
    p     = model.predict(Xte)
    proba = model.predict_proba(Xte)[:,1]
    return dict(
        precision = float(precision_score(yte, p)),
        recall    = float(recall_score(yte, p)),
        f1        = float(f1_score(yte, p)),
        roc_auc   = float(roc_auc_score(yte, proba)),
        predictions      = p.tolist(),
        probabilities    = proba.tolist()
    )

metrics = {
    "dataset":       "FD001",
    "train_samples": int(len(X_train)),
    "test_samples":  int(len(X_test)),
    "feature_count": int(len(feature_cols)),
    "rul_rf":        reg_metrics(rul_rf,  X_te_sc, y_rul_test),
    "rul_gb":        reg_metrics(rul_gb,  X_te_sc, y_rul_test),
    "fail_rf":       cls_metrics(fail_rf, X_te_sc, y_fail_test),
    "fail_gb":       cls_metrics(fail_gb, X_te_sc, y_fail_test),
    "production_monitoring": {
        "rul_below_30":      int(np.sum(np.array(reg_metrics(rul_rf, X_te_sc, y_rul_test)['predictions']) < 30)),
        "rul_below_15":      int(np.sum(np.array(reg_metrics(rul_rf, X_te_sc, y_rul_test)['predictions']) < 15)),
        "fail_prob_above_07": int(np.sum(np.array(cls_metrics(fail_gb, X_te_sc, y_fail_test)['probabilities']) > 0.7))
    }
}

print("\n── Metrics ──")
print(f"RUL RF   → MAE={metrics['rul_rf']['mae']:.2f} RMSE={metrics['rul_rf']['rmse']:.2f} R²={metrics['rul_rf']['r2']:.3f}")
print(f"RUL GB   → MAE={metrics['rul_gb']['mae']:.2f} RMSE={metrics['rul_gb']['rmse']:.2f} R²={metrics['rul_gb']['r2']:.3f}")
print(f"Fail RF  → P={metrics['fail_rf']['precision']:.3f} R={metrics['fail_rf']['recall']:.3f} F1={metrics['fail_rf']['f1']:.3f} AUC={metrics['fail_rf']['roc_auc']:.3f}")
print(f"Fail GB  → P={metrics['fail_gb']['precision']:.3f} R={metrics['fail_gb']['recall']:.3f} F1={metrics['fail_gb']['f1']:.3f} AUC={metrics['fail_gb']['roc_auc']:.3f}")

# ── 6. Persist ───────────────────────────────────────────────────────────────
joblib.dump(rul_rf,   'models/rul_rf_fd001.joblib')
joblib.dump(rul_gb,   'models/rul_gb_fd001.joblib')
joblib.dump(fail_rf,  'models/failure_rf_fd001.joblib')
joblib.dump(fail_gb,  'models/failure_gb_fd001.joblib')
joblib.dump(scaler,   'models/scaler_fd001.joblib')

with open('models/feature_cols_fd001.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Remove large prediction arrays before saving metrics JSON
for k in ['rul_rf','rul_gb','fail_rf','fail_gb']:
    metrics[k].pop('predictions', None)
for k in ['fail_rf','fail_gb']:
    metrics[k].pop('probabilities', None)

with open('models/model_metrics_fd001.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ All artifacts saved to models/")
print("  rul_rf_fd001.joblib")
print("  rul_gb_fd001.joblib")
print("  failure_rf_fd001.joblib")
print("  failure_gb_fd001.joblib")
print("  scaler_fd001.joblib")
print("  feature_cols_fd001.json")
print("  model_metrics_fd001.json")
