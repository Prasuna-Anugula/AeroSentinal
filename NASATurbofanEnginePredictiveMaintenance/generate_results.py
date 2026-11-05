"""
AeroSentinel — Full Results Generation Script
==============================================
Generates all figures and measured metrics for the research report:

  Fig 2  — Dataset exploration & sensor distributions
  Fig 3  — Sensor degradation trends
  Fig 4  — Actual vs Predicted RUL (RF + GB + XGBoost)
  Fig 5  — Model performance comparison (RF, GB, XGBoost)
  Fig 6  — Anomaly detection results (Isolation Forest)
  Fig 7  — SHAP sensor contribution analysis
  Fig 8  — Machine health monitoring dashboard

  Table 3 — MAE / RMSE / R² for RF, GB, XGBoost
            (LSTM / GRU noted as out-of-scope for sklearn pipeline)

All metrics are measured from the FD001 test set using the
official RUL ground-truth file (RUL_FD001.txt).
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               IsolationForest)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import xgboost as xgb
import shap
import joblib

warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
BG   = '#070B16'
BG2  = '#0B1220'
SURF = '#111B2F'
PRI  = '#69C9FF'
SEC  = '#8B83FF'
ACC  = '#36E0C1'
SUC  = '#39D98A'
WRN  = '#FFBF69'
DNG  = '#FF647C'
TXT  = '#F4F7FF'
TXT2 = '#9AAAC2'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG2,
    'axes.edgecolor':   '#1E2D4A',
    'axes.labelcolor':  TXT2,
    'axes.titlecolor':  TXT,
    'xtick.color':      TXT2,
    'ytick.color':      TXT2,
    'text.color':       TXT,
    'grid.color':       '#1E2D4A',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

OUT = Path('results_figures')
OUT.mkdir(exist_ok=True)

# ── 1. Load & Engineer ────────────────────────────────────────────────────────
INDEX   = ['unit','cycle']
SETTINGS= ['setting1','setting2','setting3']
SENSORS = [f'sensor{i}' for i in range(1,22)]
COLS    = INDEX + SETTINGS + SENSORS

def load(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=COLS).sort_values(INDEX).reset_index(drop=True)

def add_rul(df):
    mc = df.groupby('unit')['cycle'].max().reset_index()
    mc.columns = ['unit','max_cycle']
    df = df.merge(mc, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop('max_cycle', axis=1)

def add_rolling(df, windows=[5,10]):
    cols = {}
    for w in windows:
        for s in SENSORS:
            cols[f'{s}_rolling_mean_{w}'] = df.groupby('unit')[s].transform(lambda x: x.rolling(w,min_periods=1).mean())
            cols[f'{s}_rolling_std_{w}']  = df.groupby('unit')[s].transform(lambda x: x.rolling(w,min_periods=1).std()).fillna(0)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

def add_deg(df):
    mc = df.groupby('unit')['cycle'].max().reset_index()
    mc.columns = ['unit','max_cycle']
    df = df.merge(mc, on='unit', how='left')
    cols = {}
    for s in SENSORS:
        cols[f'{s}_diff_from_start'] = df.groupby('unit')[s].transform(lambda x: x - x.iloc[0])
        cols[f'{s}_normalized']      = df[s] / (df['max_cycle'] + 1)
    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    return df.drop('max_cycle', axis=1)

def engineer(df):
    df = df.copy()
    df = add_rul(df)
    df = add_rolling(df)
    df = add_deg(df)
    df['failure_within_30'] = (df['RUL'] <= 30).astype(int)
    return df

print('Loading FD001 …')
train_raw = load('train_FD001.txt')
test_raw  = load('test_FD001.txt')
rul_true  = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])['RUL']

train_eng = engineer(train_raw)
test_eng  = engineer(test_raw)

EXCL   = {'unit','cycle','RUL','failure_within_30'}
FEAT   = [c for c in train_eng.columns if c not in EXCL]

# Test: last cycle per engine
last_idx   = test_eng.groupby('unit')['cycle'].idxmax()
test_final = test_eng.loc[last_idx].sort_values('unit').reset_index(drop=True)
test_final['RUL_true'] = rul_true.values

X_tr = train_eng[FEAT].values
y_tr = train_eng['RUL'].values
X_te = test_final[FEAT].values
y_te = test_final['RUL_true'].values

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

print(f'Train: {X_tr.shape}  Test: {X_te.shape}')

# ── 2. Train all models ───────────────────────────────────────────────────────
print('Training Random Forest …')
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_tr_sc, y_tr)

print('Training Gradient Boosting …')
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_tr_sc, y_tr)

print('Training XGBoost …')
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_tr_sc, y_tr)

# ── 3. Evaluate ───────────────────────────────────────────────────────────────
def metrics(model, X, y):
    p = np.maximum(model.predict(X), 0)
    return dict(
        mae  = float(mean_absolute_error(y, p)),
        rmse = float(np.sqrt(mean_squared_error(y, p))),
        r2   = float(r2_score(y, p)),
        preds= p
    )

rf_m  = metrics(rf,        X_te_sc, y_te)
gb_m  = metrics(gb,        X_te_sc, y_te)
xgb_m = metrics(xgb_model, X_te_sc, y_te)

print('\n══ TABLE 3 — MODEL PERFORMANCE ══')
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print('─'*50)
for name, m in [('Random Forest', rf_m), ('Gradient Boosting', gb_m), ('XGBoost', xgb_m)]:
    print(f"{name:<22} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
print('─'*50)
print('LSTM / GRU: not implemented in this sklearn pipeline.')
print('  Add tensorflow/keras and sequence modelling to extend Table 3.')

# Save metrics JSON
table3 = {
    'Random Forest':      {'MAE': round(rf_m['mae'],4),  'RMSE': round(rf_m['rmse'],4),  'R2': round(rf_m['r2'],4)},
    'Gradient Boosting':  {'MAE': round(gb_m['mae'],4),  'RMSE': round(gb_m['rmse'],4),  'R2': round(gb_m['r2'],4)},
    'XGBoost':            {'MAE': round(xgb_m['mae'],4), 'RMSE': round(xgb_m['rmse'],4), 'R2': round(xgb_m['r2'],4)},
    'LSTM':               {'MAE': 'N/A', 'RMSE': 'N/A', 'R2': 'N/A'},
    'GRU':                {'MAE': 'N/A', 'RMSE': 'N/A', 'R2': 'N/A'},
}
(OUT / 'table3_metrics.json').write_text(json.dumps(table3, indent=2))

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Dataset exploration & sensor distributions
# ═══════════════════════════════════════════════════════════════════════════════
print('\nFig 2 …')
DIAG_SENSORS = ['sensor2','sensor3','sensor4','sensor7','sensor11','sensor12','sensor15','sensor17','sensor20','sensor21']

fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle('Fig 2 — Dataset Exploration & Sensor Measurements\nFD001 Training Set  |  100 Engines  |  20,631 Observations',
             fontsize=13, color=TXT, y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# (a) Engine lifecycle histogram
ax0 = fig.add_subplot(gs[0, :2])
lc  = train_raw.groupby('unit')['cycle'].max()
ax0.hist(lc, bins=20, color=PRI, edgecolor=BG, alpha=0.85)
ax0.axvline(lc.mean(), color=WRN, lw=1.5, ls='--', label=f'Mean {lc.mean():.0f}')
ax0.axvline(lc.median(), color=ACC, lw=1.5, ls=':', label=f'Median {lc.median():.0f}')
ax0.set_xlabel('Engine Lifecycle (cycles)'); ax0.set_ylabel('Count')
ax0.set_title('(a) Engine Lifecycle Distribution', color=TXT)
ax0.legend(fontsize=8); ax0.grid(True)

# (b) True RUL distribution at test-set cutoff
ax1 = fig.add_subplot(gs[0, 2:])
ax1.hist(rul_true, bins=20, color=ACC, edgecolor=BG, alpha=0.85)
ax1.axvline(rul_true.mean(), color=WRN, lw=1.5, ls='--', label=f'Mean {rul_true.mean():.0f}')
ax1.set_xlabel('True RUL at Test Cutoff (cycles)'); ax1.set_ylabel('Count')
ax1.set_title('(b) Test-Set True RUL Distribution', color=TXT)
ax1.legend(fontsize=8); ax1.grid(True)

# (c-l) Sensor violin plots
positions = [(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]
for i, (s, pos) in enumerate(zip(DIAG_SENSORS[:8], positions)):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    data = train_raw[s].dropna()
    ax.hist(data, bins=25, color=SEC if i%2==0 else PRI, edgecolor=BG, alpha=0.8)
    ax.set_title(f'({chr(99+i)}) {s}', color=TXT, fontsize=9)
    ax.set_xlabel('Value', fontsize=7); ax.set_ylabel('Count', fontsize=7)
    ax.tick_params(labelsize=7); ax.grid(True)

plt.savefig(OUT / 'fig2_dataset_exploration.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig2_dataset_exploration.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Sensor degradation trends
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 3 …')
TREND_SENSORS = ['sensor2','sensor3','sensor4','sensor11','sensor12','sensor15']
SAMPLE_ENGINES = [1, 5, 10, 20, 50]
COLORS_TREND   = [PRI, ACC, WRN, SEC, DNG]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig.suptitle('Fig 3 — Sensor Degradation Trends Over Operating Cycles\nFD001 Training Set — 5 Sample Engines',
             fontsize=13, color=TXT, y=1.01)
axes = axes.flatten()

for ax, sensor in zip(axes, TREND_SENSORS):
    for eng, col in zip(SAMPLE_ENGINES, COLORS_TREND):
        edf = train_raw[train_raw['unit'] == eng]
        ax.plot(edf['cycle'], edf[sensor], color=col, lw=1.3, alpha=0.85, label=f'Eng {eng}')
    ax.set_title(sensor, color=TXT)
    ax.set_xlabel('Cycle'); ax.set_ylabel('Sensor Value')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True)
    ax.set_facecolor(BG2)

plt.tight_layout()
plt.savefig(OUT / 'fig3_sensor_degradation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig3_sensor_degradation.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Actual vs Predicted RUL
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 4 …')
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle('Fig 4 — Actual versus Predicted Remaining Useful Life\nFD001 Test Set  |  100 Engines  |  Official Ground Truth',
             fontsize=13, color=TXT)

model_results = [
    ('Random Forest',     rf_m,  PRI),
    ('Gradient Boosting', gb_m,  ACC),
    ('XGBoost',           xgb_m, SEC),
]

for ax, (name, m, col) in zip(axes, model_results):
    ax.set_facecolor(BG2)
    ax.scatter(y_te, m['preds'], color=col, alpha=0.65, s=40, edgecolors='none')
    lim = max(y_te.max(), m['preds'].max()) + 10
    ax.plot([0, lim], [0, lim], color=WRN, lw=1.5, ls='--', label='Perfect prediction')
    ax.set_xlabel('True RUL (cycles)', color=TXT2)
    ax.set_ylabel('Predicted RUL (cycles)', color=TXT2)
    ax.set_title(f'{name}\nMAE={m["mae"]:.2f}  RMSE={m["rmse"]:.2f}  R²={m["r2"]:.3f}', color=TXT)
    ax.legend(fontsize=8)
    ax.grid(True)

plt.tight_layout()
plt.savefig(OUT / 'fig4_actual_vs_predicted.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig4_actual_vs_predicted.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Model performance comparison
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 5 …')
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle('Fig 5 — Model Performance Comparison\nFD001 Test Set — Random Forest · Gradient Boosting · XGBoost',
             fontsize=13, color=TXT)

model_names = ['Random\nForest', 'Gradient\nBoosting', 'XGBoost']
mae_vals  = [rf_m['mae'],  gb_m['mae'],  xgb_m['mae']]
rmse_vals = [rf_m['rmse'], gb_m['rmse'], xgb_m['rmse']]
r2_vals   = [rf_m['r2'],   gb_m['r2'],   xgb_m['r2']]
bar_cols   = [PRI, ACC, SEC]

for ax, (title, vals, best_fn) in zip(axes, [
    ('MAE (lower is better)',  mae_vals,  min),
    ('RMSE (lower is better)', rmse_vals, min),
    ('R² (higher is better)',  r2_vals,   max),
]):
    ax.set_facecolor(BG2)
    best = best_fn(vals)
    colors = [WRN if v == best else c for v, c in zip(vals, bar_cols)]
    bars = ax.bar(model_names, vals, color=colors, edgecolor=BG, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, color=TXT)
    ax.set_title(title, color=TXT)
    ax.set_ylabel(title.split(' ')[0], color=TXT2)
    ax.grid(True, axis='y')

axes[0].text(0.5, -0.18, '★ Gold = best performer', transform=axes[0].transAxes,
             ha='center', fontsize=8, color=WRN)
plt.tight_layout()
plt.savefig(OUT / 'fig5_model_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig5_model_comparison.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Anomaly detection results
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 6 …')
# Train Isolation Forest on all training sensor readings
X_anom = train_raw[SENSORS].values
sc_anom = StandardScaler()
X_anom_sc = sc_anom.fit_transform(X_anom)

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_anom_sc)

# Score all test observations (use a single engine's full trajectory for Fig 6b)
ENGINE_DEMO = 42  # known critical engine
edf_demo = test_raw[test_raw['unit'] == ENGINE_DEMO].copy()
X_demo   = edf_demo[SENSORS].values
X_demo_sc= sc_anom.transform(X_demo)
scores_demo  = iso.decision_function(X_demo_sc)
labels_demo  = iso.predict(X_demo_sc)

# Score all test-set last cycles for fleet view
X_fleet  = test_final[SENSORS].values
X_fleet_sc= sc_anom.transform(X_fleet)
scores_fleet = iso.decision_function(X_fleet_sc)
labels_fleet = iso.predict(X_fleet_sc)

n_anomaly = int(np.sum(labels_fleet == -1))
thr_usual    = -0.10
thr_high     = -0.20

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle('Fig 6 — Anomaly Detection Results  |  Isolation Forest  |  FD001\n'
             f'Fleet: {n_anomaly}/100 engines flagged at last cycle  |  Contamination = 5%',
             fontsize=13, color=TXT)

# (a) Fleet anomaly score distribution
ax = axes[0]; ax.set_facecolor(BG2)
ax.hist(scores_fleet[labels_fleet==1],  bins=20, color=SUC, alpha=0.7, label='Normal')
ax.hist(scores_fleet[labels_fleet==-1], bins=15, color=DNG, alpha=0.8, label='Anomaly')
ax.axvline(thr_usual, color=WRN, lw=1.5, ls='--', label='Unusual threshold')
ax.axvline(thr_high,  color=DNG, lw=1.5, ls=':',  label='High-anomaly threshold')
ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Engine Count')
ax.set_title('(a) Fleet Anomaly Score Distribution', color=TXT)
ax.legend(fontsize=8); ax.grid(True)

# (b) Engine 42 anomaly score over cycles
ax = axes[1]; ax.set_facecolor(BG2)
normal_mask = labels_demo == 1
ax.plot(edf_demo['cycle'][normal_mask],  scores_demo[normal_mask],  color=SUC, lw=1.2, label='Normal')
ax.scatter(edf_demo['cycle'][~normal_mask], scores_demo[~normal_mask], color=DNG, s=30, zorder=5, label='Anomaly')
ax.axhline(thr_usual, color=WRN, lw=1.2, ls='--', label='Unusual')
ax.axhline(thr_high,  color=DNG, lw=1.2, ls=':',  label='High anomaly')
ax.set_xlabel('Cycle'); ax.set_ylabel('Anomaly Score')
ax.set_title(f'(b) Engine {ENGINE_DEMO} — Anomaly Score Timeline', color=TXT)
ax.legend(fontsize=8); ax.grid(True)

# (c) Score vs RUL scatter (all fleet engines)
ax = axes[2]; ax.set_facecolor(BG2)
sc = ax.scatter(y_te, scores_fleet, c=scores_fleet, cmap='RdYlGn', s=40, alpha=0.8, vmin=-0.3, vmax=0.2)
ax.axhline(thr_usual, color=WRN, lw=1.2, ls='--', label='Unusual')
ax.axhline(thr_high,  color=DNG, lw=1.2, ls=':',  label='High anomaly')
ax.set_xlabel('True RUL (cycles)'); ax.set_ylabel('Anomaly Score')
ax.set_title('(c) Anomaly Score vs True RUL', color=TXT)
ax.legend(fontsize=8); ax.grid(True)
plt.colorbar(sc, ax=ax, label='Score')

plt.tight_layout()
plt.savefig(OUT / 'fig6_anomaly_detection.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'  Saved fig6_anomaly_detection.png  (anomalies={n_anomaly}/100)')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — SHAP sensor contribution analysis
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 7 — SHAP (this may take a minute) …')

# Use a fast TreeExplainer on the RF model
explainer    = shap.TreeExplainer(rf)
# Use only the 100 test-set samples (one per engine) for SHAP — fast and representative
shap_values  = explainer.shap_values(X_te_sc)   # shape (100, 150)

# Aggregate importance by BASE sensor name
sensor_shap = {}
for j, feat in enumerate(FEAT):
    base = next((s for s in SENSORS if feat.startswith(s)), None)
    if base:
        sensor_shap[base] = sensor_shap.get(base, 0) + np.abs(shap_values[:, j]).mean()

sensor_shap_sorted = sorted(sensor_shap.items(), key=lambda x: x[1], reverse=True)
snames = [x[0] for x in sensor_shap_sorted]
svals  = [x[1] for x in sensor_shap_sorted]

# Top 20 individual features by mean |SHAP|
feat_imp_shap = np.abs(shap_values).mean(axis=0)
top20_idx     = np.argsort(feat_imp_shap)[::-1][:20]
top20_names   = [FEAT[i] for i in top20_idx]
top20_vals    = feat_imp_shap[top20_idx]

fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
fig.suptitle('Fig 7 — SHAP Sensor Contribution Analysis\nRandom Forest · FD001 Test Set (100 engines)  |  Mean |SHAP| value',
             fontsize=13, color=TXT)

# (a) Aggregated by sensor
ax = axes[0]; ax.set_facecolor(BG2)
colors_shap = [PRI if v > np.median(svals) else SEC for v in svals]
ax.barh(snames[::-1], svals[::-1], color=colors_shap[::-1], edgecolor=BG)
ax.set_xlabel('Mean |SHAP value| (aggregated across features)')
ax.set_title('(a) Sensor-Level Contribution Ranking', color=TXT)
ax.grid(True, axis='x')

# (b) Top 20 individual features
ax = axes[1]; ax.set_facecolor(BG2)
bar_c = [PRI if i < 5 else SEC if i < 10 else TXT2 for i in range(20)]
ax.barh(range(20), top20_vals[::-1], color=bar_c[::-1], edgecolor=BG)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_names[::-1], fontsize=8)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('(b) Top 20 Feature-Level SHAP Contributions', color=TXT)
ax.grid(True, axis='x')

plt.tight_layout()
plt.savefig(OUT / 'fig7_shap_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig7_shap_analysis.png')

# Save SHAP sensor ranking
shap_ranking = {s: round(v, 6) for s, v in sensor_shap_sorted}
(OUT / 'shap_sensor_ranking.json').write_text(json.dumps(shap_ranking, indent=2))

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Machine health monitoring dashboard
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 8 …')

# Snapshot of 6 representative engines
DEMO_ENGINES = [1, 10, 23, 42, 67, 88]
rf_preds_all  = np.maximum(rf.predict(X_te_sc), 0)
iso_scores_all= iso.decision_function(X_fleet_sc)
iso_labels_all= iso.predict(X_fleet_sc)

# Per-engine SHAP top sensor
top_sensor_per_engine = []
for i in range(len(test_final)):
    unit_shap  = shap_values[i]
    top_feat_i = FEAT[np.argmax(np.abs(unit_shap))]
    base       = next((s for s in SENSORS if top_feat_i.startswith(s)), top_feat_i)
    top_sensor_per_engine.append(base)

unit_list = test_final['unit'].tolist()

def health_status(rul):
    if rul > 100: return ('Healthy',   SUC)
    if rul > 50:  return ('Warning',   WRN)
    if rul > 20:  return ('High Risk', '#FB923C')
    return ('Critical', DNG)

def anom_status(score):
    if score < -0.20: return ('High Anomaly', DNG)
    if score < -0.10: return ('Unusual',      WRN)
    return ('Normal', SUC)

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle('Fig 8 — Machine Health Monitoring Dashboard\nAeroSentinel  |  FD001 Test Set  |  Random Forest + Isolation Forest',
             fontsize=14, color=TXT, y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

for panel_idx, eng_unit in enumerate(DEMO_ENGINES):
    row, col = divmod(panel_idx, 3)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor(SURF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    unit_idx  = unit_list.index(eng_unit)
    pred_rul  = float(rf_preds_all[unit_idx])
    true_rul  = float(y_te[unit_idx])
    a_score   = float(iso_scores_all[unit_idx])
    h_label, h_col  = health_status(pred_rul)
    a_label, a_col  = anom_status(a_score)
    top_s     = top_sensor_per_engine[unit_idx]
    last_cyc  = int(test_final['cycle'].iloc[unit_idx])

    # Engine header
    ax.text(0.5, 0.94, f'ENGINE {eng_unit:03d}', ha='center', va='top',
            fontsize=11, fontweight='bold', color=TXT,
            transform=ax.transAxes)

    # RUL gauge arc (simple semicircle)
    max_rul = 200
    frac    = min(pred_rul / max_rul, 1.0)
    theta   = np.linspace(np.pi, np.pi - frac * np.pi, 100)
    xs = 0.5 + 0.22 * np.cos(theta)
    ys = 0.62 + 0.20 * np.sin(theta)
    ax.plot([0.5 + 0.22*np.cos(np.pi), 0.5 + 0.22*np.cos(0)],
            [0.62, 0.62], lw=2, color='#1E2D4A', transform=ax.transAxes)
    ax.plot([0.28, 0.72], [0.62, 0.62], lw=6, color='#1E2D4A',
            solid_capstyle='round', transform=ax.transAxes)
    ax.plot(xs, ys, lw=6, color=h_col, solid_capstyle='round', transform=ax.transAxes)
    ax.text(0.5, 0.55, f'{pred_rul:.0f}', ha='center', va='center',
            fontsize=14, fontweight='bold', color=h_col, transform=ax.transAxes)
    ax.text(0.5, 0.47, 'cycles RUL', ha='center', va='center',
            fontsize=7, color=TXT2, transform=ax.transAxes)

    # Status badges
    for bx, (lbl, col) in enumerate([(h_label, h_col), (a_label, a_col)]):
        rect = FancyBboxPatch((0.05 + bx*0.5, 0.33), 0.42, 0.1,
                               boxstyle='round,pad=0.02', transform=ax.transAxes,
                               facecolor=col+'25', edgecolor=col, lw=0.8)
        ax.add_patch(rect)
        ax.text(0.26 + bx*0.5, 0.385, lbl, ha='center', va='center',
                fontsize=7, color=col, transform=ax.transAxes)

    # Info rows
    infos = [
        ('Last Cycle',   f'{last_cyc}'),
        ('Anom Score',   f'{a_score:.3f}'),
        ('Top Sensor',   top_s),
        ('Classifier',   f'{"FAIL" if pred_rul<30 else "OK"}'),
    ]
    for ii, (lbl, val) in enumerate(infos):
        y_pos = 0.26 - ii * 0.07
        ax.text(0.05, y_pos, lbl+':', fontsize=7, color=TXT2, transform=ax.transAxes)
        ax.text(0.95, y_pos, val, fontsize=7, color=PRI if ii<2 else ACC,
                ha='right', transform=ax.transAxes, fontweight='bold')

plt.savefig(OUT / 'fig8_health_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig8_health_dashboard.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print('ALL RESULTS GENERATED')
print('═'*60)
print(f'\nOutput directory: {OUT.resolve()}\n')
print('FIGURES:')
for f in sorted(OUT.glob('*.png')): print(f'  {f.name}')
print('\nDATA FILES:')
for f in sorted(OUT.glob('*.json')): print(f'  {f.name}')

print('\n══ TABLE 3 — FINAL MEASURED VALUES ══')
print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print('─'*55)
for name, m in [('Random Forest',rf_m),('Gradient Boosting',gb_m),('XGBoost',xgb_m)]:
    print(f"{name:<22} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f}")
print(f"{'LSTM':<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
print(f"{'GRU':<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
print('\nAnomaly Detection (Isolation Forest, contamination=5%):')
print(f'  Engines flagged at test-set last cycle: {n_anomaly}/100')
print(f'  Threshold Unusual:     score < -0.10')
print(f'  Threshold High Anomaly: score < -0.20')

print('\nSHAP Top-5 Sensors (by aggregated mean |SHAP|):')
for s, v in sensor_shap_sorted[:5]: print(f'  {s}: {v:.5f}')
