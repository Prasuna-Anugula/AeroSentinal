"""
AeroSentinel — LSTM Training on FD001
======================================
Sequence-to-one LSTM:
  Input  : sliding window of W=30 cycles × 14 selected sensors (normalised)
  Output : RUL at the last cycle of the window

Pipeline:
  1. Load + preprocess FD001
  2. Build sliding-window sequences (train)
  3. Build last-window sequences (test) + merge ground-truth RUL
  4. Train LSTM (2 layers, dropout, early stopping)
  5. Evaluate: MAE, RMSE, R²
  6. Generate Fig 4b (actual vs predicted), Fig 5 update, training curve
  7. Save model + update table3_metrics.json
"""

import os, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

OUT = Path('results_figures')
OUT.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
BG   = '#070B16';  BG2  = '#0B1220';  SURF = '#111B2F'
PRI  = '#69C9FF';  SEC  = '#8B83FF';  ACC  = '#36E0C1'
WRN  = '#FFBF69';  DNG  = '#FF647C';  SUC  = '#39D98A'
TXT  = '#F4F7FF';  TXT2 = '#9AAAC2'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': BG2,
    'axes.edgecolor': '#1E2D4A', 'axes.labelcolor': TXT2,
    'axes.titlecolor': TXT, 'xtick.color': TXT2,
    'ytick.color': TXT2, 'text.color': TXT,
    'grid.color': '#1E2D4A', 'grid.alpha': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WINDOW      = 30        # sequence length (cycles)
RUL_CAP     = 125       # piece-wise linear cap — standard CMAPSS practice
BATCH_SIZE  = 128
EPOCHS      = 25
LR          = 1e-3

# 14 informative sensors (low-variance sensors dropped — standard practice)
SELECTED_SENSORS = [
    'sensor2','sensor3','sensor4','sensor7','sensor8',
    'sensor9','sensor11','sensor12','sensor13','sensor14',
    'sensor15','sensor17','sensor20','sensor21'
]
N_FEATURES = len(SELECTED_SENSORS)   # 14

INDEX    = ['unit','cycle']
SETTINGS = ['setting1','setting2','setting3']
ALL_SENS = [f'sensor{i}' for i in range(1,22)]
COLS     = INDEX + SETTINGS + ALL_SENS

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading FD001 …")
train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=COLS)
test_df  = pd.read_csv('test_FD001.txt',  sep=r'\s+', header=None, names=COLS)
rul_file = pd.read_csv('RUL_FD001.txt',   sep=r'\s+', header=None, names=['RUL'])['RUL']

train_df = train_df.sort_values(INDEX).reset_index(drop=True)
test_df  = test_df.sort_values(INDEX).reset_index(drop=True)

print(f"  Train: {train_df.shape}  Test: {test_df.shape}  RUL entries: {len(rul_file)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ADD PIECEWISE-CAPPED RUL TARGET
# ─────────────────────────────────────────────────────────────────────────────
def add_rul(df, cap=RUL_CAP):
    mc = df.groupby('unit')['cycle'].max().reset_index()
    mc.columns = ['unit','max_cycle']
    df = df.merge(mc, on='unit', how='left')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=cap)
    return df.drop('max_cycle', axis=1)

train_df = add_rul(train_df)

# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALISE sensors (MinMax fitted on training set only)
# ─────────────────────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
train_df[SELECTED_SENSORS] = scaler.fit_transform(train_df[SELECTED_SENSORS])
test_df[SELECTED_SENSORS]  = scaler.transform(test_df[SELECTED_SENSORS])

# ─────────────────────────────────────────────────────────────────────────────
# 5. SLIDING WINDOW SEQUENCES (training)
# ─────────────────────────────────────────────────────────────────────────────
def make_sequences(df, window, features):
    X_list, y_list = [], []
    for unit, grp in df.groupby('unit'):
        data = grp[features].values        # (cycles, 14)
        rul  = grp['RUL'].values           # (cycles,)
        n    = len(grp)
        if n < window:
            # pad with first row if engine shorter than window
            pad   = np.repeat(data[[0]], window - n, axis=0)
            data  = np.vstack([pad, data])
            rul   = np.concatenate([np.full(window - n, rul[0]), rul])
            n     = window
        for i in range(n - window + 1):
            X_list.append(data[i : i + window])
            y_list.append(rul[i + window - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

print("Building training sequences …")
X_train, y_train = make_sequences(train_df, WINDOW, SELECTED_SENSORS)
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TEST SEQUENCES — last window per engine
# ─────────────────────────────────────────────────────────────────────────────
def make_test_sequences(df, window, features):
    X_list = []
    for unit, grp in df.groupby('unit'):
        data = grp[features].values
        n    = len(grp)
        if n < window:
            pad  = np.repeat(data[[0]], window - n, axis=0)
            data = np.vstack([pad, data])
        X_list.append(data[-window:])   # last W cycles
    return np.array(X_list, dtype=np.float32)

print("Building test sequences …")
X_test   = make_test_sequences(test_df, WINDOW, SELECTED_SENSORS)
y_test   = rul_file.values.astype(np.float32)   # official ground truth (uncapped)
print(f"  X_test: {X_test.shape}  y_test: {y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding LSTM model …")

model = Sequential([
    Input(shape=(WINDOW, N_FEATURES)),
    LSTM(128, return_sequences=True, activation='tanh',
         recurrent_dropout=0.0),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(LR), loss='mse', metrics=['mae'])
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
]

print("\nTraining …")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
y_pred_raw = model.predict(X_test, verbose=0).flatten()
y_pred     = np.maximum(y_pred_raw, 0)   # no negative RUL

mae  = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))

print('\n' + '═'*55)
print('LSTM RESULTS — FD001 Test Set (100 engines)')
print('─'*55)
print(f'  MAE  : {mae:.4f} cycles')
print(f'  RMSE : {rmse:.4f} cycles')
print(f'  R²   : {r2:.4f}')
print(f'  Window  : {WINDOW} cycles')
print(f'  Features: {N_FEATURES} sensors')
print(f'  RUL cap : {RUL_CAP} cycles (train only)')
print(f'  Epochs  : {len(history.history["loss"])} (early stop)')
print('═'*55)

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE MODEL + UPDATE METRICS JSON
# ─────────────────────────────────────────────────────────────────────────────
model.save('models/lstm_fd001.keras')
print("Model saved: models/lstm_fd001.keras")

metrics_path = Path('models/model_metrics_fd001.json')
if metrics_path.exists():
    all_metrics = json.loads(metrics_path.read_text())
else:
    all_metrics = {}

all_metrics['lstm'] = {
    'mae':          round(mae,  4),
    'rmse':         round(rmse, 4),
    'r2':           round(r2,   4),
    'window':       WINDOW,
    'n_features':   N_FEATURES,
    'rul_cap':      RUL_CAP,
    'epochs_run':   len(history.history['loss']),
}
metrics_path.write_text(json.dumps(all_metrics, indent=2))
print("Updated models/model_metrics_fd001.json")

# Update table3
t3_path = OUT / 'table3_metrics.json'
if t3_path.exists():
    t3 = json.loads(t3_path.read_text())
else:
    t3 = {}
t3['LSTM'] = {'MAE': round(mae,4), 'RMSE': round(rmse,4), 'R2': round(r2,4)}
t3_path.write_text(json.dumps(t3, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# 11. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# ── Fig: Training curve ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('LSTM Training History — FD001\nWindow=30 · 14 sensors · Piecewise RUL cap=125',
             fontsize=12, color=TXT)

ax = axes[0]; ax.set_facecolor(BG2)
ax.plot(history.history['loss'],     color=PRI, lw=1.8, label='Train Loss (MSE)')
ax.plot(history.history['val_loss'], color=WRN, lw=1.8, ls='--', label='Val Loss (MSE)')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
ax.set_title('Loss Curve', color=TXT); ax.legend(fontsize=9); ax.grid(True)

ax = axes[1]; ax.set_facecolor(BG2)
ax.plot(history.history['mae'],     color=ACC, lw=1.8, label='Train MAE')
ax.plot(history.history['val_mae'], color=DNG, lw=1.8, ls='--', label='Val MAE')
ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (cycles)')
ax.set_title('MAE Curve', color=TXT); ax.legend(fontsize=9); ax.grid(True)

plt.tight_layout()
plt.savefig(OUT / 'fig_lstm_training_curve.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 4 updated — Actual vs Predicted (LSTM) ────────────────────────────────
# Load existing RF predictions for comparison
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler as SS
import joblib

# Quick RF for comparison (reload saved model)
rf_path = Path('models/rul_rf_fd001.joblib')
if rf_path.exists():
    rf = joblib.load(rf_path)
    sc = joblib.load('models/scaler_fd001.joblib')
    with open('models/feature_cols_fd001.json') as f:
        feat_cols = json.load(f)

    # rebuild RF test set
    from sklearn.preprocessing import StandardScaler
    import warnings; warnings.filterwarnings('ignore')
    ALL_SENS_F = [f'sensor{i}' for i in range(1,22)]
    COLS_F     = ['unit','cycle','setting1','setting2','setting3'] + ALL_SENS_F

    def add_rul_lin(df):
        mc = df.groupby('unit')['cycle'].max().reset_index()
        mc.columns = ['unit','max_cycle']
        df = df.merge(mc, on='unit', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        return df.drop('max_cycle', axis=1)

    def add_rolling_f(df, ws=[5,10]):
        cols = {}
        for w in ws:
            for s in ALL_SENS_F:
                cols[f'{s}_rolling_mean_{w}'] = df.groupby('unit')[s].transform(lambda x: x.rolling(w,min_periods=1).mean())
                cols[f'{s}_rolling_std_{w}']  = df.groupby('unit')[s].transform(lambda x: x.rolling(w,min_periods=1).std()).fillna(0)
        return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

    def add_deg_f(df):
        mc = df.groupby('unit')['cycle'].max().reset_index()
        mc.columns = ['unit','max_cycle']
        df = df.merge(mc, on='unit', how='left')
        cols = {}
        for s in ALL_SENS_F:
            cols[f'{s}_diff_from_start'] = df.groupby('unit')[s].transform(lambda x: x - x.iloc[0])
            cols[f'{s}_normalized']      = df[s] / (df['max_cycle'] + 1)
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        return df.drop('max_cycle', axis=1)

    test_raw2   = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=COLS_F)
    test_raw2   = test_raw2.sort_values(['unit','cycle']).reset_index(drop=True)
    test_eng2   = add_rul_lin(test_raw2)
    test_eng2   = add_rolling_f(test_eng2)
    test_eng2   = add_deg_f(test_eng2)
    last_idx2   = test_eng2.groupby('unit')['cycle'].idxmax()
    test_fin2   = test_eng2.loc[last_idx2].sort_values('unit').reset_index(drop=True)
    X_te2       = sc.transform(test_fin2[feat_cols].values)
    rf_preds    = np.maximum(rf.predict(X_te2), 0)
else:
    rf_preds = None

fig, axes = plt.subplots(1, 2 if rf_preds is None else 2, figsize=(14, 6), facecolor=BG)
fig.suptitle('Fig 4b — LSTM: Actual versus Predicted RUL\nFD001 Test Set · 100 Engines · Official Ground Truth',
             fontsize=12, color=TXT)

# LSTM scatter
ax = axes[0]; ax.set_facecolor(BG2)
ax.scatter(y_test, y_pred, color=SEC, alpha=0.7, s=45, edgecolors='none', label='LSTM')
lim = max(y_test.max(), y_pred.max()) + 10
ax.plot([0, lim], [0, lim], color=WRN, lw=1.5, ls='--', label='Perfect prediction')
ax.set_xlabel('True RUL (cycles)'); ax.set_ylabel('Predicted RUL (cycles)')
ax.set_title(f'LSTM\nMAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}', color=TXT)
ax.legend(fontsize=9); ax.grid(True)

# Error distribution
ax = axes[1]; ax.set_facecolor(BG2)
errors = y_test - y_pred
ax.hist(errors, bins=20, color=SEC, edgecolor=BG, alpha=0.85)
ax.axvline(0,            color=WRN, lw=1.5, ls='--', label='Zero error')
ax.axvline(errors.mean(), color=DNG, lw=1.5, ls=':',  label=f'Mean error {errors.mean():.1f}')
ax.set_xlabel('True RUL − Predicted RUL (cycles)')
ax.set_ylabel('Frequency')
ax.set_title('LSTM Prediction Error Distribution', color=TXT)
ax.legend(fontsize=9); ax.grid(True)

plt.tight_layout()
plt.savefig(OUT / 'fig4b_lstm_actual_vs_predicted.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 5 updated — All models comparison ─────────────────────────────────────
t3_final = json.loads(t3_path.read_text())

model_names = [k for k in t3_final if t3_final[k].get('MAE') not in ('N/A', None)]
mae_v  = [t3_final[k]['MAE']  for k in model_names]
rmse_v = [t3_final[k]['RMSE'] for k in model_names]
r2_v   = [t3_final[k]['R2']   for k in model_names]

MCOLORS = [PRI, ACC, SEC, WRN, DNG][:len(model_names)]
short_names = [n.replace(' ', '\n') for n in model_names]

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle('Fig 5 — Model Performance Comparison (Updated with LSTM)\nFD001 Test Set · 100 Engines',
             fontsize=12, color=TXT)

for ax, (title, vals, best_fn) in zip(axes, [
    ('MAE (lower is better)',  mae_v,  min),
    ('RMSE (lower is better)', rmse_v, min),
    ('R² (higher is better)',  r2_v,   max),
]):
    ax.set_facecolor(BG2)
    best = best_fn(vals)
    cols = [WRN if v == best else MCOLORS[i] for i, v in enumerate(vals)]
    bars = ax.bar(short_names, vals, color=cols, edgecolor=BG, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8.5, color=TXT)
    ax.set_title(title, color=TXT)
    ax.set_ylabel(title.split(' ')[0], color=TXT2)
    ax.grid(True, axis='y')

axes[0].text(0.5, -0.22, '★ Gold = best performer', transform=axes[0].transAxes,
             ha='center', fontsize=8, color=WRN)
plt.tight_layout()
plt.savefig(OUT / 'fig5_model_comparison_updated.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 12. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '═'*60)
print('COMPLETE TABLE 3 — UPDATED WITH LSTM')
print('─'*60)
print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print('─'*60)
for k, v in t3_final.items():
    mae_s  = f"{v['MAE']:.4f}"  if v['MAE']  != 'N/A' else 'N/A'
    rmse_s = f"{v['RMSE']:.4f}" if v['RMSE'] != 'N/A' else 'N/A'
    r2_s   = f"{v['R2']:.4f}"   if v['R2']   != 'N/A' else 'N/A'
    print(f"{k:<22} {mae_s:>10} {rmse_s:>10} {r2_s:>10}")
print('═'*60)
print('\nFigures saved:')
for f in sorted(OUT.glob('*.png')): print(f'  {f.name}')
