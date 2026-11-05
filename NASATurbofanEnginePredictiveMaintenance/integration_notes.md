# AeroSentinel — Integration Notes
**Generated:** 2026-11-03  
**Source repository:** NASATurbofanEnginePredictiveMaintenance/

---

## 1. Existing Training Pipeline

### Script: `v1.py`
Single-file pipeline. Does NOT save model artifacts. Now supplemented by `save_models.py`.

### Pipeline Stages
1. **Data Loading** — `load_cmapss_data(dataset_number)` reads whitespace-separated txt files with no header row.
2. **EDA** — `explore_data(train_df)` produces `eda_visualization.png`.
3. **RUL Target** — `add_rul_target(df)` computes `RUL = max_cycle_per_engine − current_cycle` (linear, uncapped).
4. **Rolling Features** — `add_rolling_features(df, window_sizes=[5,10])` adds rolling mean and std per sensor per engine.
5. **Degradation Features** — `add_degradation_features(df)` adds `sensor_diff_from_start` and `sensor_normalized`.
6. **Binary Target** — `failure_within_30 = (RUL <= 30).astype(int)`.
7. **Test Preparation** — takes the **last cycle** of each test engine; merges ground-truth RUL from `RUL_FD00x.txt`.
8. **Scaling** — `StandardScaler` fitted on training features; applied to test features.
9. **RUL Models** — `RandomForestRegressor(n_estimators=100, max_depth=15)` and `GradientBoostingRegressor(n_estimators=100, max_depth=5)`.
10. **Classification Models** — `RandomForestClassifier` and `GradientBoostingClassifier` with identical hyperparameters.
11. **Evaluation** — MAE, RMSE, R² for regression; Precision, Recall, F1, ROC-AUC for classification.
12. **Visualisations** — 6 PNGs saved to working directory.

---

## 2. Dataset Locations

All files are in the **same directory** as `v1.py`:

```
NASATurbofanEnginePredictiveMaintenance/
├── train_FD001.txt   (20,631 rows × 26 cols)
├── train_FD002.txt   (53,759 rows × 26 cols)
├── train_FD003.txt   (24,720 rows × 26 cols)
├── train_FD004.txt   (61,249 rows × 26 cols)
├── test_FD001.txt    (13,096 rows × 26 cols)  → 100 engines
├── test_FD002.txt    (33,991 rows × 26 cols)  → 259 engines
├── test_FD003.txt    (16,596 rows × 26 cols)  → 100 engines
├── test_FD004.txt    (41,214 rows × 26 cols)  → 248 engines
├── RUL_FD001.txt     (100 entries)
├── RUL_FD002.txt     (259 entries)
├── RUL_FD003.txt     (100 entries)
└── RUL_FD004.txt     (248 entries)
```

**Column schema (26 columns, no header):**
```
unit | cycle | setting1 | setting2 | setting3 | sensor1..sensor21
```

---

## 3. Feature Columns (after engineering)

150 features fed to models (excludes `unit`, `cycle`, `RUL`, `failure_within_30`):

- 3 operational settings: `setting1`, `setting2`, `setting3`
- 21 raw sensors: `sensor1` … `sensor21`
- 42 rolling mean features: `sensorN_rolling_mean_5`, `sensorN_rolling_mean_10`
- 42 rolling std features: `sensorN_rolling_std_5`, `sensorN_rolling_std_10`
- 21 diff-from-start features: `sensorN_diff_from_start`
- 21 normalised features: `sensorN_normalized`

Exact ordered list saved in: `models/feature_cols_fd001.json`

---

## 4. Model Names and Artifacts

Saved by `save_models.py` to `models/`:

| File | Description |
|------|-------------|
| `rul_rf_fd001.joblib` | Random Forest — RUL regressor |
| `rul_gb_fd001.joblib` | Gradient Boosting — RUL regressor |
| `failure_rf_fd001.joblib` | Random Forest — failure classifier |
| `failure_gb_fd001.joblib` | Gradient Boosting — failure classifier |
| `scaler_fd001.joblib` | StandardScaler fitted on FD001 training features |
| `feature_cols_fd001.json` | Ordered feature column list (150 items) |
| `model_metrics_fd001.json` | All evaluation metrics |

**FD002–FD004 artifacts: NOT yet saved.** Backend must handle graceful fallback.

---

## 5. Training and Testing Methodology

- **Train split:** entire train_FD00x.txt (run-to-failure trajectories).
- **Test split:** last cycle of each engine in test_FD00x.txt; true RUL from RUL_FD00x.txt.
- **No cross-validation** in original code.
- **No time-series leakage guard** — rolling features use `min_periods=1` so early cycles may have limited smoothing.
- **RUL cap:** NONE — the pipeline uses linear uncapped RUL (max_cycle − current_cycle). This means high early-life RUL values may skew regression metrics.

---

## 6. Verified FD001 Metrics

### RUL Regression
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 13.15 | 18.40 | 0.804 |
| Gradient Boosting | 14.15 | 19.38 | 0.782 |

**Preferred RUL model:** Random Forest

### Failure Classification (within 30 cycles)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.923 | 0.960 | 0.941 | 0.993 |
| Gradient Boosting | 0.893 | 1.000 | 0.943 | 0.998 |

**Preferred failure classifier:** Gradient Boosting

### Production Monitoring (FD001 test set, 100 engines)
| Metric | Count |
|---|---:|
| Engines with predicted RUL < 30 | 24 |
| Engines with predicted RUL < 15 | 10 |
| Engines with failure probability > 0.7 | 26 |

---

## 7. Existing PNG Visualisations

| File | Contents | Backend route |
|------|----------|---------------|
| `eda_visualization.png` | 4-panel: lifecycle hist, sensor2 degradation, correlation heatmap, operating conditions scatter | `/static/eda_visualization.png` |
| `rul_model_results.png` | 4-panel: pred vs actual, error hist, model comparison bar, residuals | `/static/rul_model_results.png` |
| `failure_model_results.png` | 4-panel: confusion matrix, ROC curve, model comparison, probability dist | `/static/failure_model_results.png` |
| `feature_importance_rul.png` | Top-20 features for RUL model (horizontal bar) | `/static/feature_importance_rul.png` |
| `feature_importance_failure.png` | Top-20 features for failure classifier | `/static/feature_importance_failure.png` |
| `production_monitoring_dashboard.png` | 4-panel: RUL pred sample, failure risk alerts, error tracking, alert summary | `/static/production_monitoring_dashboard.png` |

---

## 8. Missing Components (to be implemented in AeroSentinel)

1. **Model persistence** — now addressed by `save_models.py`.
2. **Anomaly detection** — no Isolation Forest in v1.py; implement separately.
3. **Explainability** — no SHAP in v1.py; use `model.feature_importances_` + permutation importance.
4. **API layer** — none; implement with FastAPI.
5. **Database** — none; implement SQLite via SQLAlchemy.
6. **Report generation** — none; implement PDF/CSV in backend.
7. **FD002–FD004 model artifacts** — not saved; backend must note unavailability for those subsets.

---

## 9. Backend Integration Requirements

- Backend must read dataset files from the existing directory (symlink or configured absolute path).
- Backend preprocessing module must replicate `add_rul_target` + `add_rolling_features` + `add_degradation_features` exactly.
- Scaler must be loaded from `scaler_fd001.joblib` and applied before inference.
- Feature column order must match `feature_cols_fd001.json`.
- For single-engine inference: engineer the full historical trajectory, extract the last-cycle row, scale, predict.
- For dataset-level predictions: process all engines in the selected test file.
