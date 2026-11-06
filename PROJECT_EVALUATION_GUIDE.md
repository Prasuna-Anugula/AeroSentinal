# AeroSentinel — Project Evaluation Guide
### Intelligent Engine Health Monitoring & Predictive Maintenance Platform
**Prepared for:** Project Evaluation  
**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Dataset  
**Tagline:** Predict Before Failure.

---

## 1. What This Project Does — One Paragraph

AeroSentinel is a full-stack AI-powered predictive maintenance platform that takes raw turbofan engine sensor data and transforms it into actionable maintenance intelligence. It ingests multivariate time-series sensor readings from the NASA C-MAPSS dataset, engineers 150 features per engine, runs them through trained machine learning models (Random Forest, Gradient Boosting, XGBoost, LSTM), and delivers five core outputs through a web dashboard: how many cycles an engine has left before failure (RUL), whether it is at risk of failing within 30 cycles, whether its sensor patterns are anomalous, which sensors are driving predictions, and what maintenance action to take. Everything is connected — from raw `.txt` files to a live React dashboard backed by a FastAPI server and a SQLite database. 

---

## 2. The Problem Being Solved

Traditional maintenance is either:
- **Reactive** — fix it after it breaks (expensive, unsafe)
- **Scheduled** — replace parts on a fixed calendar (wasteful)

**Predictive maintenance** uses sensor data to predict *when* a specific engine will fail, so you maintain it at exactly the right time. This project implements that for aircraft turbofan engines using publicly available NASA sensor data.

The key question is: **given an engine's sensor readings up to this moment, how many operating cycles does it have left?**

---

## 3. The Dataset — NASA C-MAPSS

**Full name:** Commercial Modular Aero-Propulsion System Simulation  
**Source:** NASA Prognostics Center of Excellence  
**Important:** This is a *simulation* dataset, not data from real flying aircraft.

### Four subsets

| Subset | Engines (Train) | Engines (Test) | Operating Conditions | Fault Modes | Complexity |
|--------|----------------|----------------|---------------------|-------------|------------|
| FD001 | 100 | 100 | 1 | 1 | Simple |
| FD002 | 260 | 259 | 6 | 1 | Medium |
| FD003 | 100 | 100 | 1 | 2 | Medium |
| FD004 | 249 | 248 | 6 | 2 | Complex |

**This project trains models on FD001.** FD001 is used because it is the simplest and most studied subset — one operating condition, one fault mode, making it the standard benchmark. The application can browse all four subsets, but only FD001 has trained models with verified metrics.

### File structure

Each file has **26 columns, no header row:**

```
unit | cycle | setting1 | setting2 | setting3 | sensor1 | sensor2 | ... | sensor21
```

- **unit** = engine ID (1, 2, 3 … 100)
- **cycle** = operating cycle number (like flight hours)
- **setting1–3** = flight condition parameters
- **sensor1–21** = physical measurements (temperature, pressure, speed, etc.)

**Three file types per subset:**
- `train_FD001.txt` — full run-to-failure trajectories (you know when they failed)
- `test_FD001.txt` — trajectories cut off at some unknown point (you don't know when they'll fail)
- `RUL_FD001.txt` — the answer key: how many cycles were left at the test cutoff point

The model learns from training data and is evaluated by predicting on test data, then comparing to the RUL answer key.

---

## 4. Complete Technical Workflow

### Step 1 — Raw Data → Feature Engineering

The raw 26 columns are not enough for good prediction. We engineer **150 features**:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Original settings + sensors | 24 | setting1–3, sensor1–21 (sensor dropped if constant) |
| Rolling mean (window 5) | 21 | Smoothed sensor trend over last 5 cycles |
| Rolling mean (window 10) | 21 | Smoothed sensor trend over last 10 cycles |
| Rolling std (window 5) | 21 | Sensor volatility over last 5 cycles |
| Rolling std (window 10) | 21 | Sensor volatility over last 10 cycles |
| Diff from start | 21 | How much sensor drifted from engine's baseline |
| Normalized | 21 | Sensor value ÷ max lifecycle (accounts for different engine ages) |

**RUL target (regression):** `RUL = max_cycle_for_this_engine − current_cycle`  
Example: if engine ran 200 cycles total and we're at cycle 150, RUL = 50.

**Failure label (classification):** `failure_within_30 = 1 if RUL ≤ 30 else 0`  
This is what the binary classifier predicts.

**Critical rule:** Neither `RUL` nor `failure_within_30` is ever included in the 150 input features. They are targets only. Verified by `verify_pipeline.py`.

### Step 2 — Model Training

All models trained on `train_FD001.txt` (20,631 rows across 100 engines).

**StandardScaler** is fitted on training features only, then applied to test features. This prevents data leakage.

#### Sklearn models (save_models.py)
- Feature matrix: 20,631 rows × 150 features
- Scaler fitted → 4 models trained → evaluated on test set (100 engines, last cycle each)

#### LSTM model (train_lstm.py)
Different approach — uses **sequences** not individual snapshots:
- Input: window of **last 30 cycles × 14 sensors** per engine
- RUL target is **capped at 125** during training (standard C-MAPSS practice — prevents the model from overlearning very large early-life RUL values)
- Produces 17,731 training sequences from 100 engines
- Architecture: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)

### Step 3 — Evaluation (test set only)

For each test engine, take the **last available cycle** of its trajectory, build its 150 features, scale, predict. Compare to `RUL_FD001.txt` which has never been seen during training.

### Step 4 — The Application Serves Predictions

User selects an engine in the web UI → browser sends request to FastAPI → FastAPI loads test data, engineers features, scales, runs model, returns prediction → browser displays result.

---

## 5. Verified Results — Table 3

All values measured from the FD001 test set (100 engines). None are invented.

| Model | MAE (cycles) | RMSE (cycles) | R² |
|-------|-------------|--------------|-----|
| **Random Forest** | **13.1528** | **18.4011** | **0.8039** |
| Gradient Boosting | 14.1461 | 19.3849 | 0.7824 |
| XGBoost | 14.9077 | 20.5159 | 0.7563 |
| **LSTM** | **12.1042** | **15.6203** | **0.8587** |
| GRU | N/A | N/A | N/A |

**Best regression model: LSTM** (MAE 12.10, R² 0.8587)  
**Best sklearn regression model: Random Forest** (MAE 13.15, R² 0.8039) — used in the application because it is fast to serve.

**What these numbers mean:**
- **MAE 13.15** = on average, the Random Forest prediction is off by 13.15 cycles
- **RMSE 18.40** = penalises large errors more; some engines have bigger mistakes
- **R² 0.804** = the model explains 80.4% of the variance in remaining useful life

### Failure Classification (FD001, Gradient Boosting — preferred)

| Metric | Value | Meaning |
|--------|-------|---------|
| Precision | 0.893 | 89.3% of engines flagged as failing actually are |
| Recall | 1.000 | Catches 100% of engines that truly will fail within 30 cycles |
| F1 | 0.943 | Harmonic mean — excellent balance |
| ROC-AUC | 0.998 | Near-perfect discrimination between failing and healthy |

Recall = 1.000 is the most important metric in maintenance: **you never miss a failing engine.**

### Production Monitoring Summary (FD001, 100 test engines)

| Metric | Count |
|--------|-------|
| Engines with predicted RUL < 30 cycles | 24 / 100 |
| Engines with predicted RUL < 15 cycles | 10 / 100 |
| Engines with failure probability > 0.7 | 26 / 100 |

---

## 6. The Application — 10 Pages Explained

### How to start

```
Terminal 1:  cd AeroSentinel/backend   →  python run.py
Terminal 2:  cd AeroSentinel/frontend  →  npm run dev
Browser:     http://localhost:5173
```

---

### Page 1 — Landing Page

**What it is:** The entry point. A cinematic aerospace-style page with animated orbital rings and a turbine aesthetic.

**What it shows:**
- Product headline: "PREDICT BEFORE FAILURE"
- Five capabilities of the system
- FD001 benchmark metrics (MAE, R², F1, AUC)
- Dataset overview table (FD001–FD004)
- Bottom ticker: SYSTEM: LOCAL | DATASET: NASA C-MAPSS | SENSOR CHANNELS: 21

**Why it matters for evaluation:** Shows project branding and that you understand the problem domain. The metrics shown here are real — verifiable against `model_metrics_fd001.json`.

**Button to click:** "INITIALIZE MISSION CONTROL" → takes you to the dashboard.

---

### Page 2 — Mission Control Overview

**What it is:** The command-center home screen. Gives you the full picture of the entire fleet at a glance.

**KPI cards (top row):**
| Card | What it means |
|------|--------------|
| ENGINES | Total engines in selected dataset's test set |
| AVG RUL | Average predicted remaining useful life across all engines |
| RUL < 30 | Count of engines the model says will fail within 30 cycles |
| CRITICAL | Engines with RUL < 20 cycles (AeroSentinel threshold) |
| HIGH RISK | Engines where failure probability > 70% |
| RUL R² | Model accuracy on the test set |

**Metrics banner:** Shows MAE, RMSE, F1, ROC-AUC from the actual FD001 training run. Labelled "FD001 EXISTING TRAINING RUN" — not real-time.

**Charts:**
- **RUL Distribution bar chart** — how many engines fall in each RUL bucket (0–20, 21–50, 51–100, 100+)
- **Fleet Health donut** — breakdown of Healthy / Warning / High Risk / Critical engines

**Production monitoring panel:** Shows the three key numbers (24, 10, 26) from the verified FD001 run.

**Dataset selector (top bar):** Switching to FD002/3/4 shows a warning banner: "Models were trained on FD001 — these predictions use FD001-trained weights on this dataset."

---

### Page 3 — Engine Monitor

**What it is:** Browse individual engines. Select any engine and inspect its sensor behaviour.

**Left panel:** Scrollable list of all engines with predicted RUL and health badge. Searchable.

**Right panel (after selecting an engine):**
- Engine ID, dataset, last cycle
- Predicted RUL, failure probability, anomaly status
- **Sensor degradation chart** — choose which of the 21 sensors to plot over cycles. This shows the degradation pattern visually.
- **Sensor snapshot table** — latest reading, mean, min, max for all 21 sensors

**Why it matters:** This is where you can *see* the data. For evaluation, select a critical engine (e.g., Engine 42 — predicted RUL ~13, true RUL 10) and show how its sensors drift over cycles.

---

### Page 4 — RUL Prediction

**What it is:** The primary ML feature. Predict remaining useful life for any engine on demand.

**Workflow on this page:**
1. Select dataset (FD001 recommended)
2. Select engine unit (e.g., Engine 42)
3. Select model (Random Forest = preferred for FD001)
4. Click PREDICT RUL
5. Backend engineers features from test_FD001.txt, scales, predicts
6. Result displayed with animated gauge

**What the gauge shows:**
- A circular arc fills proportionally — more fill = more life remaining
- Colour changes: Green (Healthy >100) → Yellow (Warning 50–100) → Orange (High Risk 20–49) → Red (Critical <20)
- The number in the centre is predicted cycles remaining

**Health thresholds (AeroSentinel-defined, not aviation standards):**
| Status | RUL Range |
|--------|-----------|
| Healthy | > 100 cycles |
| Warning | 50–100 cycles |
| High Risk | 20–49 cycles |
| Critical | < 20 cycles |

**What else shows:** Last cycle, model used, health status badge.

**For evaluation:** Demonstrate that different engines produce different results (Engine 1 = Healthy 110 cycles, Engine 42 = Critical 12 cycles). This proves inference is real, not hardcoded.

---

### Page 5 — Failure Risk

**What it is:** The binary classifier. Answers one question: will this engine fail within 30 cycles?

**Difference from RUL:** RUL prediction gives a number. Failure risk gives a probability. These use different trained models.

**Workflow:** Select engine + classifier → ASSESS FAILURE RISK

**What the circular gauge shows:** The failure probability as a percentage (0–100%). The colour tells you the risk tier.

| Risk Level | Probability | Meaning |
|------------|-------------|---------|
| Low | < 40% | Engine is likely safe |
| Medium | 40–70% | Monitor closely |
| High | ≥ 70% | Treat as at risk |

**Alert box:** If probability ≥ 0.7, a red warning box appears: "High failure risk detected. Schedule precautionary inspection."

**Fleet panel (bottom):** Shows all engines in the dataset with probability > 70% — the at-risk fleet.

**For evaluation:** Show that Gradient Boosting (preferred classifier) with Recall = 1.000 means it catches every engine that will actually fail. Precision = 0.893 means ~11% of flagged engines are false alarms — acceptable trade-off in safety-critical systems.

---

### Page 6 — Anomaly Detection

**What it is:** Isolation Forest detects *abnormal sensor patterns* independently of the RUL prediction.

**Why it exists separately from RUL:** An engine can have unusual sensor behaviour before it shows up as a declining RUL. Anomaly detection acts as an early warning system.

**Model used:** Isolation Forest (sklearn), trained on FD001 training sensor readings (all 21 sensors, 20,631 rows). Contamination parameter = 5% (assumes ~5% of readings are anomalous).

**How Isolation Forest works (to explain in evaluation):** It randomly partitions data into trees. Anomalous points require fewer splits to isolate because they are far from normal clusters. The `decision_function` returns a score — more negative = more anomalous.

**Score interpretation:**
| Score | Status |
|-------|--------|
| > −0.10 | Normal |
| −0.10 to −0.20 | Unusual |
| < −0.20 | High Anomaly |

**What the page shows:**
- Overall anomaly score for the selected engine
- Timeline chart — score plotted over every cycle (anomalous cycles highlighted in red)
- Table of most anomalous cycles with which sensors triggered them

**Important distinction:** Anomaly score ≠ failure probability. A high anomaly score means unusual sensor patterns, not necessarily imminent failure.

---

### Page 7 — Sensor Intelligence

**What it is:** Feature importance and per-engine explanations. Answers: *why* is the model making this prediction?

**Two types of explanation:**

**Global feature importance:**
- Uses `model.feature_importances_` from the trained Random Forest
- Shows which of the 150 features matter most across all predictions
- Aggregated by base sensor name (e.g., all features derived from sensor2 are summed)
- Top sensors by importance: sensor1, sensor4, sensor9, sensor2, sensor7

**Local feature contribution (per engine):**
- For a specific engine at its last cycle, computes: `contribution = global_importance × |scaled_feature_value|`
- Shows which features pushed the prediction for *this specific engine*
- **Important note:** This is a contribution approximation, NOT TreeSHAP. The research figures (Fig 7) use actual `shap.TreeExplainer`. The in-app explanation is labelled "Local Feature Contribution" to be accurate.

**Reference images:** Shows the original `feature_importance_rul.png` and `feature_importance_failure.png` generated during training.

**For evaluation:** Explain that sensor1 having the highest SHAP value means it is the most influential measurement. In a real turbofan, sensor1 measures Total temperature at fan inlet — a key indicator of engine health.

---

### Page 8 — Maintenance Planner

**What it is:** Translates model outputs into maintenance recommendations. No LLM — pure deterministic rules.

**Rule engine:**
```
IF predicted RUL < 15:
    Priority = CRITICAL
    Action   = Immediate engineering review required

IF predicted RUL < 30:
    Priority = HIGH
    Action   = Schedule priority inspection

IF failure probability > 0.7:
    Priority = HIGH
    Action   = Review failure-risk indicators

IF anomaly severity = high_anomaly:
    Priority = HIGH
    Action   = Inspect abnormal sensor patterns

IF anomaly severity = unusual:
    Priority = MEDIUM
    Action   = Increase monitoring frequency

ELSE:
    Priority = NORMAL
    Action   = Continue routine monitoring
```

**What the page shows:**
- Generate a recommendation for any engine (calls RUL + failure + anomaly APIs simultaneously)
- Recommendation card with priority badge, recommendation text, evidence (RUL, probability, anomaly score)
- History of all past recommendations
- Status workflow: Pending → Reviewed → Resolved
- SQLite audit trail — every recommendation is saved

**For evaluation:** Generate a recommendation for Engine 42 (critical — RUL ~13). It should produce: CRITICAL / Immediate engineering review. Then show Engine 1 (healthy — RUL ~111) produces: NORMAL / Routine monitoring.

---

### Page 9 — Dataset Explorer

**What it is:** Interactive exploration of all four C-MAPSS subsets.

**Shows:**
- Side-by-side summary cards for FD001–FD004 (engines, observations, conditions, fault modes, complexity)
- Which datasets have trained models available (only FD001 = green badge)
- Engine lifecycle distribution histogram for selected dataset
- True RUL distribution histogram for selected dataset
- Reference image: `eda_visualization.png` from the original training run

**For evaluation:** Use this to explain the dataset structure and show why FD001 is simpler than FD004 (1 condition vs 6 conditions, 1 fault mode vs 2).

---

### Page 10 — Model Insights

**What it is:** The research results page. Shows all measured metrics and reference visualisations.

**Table 3 data** — all four measured models (RF, GB, XGBoost, LSTM).

**Reference visualisations served directly:**
- `rul_model_results.png` — actual vs predicted scatter, error distribution, model comparison
- `failure_model_results.png` — confusion matrix, ROC curve, probability distribution
- `production_monitoring_dashboard.png` — fleet-level RUL predictions, alert summary

**Production monitoring panel:** The three key numbers (24, 10, 26) with context.

**For evaluation:** This page is your evidence. Every number came from `generate_results.py` running on real data with real models.

---

## 7. The Eight Research Figures Explained

### Fig 1 — System Architecture Diagram
**File:** `results_figures/fig1_system_architecture.png`

Shows the end-to-end pipeline in 5 layers:
1. Data Collection (NASA C-MAPSS)
2. Preprocessing + Feature Engineering (150 features)
3. ML Models (RUL + Anomaly + SHAP)
4. Health Assessment + Maintenance Recommendation
5. Dashboard outputs (Predicted RUL, Failure Risk, Anomaly Status, Sensor Explanation, Maintenance Action)

Bottom strip shows the 8-step flow.

---

### Fig 2 — Dataset Exploration & Sensor Measurements
**File:** `results_figures/fig2_dataset_exploration.png`

10 sub-plots:
- **(a)** Engine lifecycle distribution — shows how long engines ran before failure (mean ~206 cycles)
- **(b)** True RUL distribution at test cutoff — shows how much life remained when test data ends
- **(c–j)** Histograms of 8 key sensor distributions across all training cycles

**What to say:** "This shows the dataset structure. FD001 has 100 engines with lifecycles ranging from 128 to 362 cycles. The sensor distributions show the operating range of each measurement."

---

### Fig 3 — Sensor Degradation Trends
**File:** `results_figures/fig3_sensor_degradation.png`

6 sub-plots, one per key sensor, each showing 5 sample engines over their full lifecycle.

**Sensors shown:** sensor2, sensor3, sensor4, sensor11, sensor12, sensor15

**What to say:** "These plots show that sensor readings change over time as engines degrade. This is the signal the models learn from. Sensors that show consistent monotonic trends (like sensor2 and sensor7) are the most informative."

---

### Fig 4 — Actual versus Predicted RUL
**File:** `results_figures/fig4_actual_vs_predicted.png`  
**LSTM version:** `results_figures/fig4b_lstm_actual_vs_predicted.png`

3 scatter plots — one per model (RF, GB, XGBoost).

Each plot: X axis = true RUL (from RUL_FD001.txt), Y axis = predicted RUL.

- The **yellow dashed line** is perfect prediction (predicted = actual)
- Points close to the line = accurate predictions
- Spread away from the line = errors
- The title shows MAE, RMSE, R² for each model

**What to say:** "If the model were perfect, all points would lie on the dashed line. Random Forest has the tightest clustering, confirmed by its lowest MAE of 13.15 cycles."

---

### Fig 5 — Model Performance Comparison
**File:** `results_figures/fig5_model_comparison_updated.png`

3 bar charts comparing all measured models (RF, GB, XGBoost, LSTM) on MAE, RMSE, and R².

- Gold bar = best performer for that metric
- For MAE and RMSE: lower = better
- For R²: higher = better

**What to say:** "LSTM achieves the best performance overall — MAE 12.10, R² 0.859 — because it captures temporal dependencies across 30 cycles. Random Forest is the best sklearn model and is used in the application because it's faster to serve."

---

### Fig 6 — Anomaly Detection Results
**File:** `results_figures/fig6_anomaly_detection.png`

3 sub-plots:
- **(a) Fleet anomaly score distribution** — histogram of scores for normal vs anomalous engines. Two dashed lines show the "Unusual" (−0.10) and "High Anomaly" (−0.20) thresholds.
- **(b) Engine 42 timeline** — anomaly score plotted over every cycle. Red dots = anomalous cycles. Shows *when* abnormal behaviour started.
- **(c) Score vs True RUL scatter** — coloured by score. Shows that engines with lower true RUL tend to have lower (more anomalous) scores.

**What to say:** "Isolation Forest with 5% contamination flagged 2 out of 100 engines at their test-set cutoff cycle. The timeline shows that Engine 42 (a critical engine) had anomalous patterns appearing in its later cycles, consistent with its low true RUL of 10."

---

### Fig 7 — SHAP Sensor Contribution Analysis
**File:** `results_figures/fig7_shap_analysis.png`

2 sub-plots using actual `shap.TreeExplainer` on Random Forest:
- **(a) Sensor-level ranking** — importance aggregated across all engineered features per sensor. **sensor1 has the highest SHAP value (40.34)**, followed by sensor4 (30.04).
- **(b) Top 20 individual features** — the specific engineered features (rolling means, diffs, etc.) that matter most.

**SHAP top 5 sensors:**
| Rank | Sensor | Mean |SHAP| |
|------|--------|---------|
| 1 | sensor1 | 40.339 |
| 2 | sensor4 | 30.038 |
| 3 | sensor9 | 11.181 |
| 4 | sensor2 | 7.291 |
| 5 | sensor7 | 3.710 |

**What to say:** "SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to each prediction. Sensor1 (total temperature at fan inlet) is the most influential — it has the highest average absolute SHAP value across the 100 test engines, meaning changes in this sensor's reading most strongly affect the model's RUL prediction."

**Note:** The in-app per-engine explanation uses a contribution approximation (importance × |scaled value|), not TreeSHAP. The Fig 7 results use actual SHAP.

---

### Fig 8 — Machine Health Monitoring Dashboard
**File:** `results_figures/fig8_health_dashboard.png`

6 engine panels showing the complete health snapshot for engines 1, 10, 23, 42, 67, 88:

Each panel contains:
- Engine ID
- RUL arc gauge (colour-coded by health status)
- Health status badge (Healthy / Warning / High Risk / Critical)
- Anomaly status badge (Normal / Unusual / High Anomaly)
- Last cycle, anomaly score, top contributing sensor, failure classification

**For evaluation:** Point to Engine 42 panel — it shows Critical health, with the gauge almost empty, confirming what the RUL prediction page shows.

---

## 8. Technical Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Dataset | NASA C-MAPSS .txt files | Raw sensor data |
| ML pipeline | Python, scikit-learn, XGBoost, TensorFlow | Model training |
| Feature engineering | Pandas, NumPy | 150 features from 26 raw columns |
| Explainability | SHAP (figures), feature_importances_ (app) | Why predictions are made |
| Backend | FastAPI, Python 3.13 | REST API, ML inference |
| Database | SQLite, SQLAlchemy | Persist predictions and recommendations |
| Frontend | React, TypeScript, Vite | Web interface |
| Styling | Tailwind CSS | Responsive layout |
| Charts | Recharts | Interactive data visualisation |
| Animation | Framer Motion | Smooth UI transitions |

---

## 9. System Architecture — Detailed Explanation

The system is divided into four distinct layers. Each layer has a single responsibility and communicates with the next through a clean interface.

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA LAYER                                           │
│  NASA C-MAPSS .txt files  →  Raw 26-column sensor readings      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — ML PIPELINE (Python, offline)                        │
│  Feature Engineering  →  Model Training  →  Saved Artifacts     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — BACKEND (FastAPI, port 8000)                         │
│  Load artifacts  →  Run inference  →  SQLite  →  REST API       │
└──────────────────────────────┬──────────────────────────────────┘
                               │  JSON over HTTP
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — FRONTEND (React + Vite, port 5173)                   │
│  10 pages  →  Charts  →  Forms  →  Reports                      │
└─────────────────────────────────────────────────────────────────┘
```

---

### Layer 1 — Data Layer

**What it is:** Seven plain-text files per dataset subset, no header row, whitespace-separated.

**Files:**
```
train_FD001.txt   →  100 engines, each run until failure (20,631 rows total)
test_FD001.txt    →  Same 100 engines, cut off at an unknown point (13,096 rows)
RUL_FD001.txt     →  100 numbers = how many cycles each test engine had left at cutoff
```

**Column structure (26 columns):**
```
[unit] [cycle] [setting1] [setting2] [setting3] [s1] [s2] ... [s21]
  ↑       ↑        ↑           ↑          ↑        ↑    ↑          ↑
Engine  Flight  Altitude   Speed      Throttle  Temp Press     Vibration
ID      number  setting    setting    setting   readings (21 total)
```

**Key constraint:** The test file deliberately withholds the failure point. You cannot look at `RUL_FD001.txt` during inference — that is the answer key used only for evaluation after predictions are made.

---

### Layer 2 — ML Pipeline

This layer runs **once offline** to produce trained model files. It is split into three scripts:

#### A. save_models.py — Sklearn models

```
train_FD001.txt
      │
      ▼
add_rul_target()        RUL = max_cycle_per_engine − current_cycle
      │
      ▼
add_rolling_features()  For each of 21 sensors:
                          rolling_mean_5, rolling_mean_10   (smoothed trend)
                          rolling_std_5,  rolling_std_10    (volatility)
      │
      ▼
add_degradation_features() For each sensor:
                          diff_from_start   (drift since engine was new)
                          normalized        (value ÷ max_lifecycle)
      │
      ▼
failure_within_30 = (RUL ≤ 30).astype(int)   ← classification target
      │
      ▼
150 features total (26 raw → 150 engineered)
      │
      ▼
StandardScaler.fit(X_train) → saved as scaler_fd001.joblib
      │
      ▼
┌─────────────────────────────────────────────────┐
│  RandomForestRegressor(100 trees, depth=15)     │ → rul_rf_fd001.joblib
│  GradientBoostingRegressor(100 trees, depth=5)  │ → rul_gb_fd001.joblib
│  XGBRegressor(200 trees, lr=0.05, depth=6)      │ → rul_xgb_fd001.joblib
│  RandomForestClassifier(100 trees, depth=15)    │ → failure_rf_fd001.joblib
│  GradientBoostingClassifier(100 trees, depth=5) │ → failure_gb_fd001.joblib
└─────────────────────────────────────────────────┘
      │
      ▼
Evaluate on test set (last cycle per engine, compare to RUL_FD001.txt)
      │
      ▼
feature_cols_fd001.json      (ordered list of 150 feature names)
model_metrics_fd001.json     (MAE, RMSE, R², F1, AUC for all models)
```

#### B. train_lstm.py — Deep learning models

Different preprocessing — uses **sequences** not snapshots:

```
train_FD001.txt
      │
      ▼
Select 14 informative sensors (drop 7 that are constant or near-zero variance)
      │
      ▼
MinMaxScaler.fit(X_train_sensors) → rescale to 0–1 range
      │
      ▼
Cap RUL at 125 cycles (piecewise linear target)
      │
      Why cap? Without cap, early-life RUL values are 200–300.
      Model wastes capacity fitting those extremes.
      After cap, every engine looks "at most 125 cycles from failure"
      which makes the learning task easier and more uniform.
      │
      ▼
Sliding window: for every position in every engine's trajectory,
  take the LAST 30 cycles × 14 sensors → shape (30, 14)
  label = RUL at the last cycle of the window
      │
      17,731 training sequences from 100 engines
      │
      ▼
┌──────────────────────────────────────────────────┐
│  LSTM architecture                               │
│  Input:     (30 cycles × 14 sensors)             │
│  LSTM(128)  return_sequences=True  → (30, 128)   │
│  Dropout(0.2)                                    │
│  LSTM(64)   return_sequences=False → (64,)       │
│  Dropout(0.2)                                    │
│  Dense(32)  ReLU                  → (32,)        │
│  Dense(1)                         → scalar RUL   │
│                                                  │
│  Total params: 124,737                           │
│  Loss: MSE  |  Optimizer: Adam(lr=0.001)         │
│  EarlyStopping(patience=10)                      │
│  ReduceLROnPlateau(patience=5, factor=0.5)       │
└──────────────────────────────────────────────────┘
      │
      ▼
GRU uses identical architecture with GRU layers instead of LSTM
      │
      ▼
lstm_fd001.keras
gru_fd001.keras
```

**Why LSTM beats Random Forest:**
Random Forest takes one snapshot — the last cycle's 150 features. It cannot see the *rate of change* across cycles.
LSTM receives 30 consecutive cycles and learns temporal patterns — a sensor that has been declining steadily over 30 cycles signals more risk than one that just happened to read low once. That temporal context is why LSTM achieves MAE 12.10 vs Random Forest's 13.15.

#### C. Isolation Forest — Anomaly Detection

```
train_FD001.txt  →  21 raw sensor readings  →  StandardScaler
      │
      ▼
IsolationForest(n_estimators=100, contamination=0.05)
      │
      How it works:
      Builds 100 random decision trees. For each point, counts
      how many splits are needed to isolate it. Normal points
      are surrounded by many others — hard to isolate, need many splits.
      Anomalous points are far from the cluster — isolated quickly.
      Short path length = low decision_function score = anomalous.
      │
      ▼
decision_function(X) → score per cycle
  score > −0.10  → Normal
  score < −0.10  → Unusual
  score < −0.20  → High Anomaly
```

---

### Layer 3 — Backend (FastAPI)

**Entry point:** `AeroSentinel/backend/run.py`  
**Starts:** `uvicorn app.main:app --port 8000`

#### Startup sequence

```
python run.py
      │
      ▼
init_db()
  Creates SQLite file at storage/aerosentinel.db
  Creates 5 tables:
    Engine                  ← one row per engine+dataset combination
    Prediction              ← every RUL or failure prediction made
    Anomaly                 ← every anomaly detection run
    MaintenanceRecommendation ← generated actions + status
    ModelMetadata           ← model info
      │
      ▼
ModelRegistry.load()
  Loads from NASATurbofanEnginePredictiveMaintenance/models/:
    rul_rf_fd001.joblib     (85 MB — Random Forest with 100 trees)
    rul_gb_fd001.joblib     (0.5 MB)
    failure_rf_fd001.joblib (4 MB)
    failure_gb_fd001.joblib (0.4 MB)
    scaler_fd001.joblib     (0.004 MB)
    feature_cols_fd001.json (150 feature names in exact order)
      │
      ▼
Isolation Forest trained on FD001 (or loaded from cache)
      │
      ▼
All 9 API routers registered
API ready at http://localhost:8000
Docs at     http://localhost:8000/api/docs
```

#### Request lifecycle — example: predict RUL for Engine 42

```
Browser: POST /api/predict/rul
  body: { "dataset": "FD001", "engine_unit": 42, "model_key": "rul_rf" }
          │
          ▼
predictions.py router receives request
          │
          ▼
prediction_service.run_rul_prediction(db, "FD001", 42, "rul_rf")
          │
          ▼
rul_predictor.predict_single_engine("FD001", 42, "rul_rf")
          │
          ▼
Load test_FD001.txt (cached in memory after first load)
Filter rows where unit == 42  →  156 rows
          │
          ▼
engineer_features(engine_df):
  add_rul_target()         → adds RUL column (used only for failure_within_30)
  add_rolling_features()   → 84 new columns
  add_degradation_features() → 42 new columns
  failure_within_30 = (RUL ≤ 30)
          │
          ▼
Take last row only (cycle 156)  →  shape (1, 154)
Extract only FEAT_COLS columns  →  shape (1, 150)
          │
          ▼
scaler.transform(X)  →  normalised (1, 150)
          │
          ▼
rul_rf.predict(X_scaled)  →  [12.87]
max(12.87, 0) = 12.87 cycles
          │
          ▼
health_status_from_rul(12.87) = "critical"
          │
          ▼
Save to SQLite:
  Engine row: unit=42, dataset=FD001, last_cycle=156
  Prediction row: predicted_rul=12.87, model=rul_rf, health=critical
          │
          ▼
Return JSON:
  { engine_unit: 42, predicted_rul: 12.87, health_status: "critical",
    last_cycle: 156, true_rul: 10.0, model: "rul_rf", ... }
          │
          ▼
Browser renders gauge at 12.87 cycles, Critical badge (red)
```

#### Backend folder structure

```
backend/
├── app/
│   ├── main.py          FastAPI app, CORS, router registration, startup
│   ├── config.py        Paths to dataset dir, models dir, storage dir
│   ├── database.py      SQLAlchemy engine + session + Base + init_db()
│   │
│   ├── api/             One file per concern (thin routers only)
│   │   ├── health.py        GET /api/health, /api/system/status
│   │   ├── dashboard.py     GET /api/dashboard/summary, /fleet
│   │   ├── predictions.py   POST /api/predict/rul, /failure
│   │   ├── anomalies.py     POST /api/anomaly/detect
│   │   ├── explanations.py  GET  /api/explanations/feature-importance
│   │   ├── maintenance.py   POST /api/maintenance/recommendations
│   │   ├── datasets.py      GET  /api/datasets/{subset}/engines
│   │   ├── models.py        GET  /api/models/comparison
│   │   └── reports.py       GET  /api/reports/engine/csv, /pdf
│   │
│   ├── services/        Business logic (called by routers)
│   │   ├── prediction_service.py   orchestrate RUL + failure inference
│   │   ├── anomaly_service.py      run Isolation Forest, save to DB
│   │   ├── explanation_service.py  feature importance + local contribution
│   │   ├── maintenance_service.py  rule engine + DB persistence
│   │   ├── data_service.py         dataset loading + caching
│   │   └── report_service.py       CSV + PDF generation
│   │
│   ├── ml/              Pure ML — no HTTP, no DB
│   │   ├── model_loader.py     ModelRegistry singleton (load once at startup)
│   │   ├── preprocessing.py    Exact replica of v1.py feature engineering
│   │   ├── rul_predictor.py    Single engine + all engines inference
│   │   ├── failure_predictor.py Single engine + all engines classification
│   │   ├── anomaly_detector.py  Isolation Forest train/load/score
│   │   └── evaluation.py       Load metrics JSON, sensor correlations
│   │
│   ├── models/          SQLAlchemy ORM table definitions
│   │   └── orm.py
│   │
│   └── schemas/         Pydantic request/response contracts
│       └── schemas.py
│
└── run.py               uvicorn entry point
```

**Key design principle:** Three-layer separation inside the backend.
- **Routers** only validate requests and call services. No ML code.
- **Services** orchestrate logic, call ML layer, write to DB.
- **ML layer** is pure Python functions — no FastAPI, no SQLAlchemy. Testable in isolation.

---

### Layer 4 — Frontend (React + Vite)

**Entry point:** `frontend/src/main.tsx`  
**Router:** React Router v6 with nested routes under `/app`

#### Component hierarchy

```
main.tsx
  └── App.tsx                    BrowserRouter + ToastProvider
        ├── /                    Landing.tsx  (no layout)
        └── /app                 AppLayout.tsx
              ├── Sidebar.tsx    Collapsible nav (10 links)
              ├── TopBar.tsx     Dataset selector + status bar
              └── <Outlet>       Routed page component
                    ├── Overview.tsx
                    ├── EngineMonitor.tsx
                    ├── RULPrediction.tsx
                    ├── FailureRisk.tsx
                    ├── AnomalyDetection.tsx
                    ├── SensorIntelligence.tsx
                    ├── MaintenancePlanner.tsx
                    ├── DatasetExplorer.tsx
                    ├── ModelInsights.tsx
                    └── Settings.tsx
```

#### How a page talks to the backend

```
User clicks "PREDICT RUL"
        │
        ▼
RULPrediction.tsx calls predictRUL(dataset, engine, modelKey)
        │
        ▼
src/services/api.ts
  axios.post('/api/predict/rul', { dataset, engine_unit, model_key })
        │
        ▼ (Vite dev server proxies /api → localhost:8000)
        │
        ▼
FastAPI backend processes request
        │
        ▼
Returns JSON
        │
        ▼
React state updated: setResult(r.data)
        │
        ▼
RULGauge component re-renders with new RUL value
Framer Motion animates the arc fill and number counter
```

#### Reusable UI components

```
components/ui/
  GlassCard.tsx      Dark glass panel with blur + border + hover lift
  KpiCard.tsx        Metric tile with icon, large number, subtitle
  StatusBadge.tsx    Coloured pill for Healthy/Warning/Critical etc.
  LoadingSpinner.tsx Animated ring, used while API calls are pending
  Toast.tsx          Notification system (success/error/warning/info)

components/layout/
  Sidebar.tsx        Collapsible nav with Lucide icons, active highlight
  TopBar.tsx         Dataset dropdown, model status, backend status, clock
  AppLayout.tsx      Sidebar + TopBar + Outlet wrapper
```

#### Data flow for the fleet overview (most complex call)

```
Overview.tsx mounts
        │
        ▼
useEffect → Promise.all([
  getDashboardSummary("FD001"),   GET /api/dashboard/summary
  getFleet("FD001")               GET /api/dashboard/fleet
])
        │
        ▼
Backend: get_fleet_overview("FD001")
  Checks _fleet_cache["FD001"] — if hit, returns instantly
  If miss: predict_all_engines + predict_failure_all for 100 engines
  Caches result → subsequent calls are instant
        │
        ▼
Returns 100 engine objects with predicted_rul + failure_probability
        │
        ▼
React state: setSummary + setFleet
        │
        ▼
Recharts renders:
  BarChart   (RUL distribution buckets)
  PieChart   (Fleet health donut)
KpiCard renders 6 metrics
GlassCard renders production monitoring numbers
```

---

### How the three main layers communicate

```
FRONTEND (React)           BACKEND (FastAPI)          ML LAYER (Python)
     │                          │                           │
     │  HTTP POST/GET (JSON)     │                           │
     │ ─────────────────────▶   │                           │
     │                          │  Python function call     │
     │                          │ ─────────────────────▶    │
     │                          │                           │  Load .joblib
     │                          │                           │  Engineer features
     │                          │                           │  Scale + predict
     │                          │  Return prediction dict   │
     │                          │ ◀─────────────────────    │
     │                          │                           │
     │                          │  Write to SQLite          │
     │                          │ ─────────────────────▶ DB │
     │                          │                           │
     │  JSON response           │                           │
     │ ◀─────────────────────   │                           │
     │                          │                           │
  Update state                  │                           │
  Re-render gauge               │                           │
```

---

### Database Schema

SQLite file at `AeroSentinel/storage/aerosentinel.db`

```
Engine
  id             INTEGER PK
  engine_unit    INTEGER          ← e.g. 42
  dataset_subset TEXT             ← e.g. "FD001"
  last_cycle     INTEGER          ← cycle number at last observation
  created_at     DATETIME

Prediction
  id                  INTEGER PK
  engine_id           FK → Engine.id
  predicted_rul       REAL         ← output of RUL model
  failure_probability REAL         ← output of failure classifier
  model_name          TEXT         ← "rul_rf", "fail_gb", etc.
  health_status       TEXT         ← "healthy","warning","high_risk","critical"
  created_at          DATETIME

Anomaly
  id                   INTEGER PK
  engine_id            FK → Engine.id
  cycle                INTEGER      ← which cycle was scored
  anomaly_score        REAL         ← Isolation Forest decision_function value
  severity             TEXT         ← "normal","unusual","high_anomaly"
  contributing_sensors TEXT         ← JSON list of top sensors
  created_at           DATETIME

MaintenanceRecommendation
  id             INTEGER PK
  engine_id      FK → Engine.id
  recommendation TEXT         ← full recommendation text
  priority       TEXT         ← "critical","high","medium","normal"
  evidence       TEXT         ← JSON: {predicted_rul, failure_prob, anomaly_score}
  status         TEXT         ← "pending" → "reviewed" → "resolved"
  created_at     DATETIME
  updated_at     DATETIME

ModelMetadata
  id                 INTEGER PK
  model_name         TEXT UNIQUE
  model_type         TEXT         ← "regression","classification","anomaly"
  dataset_subset     TEXT
  evaluation_metrics TEXT         ← JSON
  artifact_path      TEXT
  is_available       INTEGER      ← 0 or 1
  created_at         DATETIME
```

**Why SQLite and not PostgreSQL?**
This is a local research platform. SQLite requires zero infrastructure setup — no server process, no credentials, no configuration. The database is a single file you can open in any SQLite browser. For production deployment at scale, the SQLAlchemy ORM makes switching to PostgreSQL a one-line config change.

---

### Vite Proxy — Why you don't need CORS config during development

In `vite.config.ts`:
```typescript
server: {
  proxy: {
    '/api':    { target: 'http://localhost:8000', changeOrigin: true },
    '/static': { target: 'http://localhost:8000', changeOrigin: true },
  }
}
```
When the React app at port 5173 calls `/api/predict/rul`, Vite intercepts it and forwards it to `localhost:8000/api/predict/rul`. The browser thinks it's talking to the same origin — no CORS issue. The FastAPI backend also has CORS middleware configured for when the frontend is deployed separately.

---

### Feature Engineering — Why 26 → 150 Features

The raw data has 26 columns. The model needs more context to make accurate predictions. Here is why each feature group was added:

| Feature Group | Raw count | Engineered | Why it helps |
|---------------|-----------|------------|--------------|
| Raw settings + sensors | 26 | 24 | Baseline readings |
| Rolling mean (w=5) | — | 21 | Smooths noise, shows short-term trend |
| Rolling mean (w=10) | — | 21 | Shows medium-term trend |
| Rolling std (w=5) | — | 21 | Increasing volatility signals degradation |
| Rolling std (w=10) | — | 21 | Medium-term volatility |
| Diff from start | — | 21 | Absolute drift — how far has sensor moved from baseline |
| Normalized | — | 21 | Relative position in engine's lifecycle |
| **Total** | **26** | **150** | |

`sensor_diff_from_start` is particularly powerful: a sensor reading of 600 means little on its own. But knowing it started at 580 and has drifted +20 while another engine started at 600 and hasn't moved — that tells the model which one is degrading.

---

## 10. Five Contributions of This Project — Regression models predict remaining engine cycles. Best result: LSTM MAE 12.10 cycles on FD001 test set.

2. **Failure Risk Classification** — Binary classifier predicts failure within 30 cycles. Gradient Boosting achieves ROC-AUC 0.998, Recall 1.000 — catches every failing engine.

3. **Anomaly Detection** — Isolation Forest detects abnormal sensor patterns independently of RUL. Provides early warning before RUL starts declining.

4. **Explainability** — SHAP analysis (Fig 7) and local feature contributions (app) identify which sensors drive each prediction. Sensor1 is the most influential.

5. **Maintenance Recommendation** — Rule-based engine converts model outputs into prioritised, actionable maintenance decisions saved to a SQLite audit trail.

---

## 11. What to Demo at Evaluation

**Recommended demo flow (10 minutes):**

1. Open `http://localhost:5173` — show Landing page, explain the project in one sentence.

2. Click "INITIALIZE MISSION CONTROL" — show Overview page with FD001 selected. Point to KPI cards (100 engines, 24 with RUL<30, 10 critical).

3. Go to **Engine Monitor** — select Engine 42. Show sensor degradation chart. Sensors are trending — this engine is degrading.

4. Go to **RUL Prediction** — select Engine 42, click PREDICT RUL. Show Critical badge, ~13 cycles predicted. Then select Engine 1 — show Healthy, ~111 cycles.

5. Go to **Failure Risk** — select Engine 42. Show ~95% failure probability, HIGH risk badge, red alert.

6. Go to **Anomaly Detection** — run on Engine 42. Show anomaly timeline — anomalous cycles in final stretch.

7. Go to **Maintenance Planner** — generate recommendation for Engine 42. Show CRITICAL priority, "Immediate engineering review." Then Engine 1 — NORMAL priority.

8. Go to **Model Insights** — show Table 3 with all measured metrics. Show `rul_model_results.png`. Explain LSTM beats RF because it uses sequences.

9. Open `verify_pipeline.py` output — show all 5 checks PASS. Confirms no hardcoding, no leakage, no invented metrics.

---

## 12. Questions You May Be Asked

**Q: Why only FD001 for model training?**  
A: FD001 is the standard benchmark — simplest subset (1 condition, 1 fault mode). This makes results comparable to published literature. The application supports browsing all four datasets, but clearly labels that predictions use FD001-trained models on FD002–FD004.

**Q: Why is your prediction sometimes far off (like Engine 95, error = 33 cycles)?**  
A: The model uses linear uncapped RUL during training, which means it sometimes over-predicts RUL for engines that ran a long time before cutoff. The average error (MAE = 13.15 cycles) is well within acceptable range for this dataset. LSTM improves this with temporal context.

**Q: Why does LSTM beat Random Forest?**  
A: Random Forest uses a single snapshot (the last cycle's features). LSTM sees the last 30 cycles as a sequence — it can detect trends and rate-of-change patterns that a single-point model misses.

**Q: Is this SHAP or not?**  
A: Both. `generate_results.py` uses actual `shap.TreeExplainer` to produce Fig 7. The in-app per-engine explanation uses a contribution approximation (importance × |scaled value|) which is labelled "Local Feature Contribution" — not SHAP — to be accurate.

**Q: What does Recall = 1.000 mean for the failure classifier?**  
A: The Gradient Boosting classifier caught 100% of engines that were truly within 30 cycles of failure on the FD001 test set. In safety-critical maintenance, missing a failing engine is far more costly than a false alarm, so high recall is the priority metric.

**Q: What is the purpose of the rolling features?**  
A: A single sensor reading at one moment is noisy. Rolling mean over 5 and 10 cycles smooths the trend. Rolling standard deviation captures how volatile the sensor is — increasing volatility often indicates degradation. These dramatically improve prediction accuracy.

**Q: How does Isolation Forest work?**  
A: It builds 100 random decision trees. For each data point, it counts how many splits are needed to isolate it. Anomalous points are unusual — they're far from the main cluster, so they get isolated quickly with fewer splits. A low decision_function score means very few splits needed = anomalous.

---

## 13. File Locations Quick Reference

```
INTERNSHIP PROJECT F/
│
├── NASATurbofanEnginePredictiveMaintenance/
│   ├── v1.py                        Original training pipeline
│   ├── save_models.py               Trains and saves 4 sklearn models
│   ├── train_lstm.py                LSTM training (epochs=25, window=30)
│   ├── generate_results.py          Generates Figs 2–8 + Table 3
│   ├── generate_architecture.py     Generates Fig 1
│   ├── verify_pipeline.py           5-check verification (all PASS)
│   ├── integration_notes.md         Technical documentation
│   ├── models/
│   │   ├── rul_rf_fd001.joblib      Random Forest RUL model
│   │   ├── rul_gb_fd001.joblib      Gradient Boosting RUL model
│   │   ├── failure_rf_fd001.joblib  RF failure classifier
│   │   ├── failure_gb_fd001.joblib  GB failure classifier (preferred)
│   │   ├── scaler_fd001.joblib      StandardScaler (fitted on train)
│   │   ├── feature_cols_fd001.json  Ordered 150 feature names
│   │   └── model_metrics_fd001.json Measured MAE/RMSE/R²/F1/AUC
│   └── results_figures/
│       ├── fig1_system_architecture.png
│       ├── fig2_dataset_exploration.png
│       ├── fig3_sensor_degradation.png
│       ├── fig4_actual_vs_predicted.png
│       ├── fig4b_lstm_actual_vs_predicted.png
│       ├── fig5_model_comparison_updated.png
│       ├── fig6_anomaly_detection.png
│       ├── fig7_shap_analysis.png
│       ├── fig8_health_dashboard.png
│       ├── fig_lstm_training_curve.png
│       ├── table3_metrics.json
│       └── shap_sensor_ranking.json
│
└── AeroSentinel/
    ├── backend/                     FastAPI + ML inference + SQLite
    └── frontend/                    React + TypeScript dashboard
```

---

*AeroSentinel — Predict Before Failure.*  
*Built on NASA C-MAPSS · Research Grade · Not for operational aviation use.*
