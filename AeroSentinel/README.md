# AeroSentinel
## Intelligent Engine Health Monitoring & Predictive Maintenance Platform
**Tagline:** Predict Before Failure.

AeroSentinel is an AI-powered predictive maintenance platform built on the NASA C-MAPSS Turbofan Engine Degradation Dataset. It provides RUL estimation, failure risk prediction, anomaly detection, sensor intelligence, and maintenance recommendations through a cinematic aerospace-grade interface.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm

### 1. Save ML Models (first time only)
```powershell
cd "NASATurbofanEnginePredictiveMaintenance"
python save_models.py
```
This trains and saves all 4 model artifacts to `models/`.

### 2. Start Backend
```powershell
cd "AeroSentinel\backend"
pip install -r requirements.txt
python run.py
```
Backend runs at: **http://localhost:8000**
API docs: **http://localhost:8000/api/docs**

### 3. Start Frontend
```powershell
cd "AeroSentinel\frontend"
npm install
npm run dev
```
Frontend runs at: **http://localhost:5173**

---

## Verified FD001 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| RUL — Random Forest | 13.15 cycles | 18.40 | 0.804 |
| RUL — Gradient Boosting | 14.15 cycles | 19.38 | 0.782 |

| Classifier | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Failure — Gradient Boosting | 0.893 | 1.000 | 0.943 | 0.998 |
| Failure — Random Forest | 0.923 | 0.960 | 0.941 | 0.993 |

Source: FD001 Existing Training Run — 100 test engines.

---

## Features

1. **RUL Prediction** — Remaining Useful Life estimation per engine
2. **Failure Risk** — Binary failure within 30 cycles classification
3. **Anomaly Detection** — Isolation Forest sensor pattern analysis
4. **Sensor Intelligence** — Feature importance and local explanations
5. **Maintenance Planner** — Rule-based recommendations with SQLite persistence
6. **Dataset Explorer** — Interactive explorer for all 4 C-MAPSS subsets
7. **Model Insights** — Full evaluation results with reference visualisations
8. **Report Export** — CSV and PDF engine health reports

---

## Architecture

```
AeroSentinel/
├── backend/           FastAPI + SQLite + scikit-learn
│   ├── app/
│   │   ├── api/       Route handlers
│   │   ├── ml/        ML inference layer
│   │   ├── services/  Business logic
│   │   ├── models/    SQLAlchemy ORM
│   │   └── schemas/   Pydantic models
│   └── run.py
└── frontend/          React + TypeScript + Vite + Tailwind
    └── src/
        ├── pages/     10 application pages
        ├── components/ Layout, UI, charts
        └── services/  Axios API client

NASATurbofanEnginePredictiveMaintenance/
├── v1.py              Original training pipeline (preserved)
├── save_models.py     Model export script
└── models/            Saved ML artifacts
```

---

## Disclaimer

AeroSentinel uses the NASA C-MAPSS dataset for research and educational purposes. All predictions are research-grade ML outputs. This platform is not intended for operational aviation maintenance decisions.
