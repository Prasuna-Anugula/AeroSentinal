"""
AeroSentinel — Fig 1: Proposed System Architecture
====================================================
Generates a publication-quality architecture diagram showing the
complete system flow:
  Data Collection → Preprocessing → Feature Engineering →
  RUL Prediction → Anomaly Detection → SHAP Explainability →
  Health Assessment → Maintenance Recommendation
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path('results_figures')
OUT.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = '#070B16'
BG2   = '#0D1628'
SURF  = '#111B2F'
SURF2 = '#162035'

PRI   = '#69C9FF'   # blue  — data / input
SEC   = '#8B83FF'   # violet — ML models
ACC   = '#36E0C1'   # teal  — outputs
WRN   = '#FFBF69'   # amber — health
DNG   = '#FF647C'   # red   — critical
SUC   = '#39D98A'   # green — recommendation
TXT   = '#F4F7FF'
TXT2  = '#9AAAC2'
BORDER= '#1E3050'

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

# ── Helpers ───────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, fc, ec, radius=0.35, alpha=0.92, zorder=3):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f'round,pad={radius}',
        facecolor=fc, edgecolor=ec,
        linewidth=1.4, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    return box

def label(ax, x, y, text, size=9, color=TXT, weight='bold', zorder=5, ha='center', va='center'):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            fontfamily='DejaVu Sans')

def sublabel(ax, x, y, text, size=7.2, color=TXT2, zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight='normal',
            ha='center', va='center', zorder=zorder,
            fontfamily='DejaVu Sans', style='italic')

def arrow(ax, x1, y1, x2, y2, col=TXT2, lw=1.6, zorder=2, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=col,
                    lw=lw,
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=zorder)

def dashed_arrow(ax, x1, y1, x2, y2, col=TXT2, lw=1.2, zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->',
                    color=col,
                    lw=lw,
                    linestyle='dashed',
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=zorder)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(11, 13.4, 'AeroSentinel — Intelligent Predictive Maintenance System',
        fontsize=15, color=TXT, fontweight='bold', ha='center', va='center',
        fontfamily='DejaVu Sans', zorder=6)
ax.text(11, 12.95, 'Fig. 1  ·  Proposed Architecture  ·  NASA C-MAPSS FD001–FD004',
        fontsize=9, color=TXT2, ha='center', va='center',
        fontfamily='DejaVu Sans', zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Data layer  (y = 11.5)
# ══════════════════════════════════════════════════════════════════════════════
R1Y = 11.2
BOX_H = 1.15

# NASA C-MAPSS source
rounded_box(ax, 3.5, R1Y, 3.8, BOX_H, SURF2, PRI+'88')
label(ax, 3.5, R1Y+0.22, '① DATA COLLECTION', size=8.5, color=PRI)
sublabel(ax, 3.5, R1Y-0.12, 'NASA C-MAPSS Dataset\nFD001–FD004  ·  21 sensors  ·  3 settings')
sublabel(ax, 3.5, R1Y-0.42, 'train / test / RUL files')

# Raw data description
rounded_box(ax, 8.5, R1Y, 4.0, BOX_H, SURF2, PRI+'55')
label(ax, 8.5, R1Y+0.22, 'RAW SENSOR READINGS', size=8.5, color=PRI)
sublabel(ax, 8.5, R1Y-0.12, 'unit · cycle · setting1-3')
sublabel(ax, 8.5, R1Y-0.42, 'sensor1 … sensor21  |  26 cols × N rows')

# Preprocessing
rounded_box(ax, 14.0, R1Y, 4.0, BOX_H, SURF2, SEC+'88')
label(ax, 14.0, R1Y+0.22, '② PREPROCESSING', size=8.5, color=SEC)
sublabel(ax, 14.0, R1Y-0.12, 'Whitespace parsing  ·  Column naming')
sublabel(ax, 14.0, R1Y-0.42, 'Sort by unit+cycle  ·  Null check  ·  Scaling')

# Feature Engineering
rounded_box(ax, 19.5, R1Y, 4.2, BOX_H, SURF2, SEC+'88')
label(ax, 19.5, R1Y+0.22, '③ FEATURE ENGINEERING', size=8.5, color=SEC)
sublabel(ax, 19.5, R1Y-0.12, 'RUL target (max_cycle − cycle)')
sublabel(ax, 19.5, R1Y-0.42, 'Rolling mean/std (w=5,10)  ·  Δ from start  ·  Norm')

# Arrows row 1
arrow(ax, 5.4,  R1Y, 6.5,  R1Y, PRI, lw=2.0)
arrow(ax, 10.5, R1Y, 12.0, R1Y, SEC, lw=2.0)
arrow(ax, 16.0, R1Y, 17.4, R1Y, SEC, lw=2.0)

# ── down arrow from FE to ML row ──
arrow(ax, 19.5, R1Y - BOX_H/2 - 0.05, 19.5, 9.3, SEC, lw=1.8)
ax.text(19.85, 9.9, '150 features', fontsize=7, color=SEC, ha='left', va='center')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — ML Models  (y = 8.5)
# ══════════════════════════════════════════════════════════════════════════════
R2Y = 8.5
ML_W = 3.4

# StandardScaler
rounded_box(ax, 3.0, R2Y, 3.2, BOX_H, SURF2, SEC+'66')
label(ax, 3.0, R2Y+0.20, 'STANDARD SCALER', size=8.2, color=SEC)
sublabel(ax, 3.0, R2Y-0.15, 'Fitted on train set')
sublabel(ax, 3.0, R2Y-0.42, 'Applied to test set')

# RUL Prediction block
rounded_box(ax, 8.0, R2Y, 4.2, BOX_H, '#0E2040', PRI)
label(ax, 8.0, R2Y+0.25, '④ RUL PREDICTION', size=8.8, color=PRI)
sublabel(ax, 8.0, R2Y-0.05, 'Random Forest   MAE 13.15  R² 0.804')
sublabel(ax, 8.0, R2Y-0.30, 'Gradient Boosting  MAE 14.15  R² 0.782')
sublabel(ax, 8.0, R2Y-0.52, 'XGBoost              MAE 14.91  R² 0.756')

# Anomaly Detection block
rounded_box(ax, 13.5, R2Y, 4.0, BOX_H, '#0E2040', WRN)
label(ax, 13.5, R2Y+0.25, '⑤ ANOMALY DETECTION', size=8.8, color=WRN)
sublabel(ax, 13.5, R2Y-0.05, 'Isolation Forest  ·  contamination=5%')
sublabel(ax, 13.5, R2Y-0.30, 'Score per cycle  ·  Fleet-level flagging')
sublabel(ax, 13.5, R2Y-0.52, 'Thresholds: −0.10 unusual  / −0.20 anomaly')

# SHAP block
rounded_box(ax, 19.2, R2Y, 4.0, BOX_H, '#0E2040', ACC)
label(ax, 19.2, R2Y+0.25, '⑥ SHAP EXPLAINABILITY', size=8.8, color=ACC)
sublabel(ax, 19.2, R2Y-0.05, 'TreeExplainer on Random Forest')
sublabel(ax, 19.2, R2Y-0.30, 'Mean |SHAP| per feature  ·  Sensor ranking')
sublabel(ax, 19.2, R2Y-0.52, 'Local (per-engine) contribution')

# Arrows row 2
arrow(ax, 4.6,  R2Y, 5.9,  R2Y, PRI,  lw=2.0)
arrow(ax, 10.1, R2Y, 11.5, R2Y, WRN,  lw=2.0)
arrow(ax, 15.5, R2Y, 17.2, R2Y, ACC,  lw=2.0)

# Scaler feeds into ML
arrow(ax, 3.0, R2Y - BOX_H/2 - 0.05, 3.0, 6.15, TXT2, lw=1.4)
dashed_arrow(ax, 3.0, 6.2, 8.0, 6.2, TXT2, lw=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Assessment & Output  (y = 5.7)
# ══════════════════════════════════════════════════════════════════════════════
R3Y = 5.7

# Health Assessment
rounded_box(ax, 5.8, R3Y, 4.8, BOX_H, '#0E2040', WRN+'CC')
label(ax, 5.8, R3Y+0.25, '⑦ HEALTH ASSESSMENT', size=8.8, color=WRN)
sublabel(ax, 5.8, R3Y-0.05, 'Healthy (RUL > 100)  ·  Warning (50–100)')
sublabel(ax, 5.8, R3Y-0.30, 'High Risk (20–49)  ·  Critical (< 20)')
sublabel(ax, 5.8, R3Y-0.52, 'AeroSentinel project-defined thresholds')

# Maintenance Recommendation
rounded_box(ax, 13.5, R3Y, 5.0, BOX_H, '#0E2040', SUC+'CC')
label(ax, 13.5, R3Y+0.25, '⑧ MAINTENANCE RECOMMENDATION', size=8.8, color=SUC)
sublabel(ax, 13.5, R3Y-0.05, 'Rule engine: RUL + fail_prob + anomaly_score')
sublabel(ax, 13.5, R3Y-0.30, 'Priority: Critical / High / Medium / Normal')
sublabel(ax, 13.5, R3Y-0.52, 'Persisted in SQLite  ·  Status: pending→resolved')

# Arrows feeding row 3
# RUL prediction → Health Assessment
arrow(ax, 8.0, R2Y - BOX_H/2 - 0.05, 8.0, R3Y + BOX_H/2 + 0.08, PRI, lw=1.8)
# Anomaly → Health
arrow(ax, 13.5, R2Y - BOX_H/2 - 0.05, 13.5, R3Y + BOX_H/2 + 0.35, WRN, lw=1.4)
ax.annotate('', xy=(8.5, R3Y + BOX_H/2 + 0.05),
            xytext=(13.0, R3Y + BOX_H/2 + 0.35),
            arrowprops=dict(arrowstyle='->', color=WRN, lw=1.2,
                            connectionstyle='arc3,rad=-0.2'), zorder=2)
# SHAP → Health
dashed_arrow(ax, 19.2, R2Y - BOX_H/2 - 0.05, 19.2, R3Y + 0.3, ACC, lw=1.2)
ax.annotate('', xy=(8.9, R3Y + 0.1),
            xytext=(19.0, R3Y + 0.3),
            arrowprops=dict(arrowstyle='->', color=ACC, lw=1.1,
                            linestyle='dashed',
                            connectionstyle='arc3,rad=-0.25'), zorder=2)
# Health → Recommendation
arrow(ax, 8.3, R3Y, 11.0, R3Y, SUC, lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — AeroSentinel dashboard output  (y = 3.5)
# ══════════════════════════════════════════════════════════════════════════════
R4Y = 3.5

rounded_box(ax, 11, R4Y, 14.5, 1.2, '#0A1828', PRI+'44', radius=0.5)
label(ax, 11, R4Y+0.28, 'AEROSENTINEL MISSION CONTROL DASHBOARD', size=10, color=PRI)
sublabel(ax, 11, R4Y-0.05,
         'React + TypeScript frontend  ·  FastAPI backend  ·  SQLite persistence')
sublabel(ax, 11, R4Y-0.35,
         'Overview  ·  Engine Monitor  ·  RUL Prediction  ·  Failure Risk  ·  Anomaly  ·  Sensor Intelligence  ·  Maintenance Planner  ·  Dataset Explorer  ·  Model Insights')

# Arrows feeding dashboard
arrow(ax, 5.8,  R3Y - BOX_H/2 - 0.05, 5.8,  R4Y + 0.62, WRN, lw=1.8)
arrow(ax, 13.5, R3Y - BOX_H/2 - 0.05, 13.5, R4Y + 0.62, SUC, lw=1.8)
# merge lines
ax.annotate('', xy=(11, R4Y + 0.62),
            xytext=(5.8, R4Y + 0.62),
            arrowprops=dict(arrowstyle='-', color=WRN, lw=1.2), zorder=1)
ax.annotate('', xy=(11, R4Y + 0.62),
            xytext=(13.5, R4Y + 0.62),
            arrowprops=dict(arrowstyle='-', color=SUC, lw=1.2), zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Outputs  (y = 1.8)
# ══════════════════════════════════════════════════════════════════════════════
R5Y = 1.85
OUTPUT_BOXES = [
    (4.0,   'PREDICTED RUL',         PRI, 'RF/GB/XGBoost\ncycles remaining'),
    (8.0,   'FAILURE RISK',          DNG, 'Probability + class\nwithin 30 cycles'),
    (12.0,  'ANOMALY STATUS',        WRN, 'Score + severity\nper engine/cycle'),
    (16.0,  'SENSOR EXPLANATION',    ACC, 'SHAP contribution\ntop sensors'),
    (20.0,  'MAINTENANCE ACTION',    SUC, 'Priority + evidence\nSQLite audit trail'),
]

for xc, title, col, sub in OUTPUT_BOXES:
    rounded_box(ax, xc, R5Y, 3.5, 1.1, SURF2, col+'99')
    label(ax, xc, R5Y+0.28, title, size=7.8, color=col)
    sublabel(ax, xc, R5Y-0.18, sub, size=7.0)
    arrow(ax, xc, R4Y - 0.62, xc, R5Y + 0.57, col, lw=1.6)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND / Key
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (PRI, 'Data / Input layer'),
    (SEC, 'Preprocessing / Feature Eng.'),
    (WRN, 'Anomaly / Health Assessment'),
    (ACC, 'Explainability (SHAP)'),
    (SUC, 'Recommendation / Output'),
]
lx, ly = 0.55, 2.6
ax.text(lx, ly + 0.25, 'Legend', fontsize=8, color=TXT, fontweight='bold', ha='left')
for i, (col, lbl) in enumerate(legend_items):
    rect = FancyBboxPatch((lx, ly - i*0.42 - 0.08), 0.28, 0.22,
                           boxstyle='round,pad=0.04',
                           facecolor=col+'25', edgecolor=col, lw=1.0)
    ax.add_patch(rect)
    ax.text(lx + 0.38, ly - i*0.42 + 0.03, lbl, fontsize=7.2, color=TXT2, ha='left', va='center')

# ══════════════════════════════════════════════════════════════════════════════
# Flow label strip
# ══════════════════════════════════════════════════════════════════════════════
flow_steps = ['Data\nCollection', 'Pre-\nprocessing', 'Feature\nEngineering',
              'RUL\nPrediction', 'Anomaly\nDetection', 'SHAP\nExplainability',
              'Health\nAssessment', 'Maintenance\nRecommendation']

for i, step in enumerate(flow_steps):
    xp = 1.5 + i * 2.72
    ax.text(xp, 0.55, step, fontsize=6.8, color=TXT2, ha='center', va='center',
            fontfamily='DejaVu Sans')
    if i < len(flow_steps) - 1:
        ax.annotate('', xy=(xp + 2.3, 0.55), xytext=(xp + 0.6, 0.55),
                    arrowprops=dict(arrowstyle='->', color=TXT2+'88', lw=0.9), zorder=1)

ax.axhline(0.85, xmin=0.02, xmax=0.98, color=BORDER, lw=0.8, zorder=1)
ax.text(11, 0.22, 'AeroSentinel  ·  Intelligent Engine Health Monitoring & Predictive Maintenance Platform  '
        '·  NASA C-MAPSS Dataset  ·  Research Grade',
        fontsize=7, color=TXT2+'66', ha='center', va='center')

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
out_path = OUT / 'fig1_system_architecture.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG, pad_inches=0.15)
plt.close()
print(f'Saved: {out_path.resolve()}')
