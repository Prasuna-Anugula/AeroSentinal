"""
AeroSentinel — FastAPI Application Entry Point
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_ORIGINS, STATIC_DIR
from app.database import init_db
from app.ml.model_loader import ModelRegistry

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.health       import router as health_router
from app.api.dashboard    import router as dashboard_router
from app.api.datasets     import router as datasets_router
from app.api.predictions  import router as predictions_router
from app.api.anomalies    import router as anomalies_router
from app.api.explanations import router as explanations_router
from app.api.maintenance  import router as maintenance_router
from app.api.models       import router as models_router
from app.api.reports      import router as reports_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("aerosentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("AeroSentinel API starting …")
    init_db()
    logger.info("Database initialised.")
    reg = ModelRegistry.get()
    reg.load()
    logger.info("ML models loaded.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("AeroSentinel API shutting down.")


app = FastAPI(
    title       = "AeroSentinel API",
    description = "Intelligent Engine Health Monitoring & Predictive Maintenance Platform",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
    openapi_url = "/api/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Static files (PNGs) ───────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Routes ────────────────────────────────────────────────────────────────────
PREFIX = "/api"
app.include_router(health_router,       prefix=PREFIX)
app.include_router(dashboard_router,    prefix=PREFIX)
app.include_router(datasets_router,     prefix=PREFIX)
app.include_router(predictions_router,  prefix=PREFIX)
app.include_router(anomalies_router,    prefix=PREFIX)
app.include_router(explanations_router, prefix=PREFIX)
app.include_router(maintenance_router,  prefix=PREFIX)
app.include_router(models_router,       prefix=PREFIX)
app.include_router(reports_router,      prefix=PREFIX)


@app.get("/")
def root():
    return {
        "product":  "AeroSentinel",
        "tagline":  "Predict Before Failure.",
        "docs":     "/api/docs",
        "status":   "/api/health",
    }
