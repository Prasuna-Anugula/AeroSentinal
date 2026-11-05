from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.ml.rul_predictor     import predict_single_engine
from app.ml.failure_predictor import predict_failure_single
from app.ml.anomaly_detector  import detect_anomalies_engine
from app.services.maintenance_service import _build_recommendation
from app.services.report_service      import generate_csv_report, generate_pdf_report

router = APIRouter(prefix="/reports")


def _collect_engine_data(dataset: str, engine_unit: int) -> dict:
    rul    = predict_single_engine(dataset, engine_unit)
    fail   = predict_failure_single(dataset, engine_unit)
    try:
        anom = detect_anomalies_engine(dataset, engine_unit)
        a_score = anom["overall_score"]
        a_sev   = anom["severity"]
    except Exception:
        a_score = None
        a_sev   = None

    rec = _build_recommendation(
        predicted_rul    = rul["predicted_rul"],
        failure_prob     = fail["failure_probability"],
        anomaly_score    = a_score,
        anomaly_severity = a_sev,
    )

    return {
        "engine_unit":         engine_unit,
        "dataset":             dataset,
        "last_cycle":          rul["last_cycle"],
        "health_status":       rul["health_status"],
        "predicted_rul":       rul["predicted_rul"],
        "true_rul":            rul.get("true_rul"),
        "rul_model":           rul["model"],
        "failure_probability": fail["failure_probability"],
        "risk_level":          fail["risk_level"],
        "fail_model":          fail["model"],
        "anomaly_score":       a_score,
        "anomaly_severity":    a_sev,
        "priority":            rec["priority"],
        "recommendation":      rec["recommendation"],
    }


@router.get("/engine/csv")
def engine_csv_report(
    dataset:     str = Query("FD001"),
    engine_unit: int = Query(...),
):
    try:
        data = _collect_engine_data(dataset, engine_unit)
    except Exception as e:
        raise HTTPException(500, str(e))

    content = generate_csv_report(data)
    filename = f"aerosentinel_{dataset}_engine{engine_unit}.csv"
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/engine/pdf")
def engine_pdf_report(
    dataset:     str = Query("FD001"),
    engine_unit: int = Query(...),
):
    try:
        data = _collect_engine_data(dataset, engine_unit)
    except Exception as e:
        raise HTTPException(500, str(e))

    content  = generate_pdf_report(data)
    filename = f"aerosentinel_{dataset}_engine{engine_unit}.pdf"
    media    = "application/pdf" if content[:4] == b"%PDF" else "text/csv"
    ext      = "pdf" if media == "application/pdf" else "csv"
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="aerosentinel_{dataset}_engine{engine_unit}.{ext}"'},
    )
