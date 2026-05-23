"""
Agente Orquestador
===================
Coordina la ejecución secuencial: EDA → Limpieza → Predicción.

Expone un generador run_pipeline_stream() que emite PipelineEvent objects
para consumo en tiempo real desde Streamlit (st.status).
"""

from __future__ import annotations

import datetime
import traceback
from typing import Generator

import pandas as pd

from .events import PipelineEvent
from .eda_agent import run_eda
from .cleaning_agent import run_cleaning
from .prediction_agent import run_prediction

# ── ADK (opcional) ────────────────────────────────────────────────────────────
try:
    from google.adk.agents import SequentialAgent
    from google.adk.tools import FunctionTool
    _eda_tool        = FunctionTool(run_eda)
    _cleaning_tool   = FunctionTool(run_cleaning)
    _prediction_tool = FunctionTool(run_prediction)
    retail_pipeline_adk = SequentialAgent(
        name="RetailPipelineOrchestrator",
        description="Pipeline secuencial: EDA → Limpieza → Predicción.",
        sub_agents=[],
    )
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    retail_pipeline_adk = None


# ---------------------------------------------------------------------------
# Generador principal  (consumido por Streamlit con st.status)
# ---------------------------------------------------------------------------

def run_pipeline_stream(
    df: pd.DataFrame,
    sales_col: str = None,
    cat_col: str = None,
) -> Generator[PipelineEvent, None, None]:
    """
    Generador que ejecuta EDA → Limpieza → Predicción emitiendo PipelineEvent
    en cada hito. El caller hace:

        for event in run_pipeline_stream(df):
            # actualizar UI

    El último evento es de tipo "final" y su `data` contiene el resultado
    completo del pipeline (equivalente al dict que antes devolvía run_pipeline).
    """

    logs: list[str] = []
    pipeline_result: dict = {
        "pipeline_status": {},
        "adk_available":   ADK_AVAILABLE,
        "logs":            logs,
        "sales_col_used":  sales_col,
        "cat_col_used":    cat_col,
    }

    def _log(level: str, agent: str, msg: str) -> str:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level:5s}] [{agent}] {msg}"
        logs.append(line)
        return line

    def _event(agent: str, etype: str, msg: str, data: dict = None) -> PipelineEvent:
        log_line = _log(
            {"start": "INFO", "progress": "INFO", "done": "OK   ",
             "error": "ERROR", "final": "INFO"}.get(etype, "INFO"),
            agent, msg,
        )
        ev = PipelineEvent(agent=agent, type=etype, msg=msg, data=data or {})
        ev._log_line = log_line   # adjuntar línea de log al evento
        return ev

    # ── Info inicial ──────────────────────────────────────────────────────
    _log("INFO", "ORQUESTADOR", f"Pipeline iniciado — {df.shape[0]:,} × {df.shape[1]}")
    _log("INFO", "ORQUESTADOR", f"ADK disponible: {ADK_AVAILABLE}")
    _log("INFO", "ORQUESTADOR", f"Columna objetivo: {sales_col or 'auto-detectar'}")
    _log("INFO", "ORQUESTADOR", f"Columna categoria: {cat_col or 'auto-detectar'}")

    # ── AGENTE 1: EDA ─────────────────────────────────────────────────────
    yield _event("EDA", "start", "Agente 1 – Análisis Exploratorio (EDA)")

    eda_events: list[str] = []
    try:
        eda_result = run_eda(
            df,
            sales_col=sales_col,
            cat_col=cat_col,
            emit=lambda msg: eda_events.append(msg),
        )
        # Vaciar buffer de sub-eventos
        for msg in eda_events:
            yield _event("EDA", "progress", msg)

        pipeline_result["eda"] = eda_result
        pipeline_result["pipeline_status"]["eda"] = eda_result.get("status", "OK")

        if not sales_col:
            sales_col = eda_result.get("sales_col")
            pipeline_result["sales_col_used"] = sales_col
        if not cat_col:
            cat_col = eda_result.get("cat_col")
            pipeline_result["cat_col_used"] = cat_col

        yield _event("EDA", "done", "EDA completado", data={"sales_col": sales_col})

    except Exception as exc:
        tb = traceback.format_exc()
        _log("ERROR", "AGENTE-1 EDA", str(exc))
        _log("ERROR", "AGENTE-1 EDA", tb)
        pipeline_result["eda"] = {"error": str(exc)}
        pipeline_result["pipeline_status"]["eda"] = f"Error: {exc}"
        yield _event("EDA", "error", f"Error en EDA: {exc}")
        yield _event("Pipeline", "final", "Pipeline abortado por error en EDA",
                     data=pipeline_result)
        return

    # ── AGENTE 2: Limpieza ────────────────────────────────────────────────
    yield _event("Limpieza", "start", "Agente 2 – Limpieza de Datos")

    cleaning_events: list[str] = []
    try:
        cleaning_result = run_cleaning(
            df,
            emit=lambda msg: cleaning_events.append(msg),
        )
        for msg in cleaning_events:
            yield _event("Limpieza", "progress", msg)

        pipeline_result["cleaning"] = cleaning_result
        pipeline_result["pipeline_status"]["cleaning"] = cleaning_result.get("status", "OK")
        df_clean = cleaning_result["df_clean"]

        yield _event("Limpieza", "done", "Limpieza completada",
                     data={"shape": df_clean.shape})

    except Exception as exc:
        tb = traceback.format_exc()
        _log("ERROR", "AGENTE-2 LIMPIEZA", str(exc))
        _log("ERROR", "AGENTE-2 LIMPIEZA", tb)
        pipeline_result["cleaning"] = {"error": str(exc)}
        pipeline_result["pipeline_status"]["cleaning"] = f"Error: {exc}"
        yield _event("Limpieza", "error", f"Error en Limpieza: {exc}")
        df_clean = df   # fallback: continuar con df original

    # ── AGENTE 3: Predicción ──────────────────────────────────────────────
    yield _event("Prediccion", "start", "Agente 3 – Entrenamiento de Modelos")

    prediction_events: list[str] = []
    try:
        prediction_result = run_prediction(
            df_clean,
            sales_col=sales_col,
            cat_col=cat_col,
            emit=lambda msg: prediction_events.append(msg),
        )
        for msg in prediction_events:
            yield _event("Prediccion", "progress", msg)

        pipeline_result["prediction"] = prediction_result
        pipeline_result["pipeline_status"]["prediction"] = prediction_result.get("status", "OK")

        if not prediction_result.get("error"):
            best = prediction_result.get("best_model_name", "–")
            yield _event("Prediccion", "done", f"Predicción completada — mejor modelo: {best}",
                         data={"best_model": best})
        else:
            yield _event("Prediccion", "error",
                         f"Error en Predicción: {prediction_result.get('status')}")

    except Exception as exc:
        tb = traceback.format_exc()
        _log("ERROR", "AGENTE-3 PREDICCION", str(exc))
        _log("ERROR", "AGENTE-3 PREDICCION", tb)
        pipeline_result["prediction"] = {"error": str(exc)}
        pipeline_result["pipeline_status"]["prediction"] = f"Error: {exc}"
        yield _event("Prediccion", "error", f"Error en Predicción: {exc}")

    # ── Evento final ──────────────────────────────────────────────────────
    _log("INFO", "ORQUESTADOR", "Pipeline finalizado")
    yield _event("Pipeline", "final", "Pipeline completado", data=pipeline_result)
