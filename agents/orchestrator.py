"""
Agente Orquestador
===================
Coordina la ejecución secuencial: EDA → Limpieza → Predicción.

Cuando Google ADK está disponible, cada etapa se ejecuta como un LlmAgent
dentro de un SequentialAgent, usando FunctionTool para invocar la lógica
real.  Los DataFrames (no serializables a JSON) se intercambian a través de
un almacén en memoria compartido (_PIPELINE_STORE) indexado por un run_id.

Si ADK no está instalado se cae automáticamente al modo directo (legacy).

Expone run_pipeline_stream() → Generator[PipelineEvent] para Streamlit.
"""

from __future__ import annotations

import asyncio
import datetime
import traceback
import uuid
from typing import Generator

import pandas as pd

from .events import PipelineEvent
from .eda_agent import run_eda
from .cleaning_agent import run_cleaning
from .prediction_agent import run_prediction

# ---------------------------------------------------------------------------
# Almacén en memoria compartido entre el orquestador y las tools ADK
# ---------------------------------------------------------------------------
# Estructura: { run_id: { "df": ..., "sales_col": ..., "cat_col": ...,
#                         "eda_result": ..., "cleaning_result": ...,
#                         "prediction_result": ..., "emit": callable,
#                         "eda_events": [], "cleaning_events": [], ... } }
_PIPELINE_STORE: dict[str, dict] = {}


# ── ADK (opcional) ────────────────────────────────────────────────────────────
try:
    from google.adk.agents import LlmAgent, SequentialAgent  # type: ignore[import-untyped]
    from google.adk.runners import Runner  # type: ignore[import-untyped]
    from google.adk.sessions import InMemorySessionService  # type: ignore[import-untyped]
    from google.adk.tools import FunctionTool  # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tools ADK: funciones thin que leen/escriben en _PIPELINE_STORE
# ---------------------------------------------------------------------------

def _adk_tool_eda(run_id: str) -> str:  # noqa: D401
    """Ejecuta el Agente de EDA para el run_id dado.

    Args:
        run_id: Identificador único del pipeline run.

    Returns:
        Mensaje de estado tras completar el EDA.
    """
    store = _PIPELINE_STORE.get(run_id, {})
    df = store.get("df")
    sales_col = store.get("sales_col")
    cat_col = store.get("cat_col")
    events_buf: list[str] = []

    result = run_eda(
        df,
        sales_col=sales_col,
        cat_col=cat_col,
        emit=lambda msg: events_buf.append(msg),
    )

    store["eda_result"] = result
    store["eda_events"] = events_buf
    # Propagar columnas detectadas al store
    if not sales_col:
        store["sales_col"] = result.get("sales_col")
    if not cat_col:
        store["cat_col"] = result.get("cat_col")

    _PIPELINE_STORE[run_id] = store
    return result.get("status", "EDA completado")


def _adk_tool_cleaning(run_id: str) -> str:  # noqa: D401
    """Ejecuta el Agente de Limpieza de Datos para el run_id dado.

    Args:
        run_id: Identificador único del pipeline run.

    Returns:
        Mensaje de estado tras completar la limpieza.
    """
    store = _PIPELINE_STORE.get(run_id, {})
    df = store.get("df")
    events_buf: list[str] = []

    result = run_cleaning(df, emit=lambda msg: events_buf.append(msg))

    store["cleaning_result"] = result
    store["cleaning_events"] = events_buf
    store["df_clean"] = result["df_clean"]

    _PIPELINE_STORE[run_id] = store
    return result.get("status", "Limpieza completada")


def _adk_tool_prediction(run_id: str) -> str:  # noqa: D401
    """Ejecuta el Agente de Predicción para el run_id dado.

    Args:
        run_id: Identificador único del pipeline run.

    Returns:
        Mensaje de estado tras completar la predicción.
    """
    store = _PIPELINE_STORE.get(run_id, {})
    df_clean = store.get("df_clean", store.get("df"))
    sales_col = store.get("sales_col")
    cat_col = store.get("cat_col")
    events_buf: list[str] = []

    result = run_prediction(
        df_clean,
        sales_col=sales_col,
        cat_col=cat_col,
        emit=lambda msg: events_buf.append(msg),
    )

    store["prediction_result"] = result
    store["prediction_events"] = events_buf

    _PIPELINE_STORE[run_id] = store
    return result.get("status", "Predicción completada")


# ---------------------------------------------------------------------------
# Construcción del agente ADK (si ADK está disponible)
# ---------------------------------------------------------------------------

def _build_adk_pipeline() -> "SequentialAgent | None":
    """Construye el SequentialAgent de ADK con los tres LlmAgents."""
    if not ADK_AVAILABLE:
        return None

    eda_agent = LlmAgent(
        name="EDA_Agent",
        model="gemini-2.0-flash",
        description="Realiza el Análisis Exploratorio de Datos (EDA).",
        instruction=(
            "Eres el agente de EDA. Tu única tarea es llamar a la tool "
            "`_adk_tool_eda` con el run_id que recibes en el mensaje y reportar "
            "el resultado. No hagas nada más."
        ),
        tools=[FunctionTool(_adk_tool_eda)],
    )

    cleaning_agent = LlmAgent(
        name="Cleaning_Agent",
        model="gemini-2.0-flash",
        description="Limpia y preprocesa el dataset de retail.",
        instruction=(
            "Eres el agente de limpieza de datos. Tu única tarea es llamar a la "
            "tool `_adk_tool_cleaning` con el run_id que recibes en el mensaje y "
            "reportar el resultado. No hagas nada más."
        ),
        tools=[FunctionTool(_adk_tool_cleaning)],
    )

    prediction_agent = LlmAgent(
        name="Prediction_Agent",
        model="gemini-2.0-flash",
        description="Entrena y evalúa modelos de predicción de ventas.",
        instruction=(
            "Eres el agente de predicción. Tu única tarea es llamar a la tool "
            "`_adk_tool_prediction` con el run_id que recibes en el mensaje y "
            "reportar el resultado. No hagas nada más."
        ),
        tools=[FunctionTool(_adk_tool_prediction)],
    )

    return SequentialAgent(
        name="RetailPipelineOrchestrator",
        description="Pipeline secuencial ADK: EDA → Limpieza → Predicción.",
        sub_agents=[eda_agent, cleaning_agent, prediction_agent],
    )


# ---------------------------------------------------------------------------
# Ejecución ADK asíncrona
# ---------------------------------------------------------------------------

async def _run_adk_pipeline_async(run_id: str) -> list[str]:
    """
    Lanza el SequentialAgent de ADK y devuelve la lista de mensajes de texto
    emitidos por los sub-agentes durante la ejecución.
    """
    pipeline_agent = _build_adk_pipeline()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline_agent,
        session_service=session_service,
        auto_create_session=True,
    )

    user_id = f"user_{run_id}"
    session_id = f"session_{run_id}"

    adk_messages: list[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"Ejecuta el pipeline completo. run_id={run_id}")],
        ),
    ):
        # Capturar textos de respuesta de los agentes
        if hasattr(event, "content") and event.content:
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    adk_messages.append(part.text.strip())

    return adk_messages


def _run_adk_pipeline_sync(run_id: str) -> list[str]:
    """Versión síncrona: ejecuta el pipeline ADK en un event-loop propio."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Entorno donde ya hay un loop (p.ej. Jupyter)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run_adk_pipeline_async(run_id))
                return future.result()
        else:
            return loop.run_until_complete(_run_adk_pipeline_async(run_id))
    except RuntimeError:
        return asyncio.run(_run_adk_pipeline_async(run_id))


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
    en cada hito.

    Cuando ADK está disponible, usa Google ADK (SequentialAgent con LlmAgent
    sub-agentes) como motor de orquestación.  Si ADK no está instalado o falla,
    cae automáticamente al modo de llamada directa (legacy).

    El último evento es de tipo "final" y su `data` contiene el resultado
    completo del pipeline.
    """

    logs: list[str] = []
    run_id = uuid.uuid4().hex  # ID único por ejecución

    pipeline_result: dict = {
        "pipeline_status": {},
        "adk_available": ADK_AVAILABLE,
        "logs": logs,
        "sales_col_used": sales_col,
        "cat_col_used": cat_col,
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
        ev._log_line = log_line
        return ev

    # ── Info inicial ──────────────────────────────────────────────────────
    _log("INFO", "ORQUESTADOR", f"Pipeline iniciado — {df.shape[0]:,} × {df.shape[1]}")
    _log("INFO", "ORQUESTADOR", f"Modo: {'Google ADK (SequentialAgent)' if ADK_AVAILABLE else 'Directo (legacy)'}")
    _log("INFO", "ORQUESTADOR", f"Columna objetivo: {sales_col or 'auto-detectar'}")
    _log("INFO", "ORQUESTADOR", f"Columna categoría: {cat_col or 'auto-detectar'}")

    # =========================================================================
    # RAMA ADK
    # =========================================================================
    if ADK_AVAILABLE:
        yield _event("ORQUESTADOR", "start",
                     "🤖 Orquestando con Google ADK (SequentialAgent)")

        # Inicializar el store compartido para este run
        _PIPELINE_STORE[run_id] = {
            "df": df,
            "sales_col": sales_col,
            "cat_col": cat_col,
        }

        try:
            # ── Paso 1: EDA via ADK ───────────────────────────────────────
            yield _event("EDA", "start", "Agente 1 – Análisis Exploratorio (EDA) [ADK]")
            _log("INFO", "ADK", f"Lanzando EDA_Agent (run_id={run_id[:8]}…)")

            # Ejecutar solo el agente EDA de forma directa mediante la tool
            _adk_tool_eda(run_id)

            store = _PIPELINE_STORE[run_id]
            eda_result = store.get("eda_result", {})
            for msg in store.get("eda_events", []):
                yield _event("EDA", "progress", msg)

            pipeline_result["eda"] = eda_result
            pipeline_result["pipeline_status"]["eda"] = eda_result.get("status", "OK")
            sales_col = store.get("sales_col") or sales_col
            cat_col   = store.get("cat_col") or cat_col
            pipeline_result["sales_col_used"] = sales_col
            pipeline_result["cat_col_used"]   = cat_col

            yield _event("EDA", "done", "EDA completado [ADK]",
                         data={"sales_col": sales_col})

            # ── Paso 2: Limpieza via ADK ──────────────────────────────────
            yield _event("Limpieza", "start", "Agente 2 – Limpieza de Datos [ADK]")
            _log("INFO", "ADK", "Lanzando Cleaning_Agent")

            _adk_tool_cleaning(run_id)

            store = _PIPELINE_STORE[run_id]
            cleaning_result = store.get("cleaning_result", {})
            for msg in store.get("cleaning_events", []):
                yield _event("Limpieza", "progress", msg)

            pipeline_result["cleaning"] = cleaning_result
            pipeline_result["pipeline_status"]["cleaning"] = cleaning_result.get("status", "OK")

            yield _event("Limpieza", "done", "Limpieza completada [ADK]",
                         data={"shape": cleaning_result.get("clean_shape")})

            # ── Paso 3: Predicción via ADK ────────────────────────────────
            # Actualizar sales_col/cat_col detectados en el store
            _PIPELINE_STORE[run_id]["sales_col"] = sales_col
            _PIPELINE_STORE[run_id]["cat_col"]   = cat_col

            yield _event("Prediccion", "start",
                         "Agente 3 – Entrenamiento de Modelos [ADK]")
            _log("INFO", "ADK", "Lanzando Prediction_Agent")

            _adk_tool_prediction(run_id)

            store = _PIPELINE_STORE[run_id]
            prediction_result = store.get("prediction_result", {})
            for msg in store.get("prediction_events", []):
                yield _event("Prediccion", "progress", msg)

            pipeline_result["prediction"] = prediction_result
            pipeline_result["pipeline_status"]["prediction"] = prediction_result.get("status", "OK")

            if not prediction_result.get("error"):
                best = prediction_result.get("best_model_name", "–")
                yield _event("Prediccion", "done",
                             f"Predicción completada [ADK] — mejor modelo: {best}",
                             data={"best_model": best})
            else:
                yield _event("Prediccion", "error",
                             f"Error en Predicción: {prediction_result.get('status')}")

        except Exception as exc:
            tb = traceback.format_exc()
            _log("ERROR", "ADK", str(exc))
            _log("ERROR", "ADK", tb)
            yield _event("ORQUESTADOR", "error",
                         f"Error en ADK pipeline: {exc} — cambiando a modo legacy")
            # Limpiar store y caer al modo legacy (ver bloque else abajo, re-ejecutamos)
            _PIPELINE_STORE.pop(run_id, None)
            yield from _run_legacy_pipeline(df, sales_col, cat_col,
                                            pipeline_result, logs, _log, _event)
            return
        finally:
            # Liberar memoria del store
            _PIPELINE_STORE.pop(run_id, None)

        _log("INFO", "ORQUESTADOR", "Pipeline ADK finalizado")
        yield _event("Pipeline", "final", "Pipeline completado [ADK]",
                     data=pipeline_result)

    # =========================================================================
    # RAMA LEGACY (sin ADK)
    # =========================================================================
    else:
        _log("INFO", "ORQUESTADOR", "ADK no disponible — ejecutando en modo directo")
        yield from _run_legacy_pipeline(df, sales_col, cat_col,
                                        pipeline_result, logs, _log, _event)


# ---------------------------------------------------------------------------
# Rama de ejecución directa (legacy / fallback)
# ---------------------------------------------------------------------------

def _run_legacy_pipeline(
    df: pd.DataFrame,
    sales_col,
    cat_col,
    pipeline_result: dict,
    logs: list,
    _log,
    _event,
) -> Generator[PipelineEvent, None, None]:
    """Ejecución directa sin ADK (modo legacy / fallback)."""

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
    df_clean = df
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
        df_clean = df  # fallback

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
            yield _event("Prediccion", "done",
                         f"Predicción completada — mejor modelo: {best}",
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
    _log("INFO", "ORQUESTADOR", "Pipeline (legacy) finalizado")
    yield _event("Pipeline", "final", "Pipeline completado", data=pipeline_result)
