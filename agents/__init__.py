"""
Sistema Multiagente de Análisis de Retail
Agentes: EDA | Limpieza | Predicción | Orquestador
"""
from .events import PipelineEvent
from .eda_agent import run_eda
from .cleaning_agent import run_cleaning
from .prediction_agent import run_prediction
from .orchestrator import run_pipeline_stream

__all__ = [
    "PipelineEvent",
    "run_eda",
    "run_cleaning",
    "run_prediction",
    "run_pipeline_stream",
]
