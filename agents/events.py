"""
agents/events.py
================
Dataclass PipelineEvent: unidad mínima de comunicación entre el
orquestador y la interfaz Streamlit.

Cada evento tiene:
  agent  – quién lo emite   ("EDA" | "Limpieza" | "Prediccion" | "Pipeline")
  type   – qué significa    ("start" | "progress" | "done" | "error" | "final")
  msg    – texto legible para el usuario
  data   – payload opcional (resultado parcial del agente)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field


@dataclass
class PipelineEvent:
    agent: str
    type: str   # "start" | "progress" | "done" | "error" | "final"
    msg: str
    data: dict = field(default_factory=dict)

    def log_line(self) -> str:
        """Devuelve una línea de log formateada con timestamp y nivel."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        level_map = {
            "start":    "INFO ",
            "progress": "INFO ",
            "done":     "OK   ",
            "error":    "ERROR",
            "final":    "INFO ",
        }
        level = level_map.get(self.type, "INFO ")
        return f"[{ts}] [{level}] [{self.agent}] {self.msg}"
