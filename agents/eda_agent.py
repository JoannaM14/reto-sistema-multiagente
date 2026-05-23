"""
Agente 1 – Análisis Exploratorio de Datos (EDA)
================================================
Calcula estadísticas descriptivas y genera visualizaciones:
  - Histograma de distribución de ventas
  - Matriz de correlación (limitada a ≤15 columnas numéricas)

Acepta un callback `emit(msg: str)` para reportar hitos al orquestador.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Tool principal del agente
# ---------------------------------------------------------------------------

def run_eda(
    df: pd.DataFrame,
    sales_col: str = None,
    cat_col: str = None,
    emit: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Ejecuta el análisis exploratorio completo sobre el DataFrame recibido.

    Parámetros
    ----------
    df        : pd.DataFrame  – Dataset crudo de retail
    sales_col : str | None    – Columna objetivo (None = auto-detectar)
    cat_col   : str | None    – Columna de categorías (None = auto-detectar)
    emit      : callable      – Callback fn(msg) para reportar progreso

    Retorna
    -------
    dict con claves:
        stats          – dict de estadísticas descriptivas
        null_summary   – dict de nulos por columna
        fig_hist       – figura Plotly del histograma de ventas
        fig_corr       – figura Plotly de la matriz de correlación (o None)
        corr_skipped   – bool: True si se omitió la correlación
        corr_skip_reason – str: motivo de omisión (si aplica)
        shape          – (filas, columnas)
        dtypes         – dict de tipos de datos
        sales_col      – columna objetivo usada
        cat_col        – columna de categoría usada
        category_summary – lista de dicts con resumen por categoría
        status         – mensaje de estado final
    """

    def _emit(msg: str):
        if emit:
            emit(msg)

    result = {}

    # ── 1. Forma del dataset ──────────────────────────────────────────────
    result["shape"] = df.shape
    result["dtypes"] = df.dtypes.astype(str).to_dict()
    _emit(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # ── 2. Estadísticas descriptivas ──────────────────────────────────────
    stats = {}
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols:
        stats[col] = {
            "media":   round(float(df[col].mean()), 4),
            "mediana": round(float(df[col].median()), 4),
            "std":     round(float(df[col].std()), 4),
            "min":     round(float(df[col].min()), 4),
            "max":     round(float(df[col].max()), 4),
        }
    result["stats"] = stats
    _emit(f"Estadísticas calculadas para {len(num_cols)} columnas numéricas")

    # ── 3. Resumen de valores nulos ───────────────────────────────────────
    nulls = df.isnull().sum()
    result["null_summary"] = {
        col: {"nulos": int(n), "porcentaje": round(n / len(df) * 100, 2)}
        for col, n in nulls.items()
    }
    null_total = int(nulls.sum())
    _emit(f"Nulos detectados: {null_total} valores en {int((nulls > 0).sum())} columnas")

    # ── 4. Detectar columna de ventas ─────────────────────────────────────
    if not sales_col or sales_col not in df.columns:
        sales_col = _detect_sales_column(df)
    if not sales_col:
        num_fallback = num_cols
        sales_col = num_fallback[0] if num_fallback else None
    result["sales_col"] = sales_col
    _emit(f"Columna objetivo: '{sales_col}'")

    # ── 5. Histograma de ventas ───────────────────────────────────────────
    if sales_col:
        fig_hist = px.histogram(
            df,
            x=sales_col,
            nbins=40,
            title=f"Distribución de {sales_col}",
            labels={sales_col: sales_col, "count": "Frecuencia"},
            color_discrete_sequence=["#2DD4BF"],
            template="plotly_dark",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(13,17,32,0.97)",
            font=dict(family="Inter", color="#CBD5E1"),
            title_font_size=16,
            title_font_color="#94A3B8",
            bargap=0.06,
        )
        fig_hist.update_traces(marker_line_width=0)
        fig_hist.add_vline(
            x=df[sales_col].mean(),
            line_dash="dash",
            line_color="#38BDF8",
            annotation_text=f"Media: {df[sales_col].mean():.1f}",
            annotation_position="top right",
            annotation_font_color="#38BDF8",
        )
    else:
        col_fallback = num_cols[0] if num_cols else None
        if col_fallback:
            fig_hist = px.histogram(
                df, x=col_fallback,
                title=f"Distribución de {col_fallback}",
                color_discrete_sequence=["#2DD4BF"],
                template="plotly_dark",
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(13,17,32,0.97)",
                font=dict(family="Inter", color="#CBD5E1"),
            )
        else:
            fig_hist = go.Figure()

    result["fig_hist"] = fig_hist
    _emit("Histograma de distribución generado")

    # ── 6. Matriz de correlación (máx. 15 columnas) ───────────────────────
    CORR_MAX_COLS = 15
    if len(num_cols) >= 2 and len(num_cols) <= CORR_MAX_COLS:
        corr_matrix = df[num_cols].corr().round(3)
        z = corr_matrix.values.tolist()
        x = corr_matrix.columns.tolist()
        y = corr_matrix.index.tolist()

        fig_corr = ff.create_annotated_heatmap(
            z=z, x=x, y=y,
            annotation_text=[[f"{v:.2f}" for v in row] for row in z],
            colorscale=[[0, "#0F2744"], [0.5, "#1D6FA4"], [1, "#2DD4BF"]],
            showscale=True,
        )
        fig_corr.update_layout(
            title="Matriz de Correlación",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(13,17,32,0.97)",
            font=dict(family="Inter", color="#CBD5E1", size=11),
            title_font_size=16,
            title_font_color="#94A3B8",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        result["fig_corr"] = fig_corr
        result["corr_skipped"] = False
        result["corr_skip_reason"] = None
        _emit(f"Matriz de correlación generada ({len(num_cols)}×{len(num_cols)})")

    elif len(num_cols) > CORR_MAX_COLS:
        result["fig_corr"] = None
        result["corr_skipped"] = True
        result["corr_skip_reason"] = (
            f"Matriz omitida: {len(num_cols)} variables numéricas "
            f"(límite: {CORR_MAX_COLS}). Usa el expander de estadísticas."
        )
        _emit(f"Matriz de correlación omitida ({len(num_cols)} variables > límite {CORR_MAX_COLS})")
    else:
        result["fig_corr"] = None
        result["corr_skipped"] = False
        result["corr_skip_reason"] = None

    # ── 7. Resumen por categorías ─────────────────────────────────────────
    if not cat_col or cat_col not in df.columns:
        cat_col = _detect_category_column(df)

    if cat_col and sales_col:
        cat_summary = (
            df.groupby(cat_col)[sales_col]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "Promedio", "sum": "Total", "count": "N"})
            .round(2)
            .reset_index()
        )
        result["category_summary"] = cat_summary.to_dict(orient="records")
        result["cat_col"] = cat_col
        _emit(f"Resumen por categoría '{cat_col}': {len(cat_summary)} grupos")
    else:
        result["category_summary"] = []
        result["cat_col"] = None

    result["status"] = "EDA completado"
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_sales_column(df: pd.DataFrame) -> str | None:
    candidates = ["Sales", "sales", "Ventas", "ventas", "Revenue",
                  "revenue", "Total", "total", "Amount", "amount",
                  "Ingresos", "ingresos"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if any(k in col.lower() for k in ["sale", "venta", "revenue", "total"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def _detect_category_column(df: pd.DataFrame) -> str | None:
    candidates = ["Category", "category", "Categoria", "categoria",
                  "ProductCategory", "product_category", "Segment",
                  "segment", "Region", "region"]
    for c in candidates:
        if c in df.columns:
            return c
    return None
