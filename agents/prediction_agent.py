"""
Agente 3 – Modelo Predictivo
==============================
Entrena y compara dos modelos:
  - Regresión Lineal (sklearn LinearRegression)
  - Árbol de Decisión (sklearn DecisionTreeRegressor)

Métrica: RMSE sobre conjunto de prueba (80/20 split)
Genera predicciones interpretables por categoría de producto.

Acepta un callback `emit(msg: str)` para reportar hitos al orquestador.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Tool principal del agente
# ---------------------------------------------------------------------------

def run_prediction(
    df_clean: pd.DataFrame,
    sales_col: str = None,
    cat_col: str = None,
    emit: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Entrena modelos predictivos y compara su desempeño.

    Parámetros
    ----------
    df_clean  : pd.DataFrame  – Dataset ya limpio y codificado
    sales_col : str | None    – Nombre de la columna objetivo
    cat_col   : str | None    – Columna de categoría (para interpretación)
    emit      : callable      – Callback fn(msg) para reportar progreso

    Retorna
    -------
    dict con claves:
        metrics            – RMSE y R² de ambos modelos
        best_model_name    – nombre del modelo ganador
        best_rmse          – RMSE del modelo ganador
        best_r2            – R² del modelo ganador
        predictions_df     – DataFrame con predicciones por categoría
        fig_comparison     – figura Plotly comparando RMSE
        fig_predictions    – figura Plotly de predicciones vs reales
        feature_importance – dict importancia de features (árbol)
        feature_cols       – lista de features usadas
        sales_col_used     – columna objetivo usada
        status             – mensaje de estado
    """

    def _emit(msg: str):
        if emit:
            emit(msg)

    result = {}

    # ── 1. Resolver columna objetivo ─────────────────────────────────────
    if not sales_col or sales_col not in df_clean.columns:
        sales_col = _detect_sales_column(df_clean)
    if not sales_col:
        num_cols_fb = df_clean.select_dtypes(include=np.number).columns.tolist()
        excl = ["id", "date", "fecha", "order", "invoice", "year", "month", "quarter"]
        num_cols_fb = [c for c in num_cols_fb if not any(k in c.lower() for k in excl)]
        if num_cols_fb:
            sales_col = max(num_cols_fb, key=lambda c: df_clean[c].var())
        else:
            return {"status": "Error: no hay columnas numéricas en el dataset", "error": True}
    result["sales_col_used"] = sales_col
    _emit(f"Variable objetivo: '{sales_col}'")

    # ── 2. Seleccionar features ───────────────────────────────────────────
    exclude_keywords = ["id", "date", "fecha", "order", "invoice"]
    feature_cols = [
        col for col in df_clean.select_dtypes(include=np.number).columns
        if col != sales_col
        and not any(k in col.lower() for k in exclude_keywords)
    ]

    if len(feature_cols) == 0:
        return {"status": "No hay features numéricas suficientes", "error": True}

    result["feature_cols"] = feature_cols
    _emit(f"Features seleccionadas: {len(feature_cols)} columnas numéricas")

    X = df_clean[feature_cols].fillna(0)
    y = df_clean[sales_col].fillna(df_clean[sales_col].median())

    # ── 3. Split 80/20 ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    _emit(f"Split 80/20: train={len(X_train):,} muestras — test={len(X_test):,} muestras")

    # ── 4. Escalar + entrenar Regresión Lineal ────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    _emit("Regresión Lineal entrenada")

    # ── 5. Entrenar Árbol de Decisión ─────────────────────────────────────
    dt = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    _emit("Árbol de Decisión entrenado (max_depth=6)")

    # ── 6. Evaluación ─────────────────────────────────────────────────────
    y_pred_lr = lr.predict(X_test_sc)
    y_pred_dt = dt.predict(X_test)

    rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    rmse_dt = float(np.sqrt(mean_squared_error(y_test, y_pred_dt)))
    r2_lr   = float(lr.score(X_test_sc, y_test))
    r2_dt   = float(dt.score(X_test, y_test))

    metrics = {
        "Regresión Lineal":  {"RMSE": round(rmse_lr, 4), "R²": round(r2_lr, 4)},
        "Árbol de Decisión": {"RMSE": round(rmse_dt, 4), "R²": round(r2_dt, 4)},
    }
    result["metrics"] = metrics

    best_model_name = "Regresión Lineal" if rmse_lr <= rmse_dt else "Árbol de Decisión"
    result["best_model_name"] = best_model_name
    result["best_rmse"] = min(rmse_lr, rmse_dt)
    result["best_r2"]   = r2_lr if best_model_name == "Regresión Lineal" else r2_dt
    _emit(
        f"Evaluación completada — LR: RMSE={rmse_lr:.4f} R²={r2_lr:.4f} | "
        f"DT: RMSE={rmse_dt:.4f} R²={r2_dt:.4f}"
    )
    _emit(f"Mejor modelo: {best_model_name} (RMSE={result['best_rmse']:.4f})")

    # ── 7. Predicciones sobre el conjunto completo ────────────────────────
    if best_model_name == "Regresión Lineal":
        X_all_sc = scaler.transform(X.fillna(0))
        preds_all = lr.predict(X_all_sc)
    else:
        preds_all = dt.predict(X.fillna(0))

    df_pred = df_clean.copy()
    df_pred["Ventas_Predichas"] = np.round(preds_all, 2)

    # ── 8. Agrupar predicciones por categoría ─────────────────────────────
    if cat_col and cat_col in df_pred.columns:
        group_col = cat_col
    else:
        group_col = _find_category_col(df_pred)

    if group_col:
        predictions_df = (
            df_pred.groupby(group_col)
            .agg(
                Ventas_Reales=(sales_col, "mean"),
                Ventas_Predichas=("Ventas_Predichas", "mean"),
                N_Registros=(sales_col, "count"),
            )
            .round(2)
            .reset_index()
        )
        predictions_df.rename(columns={group_col: "Categoría"}, inplace=True)
    else:
        predictions_df = df_pred[[sales_col, "Ventas_Predichas"]].head(20).copy()
        predictions_df.rename(columns={sales_col: "Ventas_Reales"}, inplace=True)

    result["predictions_df"] = predictions_df
    _emit(f"Predicciones generadas: {len(predictions_df)} grupos")

    # ── 9. Importancia de features (árbol) ───────────────────────────────
    feat_importance = dict(zip(feature_cols, dt.feature_importances_.round(4)))
    result["feature_importance"] = dict(
        sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # ── 10. Figura comparación RMSE ───────────────────────────────────────
    models    = list(metrics.keys())
    rmse_vals = [metrics[m]["RMSE"] for m in models]
    colors    = ["#0EA5E9", "#2DD4BF"]

    fig_comparison = go.Figure(data=[
        go.Bar(
            name="RMSE (menor = mejor)",
            x=models,
            y=rmse_vals,
            marker_color=colors,
            text=[f"{v:.4f}" for v in rmse_vals],
            textposition="auto",
        )
    ])
    winner_idx = 0 if rmse_lr <= rmse_dt else 1
    fig_comparison.update_layout(
        title="Comparación de Modelos – RMSE",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(13,17,32,0.97)",
        font=dict(family="Inter", color="#CBD5E1"),
        title_font_size=16,
        title_font_color="#94A3B8",
        yaxis_title="RMSE",
        showlegend=False,
    )
    fig_comparison.add_annotation(
        x=models[winner_idx],
        y=rmse_vals[winner_idx],
        text="Mejor",
        showarrow=True,
        arrowhead=2,
        font=dict(color="#2DD4BF", size=13),
        arrowcolor="#2DD4BF",
    )
    result["fig_comparison"] = fig_comparison

    # ── 11. Figura predicciones vs reales por categoría ───────────────────
    if "Categoría" in predictions_df.columns:
        fig_predictions = go.Figure()
        fig_predictions.add_trace(go.Bar(
            name="Ventas Reales",
            x=predictions_df["Categoría"],
            y=predictions_df["Ventas_Reales"],
            marker_color="#0EA5E9",
        ))
        fig_predictions.add_trace(go.Bar(
            name="Ventas Predichas",
            x=predictions_df["Categoría"],
            y=predictions_df["Ventas_Predichas"],
            marker_color="#2DD4BF",
        ))
        fig_predictions.update_layout(
            title="Predicciones vs Reales por Categoría",
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(13,17,32,0.97)",
            font=dict(family="Inter", color="#CBD5E1"),
            title_font_size=16,
            title_font_color="#94A3B8",
            yaxis_title="Ventas Promedio",
        )
    else:
        fig_predictions = go.Figure()

    result["fig_predictions"] = fig_predictions
    result["status"] = f"Predicción completada — Mejor modelo: {best_model_name}"
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_sales_column(df: pd.DataFrame) -> str | None:
    candidates = ["Sales", "sales", "Ventas", "ventas", "Revenue",
                  "revenue", "Total", "total", "Amount", "amount"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if any(k in col.lower() for k in ["sale", "venta", "revenue", "total"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def _find_category_col(df: pd.DataFrame) -> str | None:
    candidates = ["Category", "category", "Categoria", "categoria",
                  "ProductCategory", "Segment", "segment", "Region", "region",
                  "CustomerSegment"]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() < 20:
            return col
    return None
