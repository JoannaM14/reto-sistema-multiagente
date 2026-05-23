"""
Agente 2 – Limpieza de Datos
=============================
Aplica:
  - Eliminación de duplicados
  - Imputación de nulos (mediana numérica, moda categórica)
  - Codificación de variables categóricas (Label Encoding)
  - Extracción de features de fecha

Acepta un callback `emit(msg: str)` para reportar hitos al orquestador.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Tool principal del agente
# ---------------------------------------------------------------------------

def run_cleaning(
    df: pd.DataFrame,
    emit: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Limpia el DataFrame recibido y devuelve el resultado con un reporte.

    Parámetros
    ----------
    df   : pd.DataFrame  – Dataset crudo de retail
    emit : callable      – Callback fn(msg) para reportar progreso

    Retorna
    -------
    dict con claves:
        df_clean        – DataFrame limpio
        report          – dict describiendo todos los cambios realizados
        encoders        – dict de LabelEncoders por columna (para inversión)
        original_shape  – shape antes de la limpieza
        clean_shape     – shape después de la limpieza
        status          – mensaje de estado
    """

    def _emit(msg: str):
        if emit:
            emit(msg)

    report = {}
    df_clean = df.copy()

    # ── 1. Estado inicial ─────────────────────────────────────────────────
    original_shape = df_clean.shape
    report["filas_originales"] = original_shape[0]
    report["columnas"] = original_shape[1]
    _emit(f"Inicio: {original_shape[0]:,} filas × {original_shape[1]} columnas")

    # ── 2. Eliminar duplicados ────────────────────────────────────────────
    n_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    n_dupes = n_before - len(df_clean)
    report["duplicados_eliminados"] = n_dupes
    _emit(f"Duplicados eliminados: {n_dupes}")

    # ── 3. Imputación numérica (mediana) ──────────────────────────────────
    imputaciones = {}
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()

    n_num_imputadas = 0
    for col in num_cols:
        n_null = df_clean[col].isnull().sum()
        if n_null > 0:
            fill_val = df_clean[col].median()
            df_clean[col].fillna(fill_val, inplace=True)
            imputaciones[col] = {
                "tipo": "mediana",
                "valor": round(float(fill_val), 4),
                "nulos_imputados": int(n_null),
            }
            n_num_imputadas += int(n_null)

    _emit(f"Imputación numérica (mediana): {n_num_imputadas} valores en {len(imputaciones)} columnas")

    # ── 4. Imputación categórica (moda) ───────────────────────────────────
    n_cat_imputadas = 0
    cat_imp_count = 0
    for col in cat_cols:
        n_null = df_clean[col].isnull().sum()
        if n_null > 0:
            fill_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Desconocido"
            df_clean[col].fillna(fill_val, inplace=True)
            imputaciones[col] = {
                "tipo": "moda",
                "valor": str(fill_val),
                "nulos_imputados": int(n_null),
            }
            n_cat_imputadas += int(n_null)
            cat_imp_count += 1

    _emit(f"Imputación categórica (moda): {n_cat_imputadas} valores en {cat_imp_count} columnas")

    report["imputaciones"] = imputaciones
    report["total_nulos_imputados"] = sum(v["nulos_imputados"] for v in imputaciones.values())

    # ── 5. Label Encoding ─────────────────────────────────────────────────
    skip_keywords = ["id", "date", "fecha", "order", "invoice"]
    cols_to_encode = [
        col for col in cat_cols
        if not any(k in col.lower() for k in skip_keywords)
    ]

    encoders = {}
    encoded_cols = {}

    for col in cols_to_encode:
        le = LabelEncoder()
        original_vals = df_clean[col].astype(str).unique().tolist()
        df_clean[col + "_encoded"] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
        encoded_cols[col] = {
            "nueva_columna": col + "_encoded",
            "categorias": original_vals,
            "n_categorias": len(original_vals),
        }

    report["columnas_codificadas"] = encoded_cols
    _emit(f"Label Encoding aplicado: {len(encoded_cols)} columnas codificadas")

    # ── 6. Extracción de features de fecha ────────────────────────────────
    date_cols = [col for col in df_clean.columns
                 if any(k in col.lower() for k in ["date", "fecha"])]
    date_features_extracted = []
    for col in date_cols:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
            df_clean[col + "_year"]    = df_clean[col].dt.year
            df_clean[col + "_month"]   = df_clean[col].dt.month
            df_clean[col + "_quarter"] = df_clean[col].dt.quarter
            report[f"{col}_extraido"] = ["year", "month", "quarter"]
            date_features_extracted.append(col)
        except Exception:
            pass

    if date_features_extracted:
        _emit(f"Features de fecha extraídas de: {', '.join(date_features_extracted)}")

    # ── 7. Resumen final ──────────────────────────────────────────────────
    clean_shape = df_clean.shape
    report["filas_resultantes"] = clean_shape[0]
    report["columnas_resultantes"] = clean_shape[1]
    report["nulos_restantes"] = int(df_clean.isnull().sum().sum())
    _emit(f"Limpieza completa: {clean_shape[0]:,} filas × {clean_shape[1]} columnas — nulos restantes: {report['nulos_restantes']}")

    return {
        "df_clean": df_clean,
        "report": report,
        "encoders": encoders,
        "original_shape": original_shape,
        "clean_shape": clean_shape,
        "status": "Limpieza completada",
    }
