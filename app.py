"""
RetailAI – Sistema Multiagente de Análisis de Retail
Interfaz Streamlit con: encoding automático, selector de columnas y st.status
"""

import io
import hashlib
import streamlit as st
import pandas as pd
from agents.orchestrator import run_pipeline_stream

st.set_page_config(
    page_title="RetailAI – Sistema Multiagente",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0b0f1e; }
.agent-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #2dd4bf);
    color: #0b0f1e; border-radius: 20px;
    padding: 0.3rem 0.9rem; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.8rem;
    letter-spacing: 0.02em;
}
.log-box {
    background: #0d1224; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 1rem; font-family: 'Courier New', monospace; font-size: 0.78rem;
    color: #a8d8ea; max-height: 600px; overflow-y: auto; white-space: pre-wrap;
}
.log-error { color: #f87171; }
.log-warn  { color: #fcd34d; }
.log-info  { color: #60a5fa; }
.log-ok    { color: #34d399; }
.winner-badge {
    background: linear-gradient(135deg, #0ea5e9, #2dd4bf);
    color: #0b0f1e; border-radius: 8px; padding: 0.5rem 1.2rem;
    font-weight: 700; font-size: 1.1rem; display: inline-block;
}
[data-testid="stSidebar"] { background: #0d1224 !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d1224; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"]      { color: #64748b; border-radius: 8px; }
.stTabs [aria-selected="true"]    { background: linear-gradient(135deg,#0ea5e9,#2dd4bf) !important; color: #0b0f1e !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_dataframe(file) -> tuple[pd.DataFrame | None, str]:
    """Intenta leer CSV/XLSX probando múltiples encodings. Retorna (df, error_msg)."""
    name = file.name.lower()
    raw = file.read()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            return pd.read_excel(io.BytesIO(raw)), ""
        except Exception as e:
            return None, str(e)

    # CSV: probar encodings comunes
    for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16", "utf-8-sig"]:
        try:
            for sep in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
                    if df.shape[1] > 1:
                        return df, ""
                except Exception:
                    continue
        except Exception:
            continue
    return None, "No se pudo leer el archivo con ningún encoding conocido (utf-8, latin-1, cp1252…)"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache_key(df: pd.DataFrame, sales_col: str, cat_col: str) -> str:
    buf = io.StringIO()
    df.to_csv(buf)
    return hashlib.md5(f"{buf.getvalue()}{sales_col}{cat_col}".encode()).hexdigest()


def color_log_line(line: str) -> str:
    if "[ERROR]" in line: return f'<span class="log-error">{line}</span>'
    if "[WARN"  in line:  return f'<span class="log-warn">{line}</span>'
    if "[OK"    in line:  return f'<span class="log-ok">{line}</span>'
    return f'<span class="log-info">{line}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RetailAI")
    st.markdown("**Sistema Multiagente de Análisis**")
    st.divider()

    st.markdown("### Cargar Dataset")
    uploaded = st.file_uploader(
        "CSV o XLSX", type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    if st.button("Usar Dataset de Ejemplo", use_container_width=True):
        st.session_state["use_sample"] = True
    elif uploaded:
        st.session_state["use_sample"] = False

    st.divider()
    st.markdown("### Agentes")
    st.markdown("1. **EDA**\n\n2. **Limpieza**\n\n3. **Predicción**\n\n→ **Orquestador**")
    st.divider()
    st.caption("Google ADK · Streamlit · scikit-learn")


# ── Carga ─────────────────────────────────────────────────────────────────────
df = None
load_error = ""

if uploaded:
    df, load_error = read_dataframe(uploaded)
elif st.session_state.get("use_sample"):
    try:
        df = pd.read_csv("data/sample_retail.csv")
    except Exception as e:
        load_error = str(e)

# ── Pantalla de bienvenida ────────────────────────────────────────────────────
if df is None:
    st.markdown("## RetailAI – Sistema Multiagente")
    if load_error:
        st.error(f"Error al leer el archivo: {load_error}")
    else:
        st.info("Sube un dataset de retail en el panel izquierdo o usa el dataset de ejemplo.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Agente 1", "EDA",        "Estadísticas + Visualizaciones")
    c2.metric("Agente 2", "Limpieza",   "Imputación + Encoding")
    c3.metric("Agente 3", "Predicción", "LinearReg + DecisionTree")
    st.stop()


# ── Selector de columnas (después de cargar) ──────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### Configuración del Pipeline")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()

    sales_col = st.selectbox(
        "Columna objetivo (ventas)",
        ["(auto-detectar)"] + num_cols,
        help="Columna numérica que se usará como variable a predecir",
    )
    cat_col = st.selectbox(
        "Columna de categorías",
        ["(ninguna)"] + cat_cols_all,
        help="Columna categórica para agrupar las predicciones",
    )


# ── Ejecutar pipeline (con caché en session_state) ────────────────────────────
sc_param = sales_col if sales_col != "(auto-detectar)" else None
cc_param = cat_col   if cat_col   != "(ninguna)"       else None
cache_key = _make_cache_key(df, sales_col, cat_col)

if st.session_state.get("cache_key") != cache_key:
    # Limpiar resultado anterior
    st.session_state.pop("pipeline_result", None)

if "pipeline_result" not in st.session_state:
    # ── Ejecutar y mostrar progreso en tiempo real ────────────────────────
    agent_icons = {
        "EDA":       "Agente 1 – EDA",
        "Limpieza":  "Agente 2 – Limpieza",
        "Prediccion":"Agente 3 – Predicción",
        "Pipeline":  "Pipeline",
    }
    with st.status("Iniciando pipeline multiagente…", expanded=True) as status:
        for event in run_pipeline_stream(df, sales_col=sc_param, cat_col=cc_param):
            if event.type == "start":
                label = agent_icons.get(event.agent, event.agent)
                status.update(label=f"{label}…")
                st.write(f"**{label}**")
            elif event.type == "progress":
                st.write(f"&nbsp;&nbsp;&nbsp;↳ {event.msg}")
            elif event.type == "done":
                st.write(f"✓ {event.msg}")
            elif event.type == "error":
                st.error(event.msg)
            elif event.type == "final":
                st.session_state["pipeline_result"] = event.data
                st.session_state["cache_key"] = cache_key
                status.update(label="Pipeline completado", state="complete", expanded=False)

result = st.session_state.get("pipeline_result", {})
logs = result.get("logs", [])


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## RetailAI – Resultados del Pipeline")

pred_res = result.get("prediction", {})
best_model = pred_res.get("best_model_name", "–") if not pred_res.get("error") else "Error"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas",        df.shape[0])
c2.metric("Columnas",     df.shape[1])
c3.metric("Col. Ventas",  result.get("sales_col_used") or "auto")
c4.metric("Mejor Modelo", best_model[:18] if best_model else "–")
st.divider()


# ── Pestañas ──────────────────────────────────────────────────────────────────
tab_eda, tab_clean, tab_pred, tab_log, tab_arch = st.tabs([
    "Agente 1 – EDA",
    "Agente 2 – Limpieza",
    "Agente 3 – Predicción",
    "Log del Pipeline",
    "Arquitectura",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 – EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_eda:
    eda = result.get("eda", {})
    if "error" in eda:
        st.error(f"Error en EDA: {eda['error']}")
    else:
        st.markdown('<span class="agent-badge">Agente 1 – EDA</span>', unsafe_allow_html=True)
        stats = eda.get("stats", {})
        sc = eda.get("sales_col", "")

        if sc and sc in stats:
            s = stats[sc]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Media",   f"{s['media']:,.2f}")
            m2.metric("Mediana", f"{s['mediana']:,.2f}")
            m3.metric("Máximo",  f"{s['max']:,.2f}")
            m4.metric("Mínimo",  f"{s['min']:,.2f}")

        null_total = sum(v["nulos"] for v in eda.get("null_summary", {}).values())
        st.info(f"**Valores nulos detectados:** {null_total} | **Columna objetivo:** `{sc or 'N/A'}`")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig = eda.get("fig_hist")
            if fig: st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            if eda.get("corr_skipped"):
                st.info(eda.get("corr_skip_reason", "Matriz de correlación omitida por exceso de variables."))
            else:
                fig = eda.get("fig_corr")
                if fig: st.plotly_chart(fig, use_container_width=True)

        with st.expander("Estadísticas descriptivas"):
            if stats:
                rows = [{"Columna": c, **v} for c, v in stats.items()]
                st.dataframe(pd.DataFrame(rows).set_index("Columna"), use_container_width=True)

        cat_sum = eda.get("category_summary", [])
        if cat_sum:
            with st.expander(f"Resumen por {eda.get('cat_col', 'categoría')}"):
                st.dataframe(pd.DataFrame(cat_sum), use_container_width=True)

        with st.expander("Nulos por columna"):
            null_df = pd.DataFrame([
                {"Columna": k, "Nulos": v["nulos"], "%": v["porcentaje"]}
                for k, v in eda.get("null_summary", {}).items()
            ])
            st.dataframe(null_df, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 – LIMPIEZA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_clean:
    cl = result.get("cleaning", {})
    if "error" in cl:
        st.error(f"Error en Limpieza: {cl['error']}")
    else:
        st.markdown('<span class="agent-badge">Agente 2 – Limpieza</span>', unsafe_allow_html=True)
        r = cl.get("report", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filas orig.",       r.get("filas_originales", "–"))
        m2.metric("Duplicados elim.",  r.get("duplicados_eliminados", 0))
        m3.metric("Nulos imputados",   r.get("total_nulos_imputados", 0))
        m4.metric("Filas result.",     r.get("filas_resultantes", "–"))

        imput = r.get("imputaciones", {})
        if imput:
            st.subheader("Imputaciones")
            st.dataframe(pd.DataFrame([
                {"Columna": col, "Método": v["tipo"], "Valor": v["valor"], "N imputados": v["nulos_imputados"]}
                for col, v in imput.items()
            ]), use_container_width=True)

        enc = r.get("columnas_codificadas", {})
        if enc:
            st.subheader("Encoding categórico")
            st.dataframe(pd.DataFrame([
                {"Columna": col, "Nueva col": v["nueva_columna"], "N categ.": v["n_categorias"]}
                for col, v in enc.items()
            ]), use_container_width=True)

        df_clean = cl.get("df_clean")
        if df_clean is not None:
            with st.expander("Vista previa DataFrame limpio"):
                st.dataframe(df_clean.head(20), use_container_width=True)
                st.caption(f"Shape: {df_clean.shape} | Nulos restantes: {int(df_clean.isnull().sum().sum())}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 – PREDICCIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_pred:
    pr = result.get("prediction", {})
    if pr.get("error"):
        st.error(f"Error en Predicción: {pr.get('status')}")
    elif not pr:
        st.warning("El agente de predicción no produjo resultados.")
    else:
        st.markdown('<span class="agent-badge">Agente 3 – Predicción</span>', unsafe_allow_html=True)
        best = pr.get("best_model_name", "–")
        sc_used = pr.get("sales_col_used", result.get("sales_col_used", "–"))

        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="winner-badge">{best}</div>', unsafe_allow_html=True)
        m2.metric("RMSE (mejor)", f"{pr.get('best_rmse', 0):.4f}")
        m3.metric("R²  (mejor)",  f"{pr.get('best_r2', 0):.4f}")
        st.caption(f"Variable objetivo usada: `{sc_used}`")
        st.divider()

        metrics = pr.get("metrics", {})
        if metrics:
            st.subheader("Comparación de Modelos")
            col_t, col_g = st.columns([1, 2])
            with col_t:
                mdf = pd.DataFrame(metrics).T.reset_index()
                mdf.columns = ["Modelo"] + list(mdf.columns[1:])
                mdf["Ganador"] = mdf["Modelo"].apply(lambda x: "★" if x == best else "")
                st.dataframe(mdf, use_container_width=True, hide_index=True)
            with col_g:
                fig = pr.get("fig_comparison")
                if fig: st.plotly_chart(fig, use_container_width=True)

        pdf = pr.get("predictions_df")
        if pdf is not None and not pdf.empty:
            st.subheader("Predicciones")
            col_tp, col_gp = st.columns([1, 2])
            with col_tp:
                st.dataframe(pdf, use_container_width=True, hide_index=True)
            with col_gp:
                fig = pr.get("fig_predictions")
                if fig: st.plotly_chart(fig, use_container_width=True)

        feat_imp = pr.get("feature_importance", {})
        if feat_imp:
            with st.expander("Importancia de features (Árbol)"):
                fi_df = pd.DataFrame(feat_imp.items(), columns=["Feature", "Importancia"]).sort_values("Importancia", ascending=False)
                st.dataframe(fi_df, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 – LOG DEL PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_log:
    st.markdown('<span class="agent-badge">Log General del Pipeline</span>', unsafe_allow_html=True)

    ps = result.get("pipeline_status", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Agente 1 EDA",        ps.get("eda",        "–")[:30])
    c2.metric("Agente 2 Limpieza",   ps.get("cleaning",   "–")[:30])
    c3.metric("Agente 3 Predicción", ps.get("prediction", "–")[:30])

    st.divider()
    st.markdown("### Traza completa del pipeline")

    if logs:
        log_text = "\n".join(logs)
        st.download_button(
            "Descargar log (.txt)",
            data=log_text,
            file_name="pipeline_log.txt",
            mime="text/plain",
        )

        level_filter = st.multiselect(
            "Filtrar por nivel",
            ["INFO", "OK", "WARN", "ERROR"],
            default=["INFO", "OK", "WARN", "ERROR"],
        )

        filtered = [
            l for l in logs
            if any(f"[{lvl}" in l for lvl in level_filter)
        ]

        colored = "<br>".join(color_log_line(l) for l in filtered)
        st.markdown(f'<div class="log-box">{colored}</div>', unsafe_allow_html=True)

        st.caption(f"Total de entradas: {len(logs)} | Mostrando: {len(filtered)}")

        with st.expander("Resumen de entradas por agente"):
            agents = ["ORQUESTADOR", "AGENTE-1 EDA", "AGENTE-2 LIMPIEZA", "AGENTE-3 PREDICCION"]
            summary = {
                ag: {
                    "Total": sum(1 for l in logs if f"[{ag}]" in l),
                    "Errores": sum(1 for l in logs if f"[{ag}]" in l and "[ERROR]" in l),
                    "OK": sum(1 for l in logs if f"[{ag}]" in l and "[OK" in l),
                }
                for ag in agents
            }
            st.dataframe(pd.DataFrame(summary).T, use_container_width=True)
    else:
        st.info("No hay entradas de log disponibles.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 – ARQUITECTURA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_arch:
    st.markdown('<span class="agent-badge">Arquitectura del Sistema</span>', unsafe_allow_html=True)
    st.markdown("""
```
┌─────────────────────────────────────────────────────────┐
│                  INTERFAZ  STREAMLIT                    │
│     Carga CSV/XLSX · Selector de columnas · Log         │
└──────────────────────────┬──────────────────────────────┘
                           │  DataFrame + columnas
                           ▼
┌─────────────────────────────────────────────────────────┐
│         AGENTE ORQUESTADOR  (orchestrator.py)           │
│         Google ADK · SequentialAgent pattern            │
│         Genera log timestamped de cada paso             │
└──────┬─────────────────┬──────────────────┬─────────────┘
       │                 │                  │
       ▼                 ▼                  ▼
┌───────────┐  ┌──────────────────┐  ┌──────────────────┐
│ AGENTE 1  │  │    AGENTE 2      │  │    AGENTE 3      │
│   EDA     │─▶│    Limpieza      │─▶│   Predicción     │
│ Stats     │  │ Imputación       │  │ LinearRegression │
│ Histograma│  │ Deduplicación    │  │ DecisionTree     │
│ Correlac. │  │ LabelEncoding    │  │ RMSE / R²        │
└───────────┘  └──────────────────┘  └──────────────────┘
```
""")
    st.markdown("### Stack Tecnológico")
    tech = {
        "Framework Agentes": "Google ADK (SequentialAgent)",
        "Interfaz Web":      "Streamlit",
        "Procesamiento":     "pandas · numpy",
        "Modelos ML":        "scikit-learn (LinearRegression, DecisionTreeRegressor)",
        "Visualizaciones":   "Plotly Express / Graph Objects",
        "Encodings CSV":     "utf-8, latin-1, cp1252, iso-8859-1, utf-16",
    }
    for k, v in tech.items():
        st.markdown(f"- **{k}**: {v}")
    adk_ok = result.get("adk_available", False)
    st.info(f"**google-adk instalado:** {'Sí' if adk_ok else 'No (pipeline funciona igual)'}")
