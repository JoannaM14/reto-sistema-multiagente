# 🏪 RetailAI – Sistema Multiagente de Análisis de Retail

Sistema multiagente construido con **Google ADK** y **Streamlit** que procesa datos de una tienda de retail a través de un pipeline de tres agentes especializados.

## 🏗️ Arquitectura

```
Streamlit (app.py)
    └── Orquestador (SequentialAgent ADK)
            ├── Agente 1 – EDA
            ├── Agente 2 – Limpieza
            └── Agente 3 – Predicción
```

## 📁 Estructura

```
ProyectoOmar/
├── agents/
│   ├── __init__.py
│   ├── eda_agent.py          # Estadísticas + histograma + correlación
│   ├── cleaning_agent.py     # Imputación + deduplicación + encoding
│   ├── prediction_agent.py   # LinearReg vs DecisionTree + RMSE
│   └── orchestrator.py       # Pipeline secuencial (ADK SequentialAgent)
├── data/
│   └── sample_retail.csv     # Dataset de ejemplo (203 filas, nulos y dupes)
├── app.py                    # Interfaz Streamlit
├── requirements.txt
└── README.md
```

## ⚡ Instalación y Ejecución

### 1. Crear entorno virtual (recomendado)
```powershell
cd ProyectoOmar
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar dependencias
```powershell
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación
```powershell
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`

## 🤖 Descripción de Agentes

| Agente | Archivo | Función |
|--------|---------|---------|
| **EDA** | `eda_agent.py` | Estadísticas descriptivas, histograma de ventas, matriz de correlación |
| **Limpieza** | `cleaning_agent.py` | Imputación (mediana/moda), eliminación de duplicados, LabelEncoding |
| **Predicción** | `prediction_agent.py` | Regresión Lineal vs Árbol de Decisión, RMSE, predicciones por categoría |
| **Orquestador** | `orchestrator.py` | Pipeline secuencial `EDA → Limpieza → Predicción` (patrón ADK SequentialAgent) |

## 📊 Uso

1. Abre la app en el navegador
2. En el panel izquierdo: **sube un CSV/XLSX** o haz click en **"Usar dataset de ejemplo"**
3. El pipeline se ejecuta automáticamente
4. Navega por las pestañas para ver los resultados de cada agente
