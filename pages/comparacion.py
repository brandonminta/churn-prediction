import streamlit as st
import pandas as pd
import seaborn as sns
from numbers import Number

from utils.loader import load_model_results
from utils.visualization import plot_metric_comparison, plot_confusion_matrix
from utils.layout import render_sidebar


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction | Comparación",
    layout="wide",
)

render_sidebar()

st.title("Comparación de modelos")
st.caption(
    "Resumen cuantitativo de los modelos entrenados bajo los dos conjuntos de "
    "features (full y reduced) utilizando la misma partición de validación."
)

st.divider()


# =========================================================
# LOAD RESULTS (CACHED)
# =========================================================
@st.cache_data
def get_results():
    return load_model_results()


results = get_results()


# =========================================================
# PREPARE RESULTS TABLE
# =========================================================
records = []

for model_key, content in results.items():
    model_name, feature_set = model_key.split("_", 1)

    metrics = content["metrics"]

    row = {
        "Model": model_name.capitalize(),
        "Feature Set": feature_set.capitalize(),
    }

    for metric_name, value in metrics.items():
        if isinstance(value, Number):
            row[metric_name] = float(value)

    records.append(row)

df_results = pd.DataFrame(records)


# =========================================================
# METRIC SELECTION
# =========================================================
st.subheader("Métricas disponibles")

numeric_metrics = [
    col for col in df_results.columns
    if col not in ["Model", "Feature Set"]
    and pd.api.types.is_numeric_dtype(df_results[col])
]

df_results[numeric_metrics] = df_results[numeric_metrics].apply(
    pd.to_numeric,
    errors="coerce",
)

if not numeric_metrics:
    st.warning("No se encontraron métricas numéricas en los resultados cargados.")
    st.stop()

metric_selected = st.selectbox(
    "Selecciona la métrica a comparar",
    numeric_metrics,
)


# =========================================================
# BARPLOT COMPARISON
# =========================================================
sns.set_theme(style="whitegrid")

left, right = st.columns([1.5, 1])
with left:
    df_plot = df_results.dropna(subset=[metric_selected])
    if df_plot.empty:
        st.info("No hay valores válidos para la métrica seleccionada.")
    else:
        fig = plot_metric_comparison(df_plot, metric_selected)
        st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("Tabla detallada")
    sorted_df = df_results.sort_values(metric_selected, ascending=False)
    st.dataframe(
        sorted_df.style.format("{:.4f}", subset=numeric_metrics),
        use_container_width=True,
    )

st.divider()


# =========================================================
# BEST MODEL HIGHLIGHT
# =========================================================
best_row = None

if not df_results[metric_selected].dropna().empty:
    best_row = df_results.loc[
        df_results[metric_selected].astype(float).idxmax()
    ]

    st.subheader("Mejor modelo (por la métrica seleccionada)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Modelo", best_row["Model"])

    with col2:
        st.metric("Set de features", best_row["Feature Set"])

    with col3:
        st.metric(
            metric_selected,
            f"{best_row[metric_selected]:.4f}",
        )
else:
    st.info("No hay métricas numéricas disponibles para comparar.")


# =========================================================
# PARAMETER INSPECTION (OPTIONAL)
# =========================================================
if best_row is not None:
    with st.expander("Parámetros del mejor modelo"):
        selected_key = f"{best_row['Model'].lower()}_{best_row['Feature Set'].lower()}"
        params = results[selected_key]["params"]
        st.json(params)


# =========================================================
# CONFUSION MATRIX (MODEL LEVEL)
# =========================================================
st.subheader("Matriz de confusión por modelo")

# Labels legibles para el usuario
model_labels = {
    key: key.replace("_", " ").title()
    for key in results.keys()
}

selected_label = st.selectbox(
    "Selecciona un modelo para análisis detallado",
    list(model_labels.values())
)

# Recuperar la key real
selected_model_key = next(
    k for k, v in model_labels.items() if v == selected_label
)

metrics_selected = results[selected_model_key]["metrics"]
cm = metrics_selected["confusion_matrix"]

fig_cm = plot_confusion_matrix(
    cm,
    labels=("Churn", "No Churn"),
    normalize=True
)

st.pyplot(fig_cm, use_container_width=False)
