import streamlit as st
import pandas as pd
import seaborn as sns

from utils.loader import load_model_results
from utils.visualization import plot_metric_comparison


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction | Comparación",
    layout="wide",
)

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
        row[metric_name] = value

    records.append(row)

df_results = pd.DataFrame(records)


# =========================================================
# METRIC SELECTION
# =========================================================
st.subheader("Métricas disponibles")

available_metrics = [
    col for col in df_results.columns
    if col not in ["Model", "Feature Set"]
]

metric_selected = st.selectbox(
    "Selecciona la métrica a comparar",
    available_metrics,
)


# =========================================================
# BARPLOT COMPARISON
# =========================================================
sns.set_theme(style="whitegrid")

fig = plot_metric_comparison(df_results, metric_selected)
st.pyplot(fig)

st.divider()


# =========================================================
# FULL VS REDUCED COMPARISON TABLE
# =========================================================
st.subheader("Tabla detallada")

st.dataframe(
    df_results
        .sort_values(metric_selected, ascending=False)
        .style
        .format("{:.4f}", subset=available_metrics),
    use_container_width=True,
)


# =========================================================
# BEST MODEL HIGHLIGHT
# =========================================================
best_row = df_results.loc[
    df_results[metric_selected].idxmax()
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


# =========================================================
# PARAMETER INSPECTION (OPTIONAL)
# =========================================================
with st.expander("Parámetros del mejor modelo"):
    selected_key = f"{best_row['Model'].lower()}_{best_row['Feature Set'].lower()}"
    params = results[selected_key]["params"]
    st.json(params)

