import streamlit as st
import pandas as pd
import seaborn as sns
from numbers import Number
from sklearn.metrics import confusion_matrix

from utils.loader import (
    load_model_results,
    load_dataset,
    load_feature_map,
    load_model,
)
from utils.preprocessing import build_preprocessing_pipeline
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


@st.cache_data
def get_dataset():
    return load_dataset()


@st.cache_resource
def get_feature_map():
    return load_feature_map()


@st.cache_resource
def get_pipeline(required_cols: tuple, fmap: dict):
    df_ref = get_dataset()
    feature_df = df_ref[[col for col in required_cols if col in df_ref.columns]].copy()
    pipeline = build_preprocessing_pipeline(feature_df, fmap)
    pipeline.fit(feature_df)
    return pipeline


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
# CONFUSION MATRIX
# =========================================================
st.divider()
st.subheader("Matriz de confusión")

feature_map = get_feature_map()
available_feature_sets = feature_map.get("feature_sets", {})

combo_options = [
    (row["Model"].lower(), row["Feature Set"].lower())
    for _, row in df_results.iterrows()
]

if combo_options:
    col_a, col_b = st.columns(2)
    with col_a:
        model_choice = st.selectbox(
            "Modelo",
            sorted(set([m for m, _ in combo_options])),
        )

    with col_b:
        feature_choice = st.selectbox(
            "Set de features",
            sorted(set([fs for _, fs in combo_options])),
        )

    selected_key = f"{model_choice}_{feature_choice}"
    if selected_key not in results:
        st.info("No hay resultados para la combinación seleccionada.")
    else:
        df_data = get_dataset()
        if "Churn" not in df_data.columns:
            st.warning("El dataset no contiene la columna 'Churn'.")
        else:
            target = (df_data["Churn"] == "Yes").astype(int)
            feature_cols = available_feature_sets.get(feature_choice, [])
            if not feature_cols:
                st.warning("No se encontraron columnas para el set de features seleccionado.")
            else:
                feature_df = df_data[[col for col in feature_cols if col in df_data.columns]].copy()

                pipeline = get_pipeline(tuple(feature_df.columns), feature_map)
                X_prep = pipeline.transform(feature_df)
                X_prep = X_prep.reindex(columns=feature_cols, fill_value=0)

                model = load_model(model_choice, feature_choice)
                preds = model.predict(X_prep)

                cm = confusion_matrix(target, preds, labels=[0, 1])
                cm_df = pd.DataFrame(
                    cm,
                    index=["True: No", "True: Yes"],
                    columns=["Pred: No", "Pred: Yes"],
                )

                fig_cm = plot_confusion_matrix(cm_df)
                st.pyplot(fig_cm, use_container_width=True)
else:
    st.info("No hay combinaciones de modelo y set de features disponibles.")


# =========================================================
# PARAMETER INSPECTION (OPTIONAL)
# =========================================================
if best_row is not None:
    with st.expander("Parámetros del mejor modelo"):
        selected_key = f"{best_row['Model'].lower()}_{best_row['Feature Set'].lower()}"
        params = results[selected_key]["params"]
        st.json(params)
