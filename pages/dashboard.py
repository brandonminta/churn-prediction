import streamlit as st
import pandas as pd

from utils.loader import load_dataset
from utils.visualization import (
    set_style,
    plot_numeric_by_churn,
    plot_categorical_by_churn
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction | EDA",
    layout="wide"
)

st.title("Análisis exploratorio")
st.caption(
    "Exploración interactiva del dataset Telco Customer Churn para entender la "
    "distribución de las variables y su relación con el churn."
)

st.divider()

# =========================================================
# LOAD DATA (CACHED)
# =========================================================
@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

# =========================================================
# DATA OVERVIEW
# =========================================================
st.subheader("Visión general del dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Filas", df.shape[0])

with col2:
    st.metric("Columnas", df.shape[1])

with col3:
    churn_rate = (df["Churn"] == "Yes").mean()
    st.metric("Tasa de churn", f"{churn_rate * 100:.2f}%")

st.divider()

# =========================================================
# VARIABLE SELECTION
# =========================================================
st.subheader("Análisis por variable")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Variables no informativas
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

analysis_type = st.radio(
    "Tipo de variable",
    ["Numerical", "Categorical"],
    horizontal=True
)

if analysis_type == "Numerical":
    selected_col = st.selectbox(
        "Variable numérica",
        numeric_cols
    )
else:
    selected_col = st.selectbox(
        "Variable categórica",
        categorical_cols
    )

# =========================================================
# PLOTTING
# =========================================================
set_style()

if analysis_type == "Numerical":
    fig = plot_numeric_by_churn(df, selected_col)
else:
    fig = plot_categorical_by_churn(df, selected_col)

st.pyplot(fig)

st.divider()

# =========================================================
# SUMMARY STATISTICS
# =========================================================
st.subheader("Estadísticos resumidos")

if analysis_type == "Numerical":
    summary = (
        df.groupby("Churn")[selected_col]
        .describe()
        .round(2)
    )
    st.dataframe(summary, use_container_width=True)

else:
    summary = (
        df.groupby([selected_col, "Churn"])
        .size()
        .unstack(fill_value=0)
    )
    st.dataframe(summary, use_container_width=True)
