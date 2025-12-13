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
    page_title="Churn Prediction | EDA Dashboard",
    layout="wide"
)

st.title("Exploratory Data Analysis")
st.caption(
    "Interactive exploration of the Telco Customer Churn dataset. "
    "Analyze distributions and relationships with customer churn."
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
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    churn_rate = (df["Churn"] == "Yes").mean()
    st.metric("Churn Rate", f"{churn_rate * 100:.2f}%")

st.divider()

# =========================================================
# VARIABLE SELECTION
# =========================================================
st.subheader("Variable Analysis")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Variables no informativas
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

analysis_type = st.radio(
    "Select variable type",
    ["Numerical", "Categorical"],
    horizontal=True
)

if analysis_type == "Numerical":
    selected_col = st.selectbox(
        "Select numerical variable",
        numeric_cols
    )
else:
    selected_col = st.selectbox(
        "Select categorical variable",
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
st.subheader("Summary Statistics")

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
