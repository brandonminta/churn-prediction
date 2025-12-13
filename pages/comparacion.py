# pages/comparacion.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.loader import load_model_results


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction | Model Comparison",
    layout="wide"
)

st.title("Model Comparison")
st.caption(
    "Performance comparison across multiple trained models and feature sets. "
    "All results were obtained on the same validation strategy."
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
        "Feature Set": feature_set.capitalize()
    }

    for metric_name, value in metrics.items():
        row[metric_name] = value

    records.append(row)

df_results = pd.DataFrame(records)


# =========================================================
# METRIC SELECTION
# =========================================================
st.subheader("Metric Overview")

available_metrics = [
    col for col in df_results.columns
    if col not in ["Model", "Feature Set"]
]

metric_selected = st.selectbox(
    "Select metric to compare",
    available_metrics
)


# =========================================================
# BARPLOT COMPARISON
# =========================================================
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=df_results,
    x="Model",
    y=metric_selected,
    hue="Feature Set",
    palette="Set2",
    ax=ax
)

ax.set_title(
    f"Comparison by {metric_selected}",
    fontsize=14
)
ax.set_xlabel("")
ax.set_ylabel(metric_selected)

st.pyplot(fig)

st.divider()


# =========================================================
# FULL VS REDUCED COMPARISON TABLE
# =========================================================
st.subheader("Detailed Metrics Table")

st.dataframe(
    df_results
        .sort_values(metric_selected, ascending=False)
        .style
        .format("{:.4f}", subset=available_metrics),
    use_container_width=True
)


# =========================================================
# BEST MODEL HIGHLIGHT
# =========================================================
best_row = df_results.loc[
    df_results[metric_selected].idxmax()
]

st.subheader("Best Model (by selected metric)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", best_row["Model"])

with col2:
    st.metric("Feature Set", best_row["Feature Set"])

with col3:
    st.metric(
        metric_selected,
        f"{best_row[metric_selected]:.4f}"
    )


# =========================================================
# PARAMETER INSPECTION (OPTIONAL)
# =========================================================
with st.expander("View model parameters"):
    selected_key = f"{best_row['Model'].lower()}_{best_row['Feature Set'].lower()}"
    params = results[selected_key]["params"]
    st.json(params)
