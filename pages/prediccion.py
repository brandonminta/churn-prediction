# pages/1_Inferencia.py

import streamlit as st
import pandas as pd

from utils.loader import (
    load_model,
    load_feature_map,
    available_models,
    available_feature_sets
)
from utils.preprocessing import build_preprocessing_pipeline


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction | Inference",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.caption(
    "Interactive inference using trained machine learning models. "
    "All preprocessing is handled automatically through a robust pipeline."
)

st.divider()


# =========================================================
# LOAD SHARED ARTIFACTS (CACHED)
# =========================================================
@st.cache_resource
def get_feature_map():
    return load_feature_map()


feature_map = get_feature_map()


# =========================================================
# MODEL SELECTION
# =========================================================
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        model_name = st.selectbox(
            "Model",
            available_models(),
            help="Choose the trained model to use for inference."
        )

    with col2:
        feature_set = st.selectbox(
            "Feature Set",
            available_feature_sets(),
            help="Full: all features | Reduced: top selected features."
        )


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def get_model(name, fs):
    return load_model(name, fs)


model = get_model(model_name, feature_set)


# =========================================================
# INPUT FORM (DYNAMIC)
# =========================================================
st.subheader("Customer Information")

raw_input = {}

groups = feature_map["groups"]
dependencies = feature_map["dependencies"]

# --- Layout per group ---
for group_name, features in groups.items():
    with st.expander(group_name, expanded=True):
        cols = st.columns(3)

        for i, (feature, options) in enumerate(features.items()):
            with cols[i % 3]:

                # Numeric features
                if options == "numeric":
                    raw_input[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        step=1.0
                    )

                # Categorical features
                else:
                    raw_input[feature] = st.selectbox(
                        feature,
                        options
                    )


# =========================================================
# APPLY DEPENDENCIES (POST-UI)
# =========================================================
for parent_feature, rules in dependencies.items():
    parent_value = raw_input.get(parent_feature)

    if parent_value in rules:
        for child_feature, forced_value in rules[parent_value].items():
            raw_input[child_feature] = forced_value


# =========================================================
# INFERENCE
# =========================================================
st.divider()

if st.button("Run Prediction", use_container_width=True):

    # -----------------------------
    # Build DataFrame (RAW)
    # -----------------------------
    X_raw = pd.DataFrame([raw_input])

    # -----------------------------
    # Build & Apply Pipeline
    # -----------------------------
    pipeline = build_preprocessing_pipeline(X_raw)
    pipeline.fit(X_raw)  # Safe because encoders already defined structurally

    X_prep = pipeline.transform(X_raw)

    # -----------------------------
    # Prediction
    # -----------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_prep)[0][1]
        prediction = int(proba >= 0.5)
    else:
        prediction = model.predict(X_prep)[0]
        proba = None

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("High risk of churn")
        else:
            st.success("Low risk of churn")

    with col2:
        if proba is not None:
            st.metric(
                
                label="Churn Probability",
                value=f"{proba*100:.1f}%"
            )
