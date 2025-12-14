# pages/prediccion.py

import warnings
import streamlit as st
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from utils.loader import (
    load_model,
    load_feature_map,
    available_models,
    available_feature_sets,
    load_dataset,
)
from utils.preprocessing import build_preprocessing_pipeline
from utils.layout import render_sidebar


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction | Predicción",
    layout="wide",
)

render_sidebar()

st.title("Predicción de churn")
st.caption(
    "Ejecute inferencias con los modelos entrenados usando exactamente el mismo "
    "esquema de features y preprocesamiento definido en el entrenamiento."
)

st.divider()


# =========================================================
# LOAD SHARED ARTIFACTS (CACHED)
# =========================================================
@st.cache_resource
def get_feature_map():
    return load_feature_map()


feature_map = get_feature_map()

INTEGER_FEATURES = {"tenure"}


def required_raw_features(feature_map: dict, feature_set: str) -> set:
    encoded_features = feature_map["feature_sets"][feature_set]
    raw = set()
    for feat in encoded_features:
        raw.add(feat.split("_", 1)[0])
    return raw


def filter_groups(groups: dict, required_features: set) -> dict:
    filtered = {}
    for group, feats in groups.items():
        subset = {
            name: opts for name, opts in feats.items()
            if name in required_features
        }
        if subset:
            filtered[group] = subset
    return filtered


def filter_dependencies(dependencies: dict, allowed_features: set) -> dict:
    scoped = {}
    for parent, rules in dependencies.items():
        if parent not in allowed_features:
            continue
        scoped_rules = {}
        for parent_value, child_map in rules.items():
            valid_children = {
                child: forced
                for child, forced in child_map.items()
                if child in allowed_features
            }
            if valid_children:
                scoped_rules[parent_value] = valid_children
        if scoped_rules:
            scoped[parent] = scoped_rules
    return scoped


def dependency_index(dependencies: dict) -> dict:
    """Reverse map: child -> list of (parent, parent_value, forced_value)."""
    child_map = {}
    for parent, rules in dependencies.items():
        for parent_value, child_rules in rules.items():
            for child, forced in child_rules.items():
                child_map.setdefault(child, []).append(
                    (parent, parent_value, forced)
                )
    return child_map


def format_label(field: str) -> str:
    return field.replace("_", " ")


# =========================================================
# MODEL SELECTION
# =========================================================
st.subheader("Configuración del modelo")
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        model_name = st.selectbox(
            "Modelo",
            available_models(),
            help="Selecciona el estimador entrenado para la inferencia.",
        )

    with col2:
        feature_set = st.selectbox(
            "Versión de features",
            available_feature_sets(),
            help=(
                "full: todas las variables del dataset; reduced: subconjunto "
                "curado con las features más informativas."
            ),
        )

st.markdown(
    "Las variables mostradas a continuación se actualizan según el set de "
    "features seleccionado y respetan las dependencias definidas en el mapa de "
    "Streamlit."
)


# =========================================================
# LOAD MODEL WITH VERSION CHECK
# =========================================================
@st.cache_resource
def get_model(name, fs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model_obj = load_model(name, fs)
    return model_obj, w


model, _load_warnings = get_model(model_name, feature_set)
train_version = getattr(model, "__sklearn_version__", None)
if train_version and train_version != sklearn.__version__:
    st.warning(
        "El modelo se entrenó con scikit-learn "
        f"{train_version} y el entorno actual usa {sklearn.__version__}. "
        "Se recomienda alinear las versiones para evitar inconsistencias."
    )


# =========================================================
# INPUT FORM (DYNAMIC)
# =========================================================
st.subheader("Información del cliente")

required_features = required_raw_features(feature_map, feature_set)
required_cols_key = tuple(sorted(required_features))
groups = filter_groups(feature_map["groups"], required_features)
dependencies = filter_dependencies(feature_map["dependencies"], required_features)
reverse_dependencies = dependency_index(dependencies)

raw_input = {}

with st.form("prediction_form"):
    for group_name, features in groups.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(3)

            for i, (feature, options) in enumerate(features.items()):
                with cols[i % 3]:
                    label = format_label(feature)

                    # Check if this feature is constrained by a dependency
                    forced_value = None
                    if feature in reverse_dependencies:
                        for parent, parent_value, forced in reverse_dependencies[feature]:
                            parent_current = raw_input.get(parent, st.session_state.get(parent))
                            if parent_current == parent_value:
                                forced_value = forced
                                disabled = True
                                break

                    if forced_value is not None:
                        raw_input[feature] = forced_value
                        st.selectbox(
                            label,
                            options,
                            index=options.index(forced_value),
                            disabled=True,
                            key=feature,
                            help="Opción fijada por la selección de servicios previa.",
                        )
                        continue

                    if options == "numeric":
                        if feature in INTEGER_FEATURES:
                            raw_input[feature] = st.number_input(
                                label,
                                min_value=0,
                                step=1,
                                format="%d",
                                key=feature,
                            )
                        else:
                            raw_input[feature] = st.number_input(
                                label,
                                min_value=0.0,
                                step=1.0,
                                key=feature,
                            )
                    elif isinstance(options, list) and all(
                        isinstance(opt, (int, float)) for opt in options
                    ):
                        raw_input[feature] = st.selectbox(
                            label,
                            options,
                            key=feature,
                        )
                    else:
                        raw_input[feature] = st.selectbox(
                            label,
                            options,
                            key=feature,
                        )

    submitted = st.form_submit_button("Ejecutar predicción", use_container_width=True)


# =========================================================
# APPLY DEPENDENCIES AND RUN PIPELINE
# =========================================================
@st.cache_resource
def get_reference_data(required_cols: tuple):
    df_ref = load_dataset()
    available_cols = [col for col in required_cols if col in df_ref.columns]
    return df_ref[available_cols].copy()


@st.cache_resource
def get_pipeline(required_cols: tuple, fmap: dict):
    ref = get_reference_data(required_cols)
    pipeline = build_preprocessing_pipeline(ref, fmap)
    pipeline.fit(ref)
    return pipeline


if submitted:
    for parent_feature, rules in dependencies.items():
        parent_value = raw_input.get(parent_feature)
        enforced_children = rules.get(parent_value, {})
        raw_input.update(enforced_children)

    # -----------------------------
    # Build DataFrame (RAW)
    # -----------------------------
    X_raw = pd.DataFrame([raw_input])

    # -----------------------------
    # Build & Apply Pipeline
    # -----------------------------
    pipeline = get_pipeline(required_cols_key, feature_map)
    X_prep = pipeline.transform(X_raw)
    expected_order = feature_map["feature_sets"][feature_set]
    X_prep = X_prep.reindex(columns=expected_order, fill_value=0)

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
    st.subheader("Resultado")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Alto riesgo de churn")
        else:
            st.success("Bajo riesgo de churn")

    with col2:
        if proba is not None:
            st.metric(
                label="Probabilidad de churn",
                value=f"{proba*100:.1f}%",
            )
