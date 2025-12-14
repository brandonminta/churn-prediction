import streamlit as st


SIDEBAR_PAGES = [
    ("app.py", "Inicio"),
    ("pages/prediccion.py", "Predicción"),
    ("pages/comparacion.py", "Comparación"),
    ("pages/dashboard.py", "EDA"),
]


def render_sidebar():
    with st.sidebar:
        st.title("Churn Prediction")

        st.subheader("Navegación")
        for page, label in SIDEBAR_PAGES:
            st.page_link(page, label=label)
