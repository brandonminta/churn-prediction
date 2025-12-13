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
        st.caption(
            "Suite en Streamlit para predicción de churn, benchmarking y EDA del "
            "dataset Telco Customer Churn."
        )

        st.divider()

        st.subheader("Navegación")
        for page, label in SIDEBAR_PAGES:
            st.page_link(page, label=label)

        st.divider()
        st.caption(
            "Modelos entrenados con pipelines reproducibles en versiones full y "
            "reduced de features."
        )
