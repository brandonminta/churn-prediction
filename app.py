import streamlit as st

from utils.layout import render_sidebar

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SIDEBAR (GLOBAL)
# =========================================================
render_sidebar()

# =========================================================
# MAIN LANDING PAGE
# =========================================================
st.title("Churn Prediction")

st.markdown(
    """
    Portal interactivo para evaluar riesgo de churn en Telco. La
    aplicación permite explorar los datos, comparar distintos modelos de
    machine learning y realizar predicciones individuales.
    """
)

st.divider()

st.subheader("Funcionalidades principales")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Predicción**

        Ejecución de inferencias con versiones *full* o *reduced* de features.
        """
    )
    st.page_link("pages/prediccion.py", label="Ir a Predicción")

with col2:
    st.markdown(
        """
        **Comparación**

        Métricas y parámetros clave de cada modelo bajo las mismas condiciones.
        """
    )
    st.page_link("pages/comparacion.py", label="Ver Comparación")

with col3:
    st.markdown(
        """
        **EDA interactivo**

        Distribuciones y relaciones nuestra variable objectivo para entender nuestra base de datos y sus variables críticas.
        """
    )
    st.page_link("pages/dashboard.py", label="Explorar EDA")
