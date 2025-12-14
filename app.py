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
st.title("Plataforma de churn prediction")

st.markdown(
    """
    Portal interactivo para evaluar riesgo de churn en el dataset Telco. La
    aplicación organiza el flujo completo (EDA, comparación de modelos y
    predicción) en páginas modulares con navegación consistente.
    """
)

st.divider()

st.subheader("Qué ofrece la aplicación")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Predicción**

        Ejecución de inferencias con versiones *full* o *reduced* de features,
        respetando el mismo pipeline del entrenamiento.
        """
    )

with col2:
    st.markdown(
        """
        **Comparación**

        Métricas y parámetros clave de cada modelo entrenado en un mismo
        conjunto de validación.
        """
    )

with col3:
    st.markdown(
        """
        **EDA interactivo**

        Distribuciones y relaciones con churn para entender el dataset de
        entrada y sus variables críticas.
        """
    )

st.info("Utiliza el panel lateral para acceder directamente a cada sección.")
