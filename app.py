import streamlit as st

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("Churn Prediction")
    st.caption(
        "Suite ligera en Streamlit para inferencia, benchmarking y análisis "
        "exploratorio del dataset Telco Customer Churn."
    )

    st.divider()

    st.subheader("Navegación")
    st.page_link("app.py", label="Inicio", icon="🏠")
    st.page_link("pages/prediccion.py", label="Predicción", icon="🤖")
    st.page_link("pages/comparacion.py", label="Comparación", icon="📈")
    st.page_link("pages/dashboard.py", label="EDA", icon="🧭")

    st.divider()

    st.caption("Modelos entrenados con pipelines reproducibles y versiones full/reduced de features.")

# =========================================================
# MAIN LANDING PAGE
# =========================================================
st.title("Plataforma de churn prediction")

st.markdown(
    """
    Portal interactivo para evaluar riesgo de churn en el dataset Telco. La
    aplicación organiza el flujo completo (EDA, comparación de modelos y
    predicción) en páginas modulares con una navegación mínima y textos
    descriptivos.
    """
)

st.divider()

st.subheader("Módulos disponibles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### Predicción
        Interfaz para ejecutar inferencias con versiones full o reduced del
        pipeline de features.
        """
    )
    st.page_link("pages/prediccion.py", label="Ir a Predicción", icon="➡️")

with col2:
    st.markdown(
        """
        ### Comparación
        Visualiza métricas y parámetros clave de cada modelo entrenado.
        """
    )
    st.page_link("pages/comparacion.py", label="Ir a Comparación", icon="➡️")

with col3:
    st.markdown(
        """
        ### EDA interactivo
        Explora la distribución de las variables y su relación con el churn.
        """
    )
    st.page_link("pages/dashboard.py", label="Ir al Dashboard", icon="➡️")

st.info(
    "Utiliza los accesos del panel lateral o los botones superiores para navegar entre las secciones."
)
