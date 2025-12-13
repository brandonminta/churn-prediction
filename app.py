import streamlit as st

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("Churn Prediction")
    st.markdown(
        """
        **Interactive machine learning application** for customer churn analysis.

        This app allows you to:
        - Run churn predictions using trained models
        - Compare multiple model versions
        - Explore the dataset interactively
        """
    )

    st.divider()

    st.markdown(
        """
        **Navigation**
        Use the menu above to switch between:
        - Inference
        - Model Comparison
        - EDA Dashboard
        """
    )

    st.divider()

    st.markdown(
        """
        **Project context**
        Telco Customer Churn  
        Midterm Exam – Machine Learning
        """
    )

# =========================================================
# MAIN LANDING PAGE
# =========================================================
st.title("Customer Churn Prediction Application")

st.markdown(
    """
    This application showcases a complete machine learning workflow,
    from exploratory data analysis to model comparison and interactive inference.

    All preprocessing is handled through a custom pipeline, ensuring consistency
    between training and deployment.
    """
)

st.divider()

st.subheader("Available Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### Inference
        Predict customer churn using different trained models and feature sets.
        """
    )

with col2:
    st.markdown(
        """
        ### Model Comparison
        Compare performance metrics across multiple models and feature configurations.
        """
    )

with col3:
    st.markdown(
        """
        ### EDA Dashboard
        Explore the Telco dataset interactively and analyze relationships with churn.
        """
    )

st.info(
    "Use the sidebar to navigate between modules."
)
