# Churn Prediction

Aplicación web en Streamlit para el dataset **Telco Customer Churn**. Incluye tres
módulos principales:

- **Predicción:** inferencia interactiva con modelos entrenados (versiones *full*
  y *reduced* de features) y pipeline de preprocesamiento consistente.
- **Comparación:** visualización de métricas y parámetros de los modelos
  disponibles.
- **EDA:** panel exploratorio para inspeccionar la distribución de variables y su
  relación con el churn.

## Requisitos

La aplicación está pensada para Python 3.11 y depende de las librerías listadas en
`requirements.txt`.

## Cómo ejecutar

1. Crea y activa un entorno virtual.
2. Instala las dependencias: `pip install -r requirements.txt`.
3. Lanza la app: `streamlit run app.py`.

El servidor abrirá `http://localhost:8501`.

## Uso rápido

- **Inicio:** desde la portada navega a cualquiera de las secciones laterales.
- **Predicción:** completa el formulario de características y obtén la probabilidad de churn para un cliente.
- **Comparación:** revisa las métricas de los modelos entrenados (*full* y *reduced*) y sus hiperparámetros.
- **EDA:** explora distribuciones y relaciones de variables con gráficos interactivos.

## Estructura

- `app.py`: landing page y navegación.
- `pages/`: páginas de Streamlit para predicción, comparación y EDA.
- `utils/`: funciones de carga de artefactos, preprocesamiento y visualización.
- `data/`: dataset original (`telco.parquet`) y mapa de features para la UI.
- `models/`, `results/`: artefactos entrenados y métricas asociadas.
