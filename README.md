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

## Estructura

- `app.py`: landing page y navegación.
- `pages/`: páginas de Streamlit para predicción, comparación y EDA.
- `utils/`: funciones de carga de artefactos, preprocesamiento y visualización.
- `data/`: dataset original (`telco.parquet`) y mapa de features para la UI.
- `models/`, `results/`: artefactos entrenados y métricas asociadas.
