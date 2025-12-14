# utils/loader.py

from pathlib import Path
import pickle
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]


def load_dataset() -> pd.DataFrame:
    """
    Load Telco dataset (Parquet).
    """
    path = BASE_DIR / "data" / "telco.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)


def load_feature_map() -> dict:
    """
    Load Streamlit feature map (features, groups, dependencies).
    """
    path = BASE_DIR / "data" / "streamlit_feature_map.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Feature map not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(
    model_name: str,
    feature_set: str
):
    """
    Load a trained model.

    Parameters
    ----------
    model_name : str
        One of: 'catboost', 'stacking', 'voting'
    feature_set : str
        One of: 'full', 'reduced'
    """
    model_path = (
        BASE_DIR
        / "models"
        / model_name
        / f"{model_name}_{feature_set}_optimized.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)



def load_model_results() -> dict:
    """
    Load metrics and params for all models.
    """
    path = BASE_DIR / "results" / "model_metrics_and_params.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_importance() -> pd.DataFrame:
    """
    Load feature importance table.
    """
    pkl_path = BASE_DIR / "results" / "feature_importance.pkl"
    csv_path = BASE_DIR / "results" / "feature_importance.csv"

    if pkl_path.exists():
        return pd.read_pickle(pkl_path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "Feature importance file not found (pkl or csv)."
    )


def load_correlation_matrix() -> pd.DataFrame:
    """
    Load correlation matrix from CSV.
    """
    path = BASE_DIR / "results" / "correlation_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Correlation matrix not found: {path}")
    return pd.read_csv(path, index_col=0)


def available_models():
    return ["catboost", "stacking", "voting"]


def available_feature_sets():
    return ["full", "reduced"]
