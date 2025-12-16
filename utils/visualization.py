import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# =========================================================
# GLOBAL STYLE
# =========================================================
def set_style():
    sns.set_theme(style="whitegrid")


# =========================================================
# EDA PLOTS
# =========================================================
def plot_numeric_by_churn(
    df: pd.DataFrame,
    column: str,
    churn_col: str = "Churn",
):
    """
    Boxplot of a numerical variable grouped by churn.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x=churn_col,
        y=column,
        hue=churn_col,
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title(f"{column} by {churn_col}", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel(column)

    return fig


def plot_categorical_by_churn(
    df: pd.DataFrame,
    column: str,
    churn_col: str = "Churn",
):
    """
    Countplot of a categorical variable grouped by churn.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        hue=churn_col,
        palette="Set2",
        ax=ax
    )

    ax.set_title(f"{column} by {churn_col}", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    return fig


# =========================================================
# MODEL COMPARISON
# =========================================================
def plot_metric_comparison(
    df_results: pd.DataFrame,
    metric: str
):
    """
    Barplot comparing models by a selected metric.
    """
    df_plot = df_results.copy()
    df_plot[metric] = pd.to_numeric(df_plot[metric], errors="coerce")
    df_plot = df_plot.dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=df_plot,
        x="Model",
        y=metric,
        hue="Feature Set",
        palette="Set2",
        ax=ax
    )

    ax.set_title(f"Comparison by {metric}", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel(metric)

    return fig


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
def plot_feature_importance(
    df_importance: pd.DataFrame,
    top_n: int = 15
):
    """
    Horizontal barplot of feature importance.
    """
    df_plot = df_importance.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=df_plot,
        x="importance",
        y="feature",
        hue="feature",
        palette="Blues_r",
        legend=False,
        ax=ax)

    ax.set_title("Top Feature Importances", fontsize=13)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")

    return fig


# =========================================================
# Matrix de confusion
# =========================================================

def plot_confusion_matrix(
    cm,
    labels=None,
    normalize: bool = False,
    title: str = "Confusion Matrix"
):
    """
    Plot a confusion matrix from a precomputed array or DataFrame.

    Parameters
    ----------
    cm : array-like (2x2) or pd.DataFrame
        Confusion matrix.
    labels : tuple or None
        Class labels (used only if cm is array-like).
    normalize : bool
        Normalize rows (recall-based).
    title : str
        Plot title.
    """

    df_cm = cm.copy()

    if normalize:
        df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
        title = f"{title} (Normalized)"

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        df_cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=13)

    return fig

def plot_correlation_matrix(
    corr: pd.DataFrame,
    title: str = "Correlation Matrix"
):
    """
    Plot correlation matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return fig
