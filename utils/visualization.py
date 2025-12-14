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
        color=sns.color_palette("Set2")[0],
        ax=ax,
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
        ax=ax,
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
        ax=ax,
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
        x="Importance",
        y="Feature",
        color=sns.color_palette("Blues_r")[2],
        ax=ax,
    )

    ax.set_title("Top Feature Importances", fontsize=13)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")

    return fig


# =========================================================
# CORRELATION HEATMAP
# =========================================================
def plot_correlation_heatmap(corr_df: pd.DataFrame):
    """
    Heatmap for a correlation matrix DataFrame.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_df,
        cmap="RdBu_r",
        center=0,
        annot=False,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Correlation matrix", fontsize=13)
    return fig


# =========================================================
# CONFUSION MATRIX
# =========================================================
def plot_confusion_matrix(cm: pd.DataFrame):
    """
    Display a confusion matrix with consistent styling.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix", fontsize=13)
    return fig
