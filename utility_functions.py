# The helper functions below enforce a consistent experimental style across the notebook:
# required-column validation, feature filtering, result export, metric ranking, and compact reporting.

from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

def require_columns(df: pd.DataFrame, columns: list[str], table_name: str = "dataframe") -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Return only features available in the dataframe."""
    selected = [col for col in features if col in df.columns]
    if not selected:
        raise ValueError("No requested telemetry features were found in the dataframe.")
    return selected


def validate_binary_target(df: pd.DataFrame, target_col: str) -> None:
    """Validate that the target exists and contains two classes."""
    require_columns(df, [target_col], table_name="episode_df_all")
    n_classes = df[target_col].dropna().nunique()
    if n_classes != 2:
        raise ValueError(f"Expected a binary target in '{target_col}', but found {n_classes} classes.")


def save_table(df: pd.DataFrame, filename: str) -> Path:
    """Save a result dataframe to the paper-ready directory."""

    PAPER_READY_DIR = Path("results/tamas/paper_ready")
    output_path = PAPER_READY_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path


def display_ranked(df: pd.DataFrame, by: str, ascending: bool = False, title: str | None = None) -> pd.DataFrame:
    """Display a dataframe sorted by a metric when the metric exists."""
    if title:
        display(Markdown(f"### {title}"))
    if df.empty:
        display(Markdown("No results were produced."))
        return df
    ranked = df.sort_values(by, ascending=ascending).reset_index(drop=True) if by in df.columns else df.copy()
    display(ranked)


def metric_summary(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Create a compact metric summary for numeric result columns."""
    cols = [col for col in metric_cols if col in df.columns]
    if not cols or df.empty:
        return pd.DataFrame()
    return df[cols].describe().T.reset_index().rename(columns={"index": "metric"})
