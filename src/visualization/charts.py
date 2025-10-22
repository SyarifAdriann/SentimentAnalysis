"""Visualization helpers for generating static and interactive charts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_sentiment_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sentiment", y="count", palette="viridis")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_airline_sentiment_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)

    sentiment_cols = [col for col in df.columns if col not in {"airline", "total"} and not col.endswith("_pct")]
    heatmap_df = df.set_index("airline")[sentiment_cols]

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, fmt="g", cmap="coolwarm")
    plt.title("Sentiment Counts by Airline")
    plt.ylabel("Airline")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_timeline(df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="tweet_created", y="tweet_count")
    plt.title("Tweet Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_negative_reasons(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)

    subset = df.head(top_n)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=subset, x="count", y="negativereason", palette="magma")
    plt.title("Top Negative Reasons")
    plt.xlabel("Count")
    plt.ylabel("Negative Reason")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_confusion_matrix(matrix: Sequence[Sequence[int]], labels: Iterable[str], output_path: Path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)

    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def create_plotly_sentiment_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)

    fig = px.pie(df, names="sentiment", values="count", title="Sentiment Share")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_sentiment_share_per_airline(df: pd.DataFrame, output_path: Path) -> Path:
    """Create a stacked bar chart showing sentiment composition per airline."""

    output_path = Path(output_path)
    _ensure_parent(output_path)

    sentiment_cols = [col for col in ("negative", "neutral", "positive") if col in df.columns]
    if not sentiment_cols:
        raise ValueError("Input dataframe must include sentiment count columns (negative/neutral/positive).")

    share_df = df.set_index("airline")[sentiment_cols]
    share_df = share_df.div(share_df.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    share_df.sort_values(by=sentiment_cols, ascending=False, inplace=False).plot(
        kind="bar",
        stacked=True,
        color=["#d63031", "#fdcb6e", "#00b894"][: len(sentiment_cols)],
        ax=plt.gca(),
    )
    plt.ylabel("Percentage of Tweets")
    plt.xlabel("Airline")
    plt.title("Sentiment Share per Airline")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_daily_sentiment_trend(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot daily sentiment counts for negative, neutral, and positive tweets."""

    output_path = Path(output_path)
    _ensure_parent(output_path)

    required_cols = {"date", "sentiment", "count"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input dataframe must contain columns: {required_cols}")

    pivot_df = df.pivot(index="date", columns="sentiment", values="count").fillna(0)
    pivot_df.sort_index(inplace=True)

    plt.figure(figsize=(12, 6))
    for sentiment in pivot_df.columns:
        sns.lineplot(x=pivot_df.index, y=pivot_df[sentiment], label=sentiment.title())
    plt.title("Daily Sentiment Trend")
    plt.xlabel("Date")
    plt.ylabel("Tweet Count")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_negative_reason_heatmap(df: pd.DataFrame, output_path: Path, top_reasons: int = 8) -> Path:
    """Visualize the frequency of top negative reasons across airlines."""

    output_path = Path(output_path)
    _ensure_parent(output_path)

    subset = df.copy()
    subset = subset.sort_values("count", ascending=False)
    top_reason_names = subset["negativereason"].dropna().unique()[:top_reasons]
    subset = subset[subset["negativereason"].isin(top_reason_names)]

    heatmap_df = subset.pivot_table(
        index="airline",
        columns="negativereason",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, fmt="g", cmap="Reds")
    plt.title("Top Negative Reasons by Airline")
    plt.ylabel("Airline")
    plt.xlabel("Negative Reason")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_negative_ratio_by_airline(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot negative sentiment ratio per airline."""

    output_path = Path(output_path)
    _ensure_parent(output_path)

    required_cols = {"airline", "negative", "total"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input dataframe must contain columns: {required_cols}")

    ratio_df = df.copy()
    ratio_df["negative_ratio"] = ratio_df["negative"] / ratio_df["total"] * 100

    plt.figure(figsize=(10, 5))
    sns.barplot(data=ratio_df, x="airline", y="negative_ratio", palette="rocket")
    plt.ylabel("Negative Sentiment (%)")
    plt.xlabel("Airline")
    plt.title("Negative Sentiment Ratio by Airline")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
