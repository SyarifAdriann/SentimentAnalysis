"""Exploratory data analysis utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return tweet counts and percentages by sentiment."""

    counts = df["airline_sentiment"].value_counts(dropna=False).rename("count")
    percentages = (counts / counts.sum() * 100).round(2).rename("percentage")
    result = pd.concat([counts, percentages], axis=1)
    result.index.name = "sentiment"
    result = result.reset_index()
    return result


def airline_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Return total tweet count per airline."""

    counts = df["airline"].value_counts(dropna=False).rename("tweet_count")
    result = counts.reset_index().rename(columns={"index": "airline"})
    return result


def temporal_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate tweet counts by time frequency."""

    if "tweet_created" not in df.columns:
        raise KeyError("tweet_created column is required for temporal trends")

    timeline = (
        df.set_index("tweet_created")
        .resample(freq)["tweet_id"]
        .count()
        .rename("tweet_count")
        .reset_index()
    )
    return timeline


def top_negative_reasons(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the most frequent negative reasons."""

    if "negativereason" not in df.columns:
        raise KeyError("negativereason column is required")

    counts = (
        df["negativereason"]
        .dropna()
        .value_counts()
        .head(top_n)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "negativereason"})
    )
    return counts


def summarize_for_markdown(df: pd.DataFrame) -> str:
    """Create a markdown-formatted EDA overview."""

    sentiment_df = sentiment_distribution(df)
    airline_df = airline_volume(df)
    negative_df = top_negative_reasons(df)

    lines = ["# Exploratory Data Analysis Summary", ""]

    lines.append("## Sentiment Distribution")
    lines.append(sentiment_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Tweet Volume by Airline")
    lines.append(airline_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Top Negative Reasons")
    lines.append(negative_df.to_markdown(index=False))

    return "\n".join(lines)


def export_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_excel(path, index=False)
