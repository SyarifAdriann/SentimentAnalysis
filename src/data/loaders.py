"""Utilities for loading the Tweets.csv dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

EXPECTED_COLUMNS: Tuple[str, ...] = (
    "tweet_id",
    "airline_sentiment",
    "airline_sentiment_confidence",
    "negativereason",
    "negativereason_confidence",
    "airline",
    "airline_sentiment_gold",
    "name",
    "negativereason_gold",
    "retweet_count",
    "text",
    "tweet_coord",
    "tweet_created",
    "tweet_location",
    "user_timezone",
)


def load_raw_dataset(path: Path | str = Path("data/raw/Tweets.csv"), *, encoding: str = "utf-8") -> pd.DataFrame:
    """Load the Tweets.csv dataset from disk.

    Parameters
    ----------
    path:
        Location of the dataset file. Defaults to the expected raw data path.
    encoding:
        File encoding used when reading the CSV.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    return df


def detect_column_issues(df: pd.DataFrame, expected: Iterable[str] = EXPECTED_COLUMNS) -> dict[str, List[str]]:
    """Compare dataframe columns against the expected schema."""

    actual = list(df.columns)
    expected_list = list(expected)

    missing = [col for col in expected_list if col not in actual]
    unexpected = [col for col in actual if col not in expected_list]

    return {"missing": missing, "unexpected": unexpected}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with useful type conversions applied."""

    normalized = df.copy()

    if "tweet_created" in normalized.columns:
        normalized["tweet_created"] = pd.to_datetime(
            normalized["tweet_created"], errors="coerce"
        )

    if "tweet_id" in normalized.columns:
        normalized["tweet_id"] = normalized["tweet_id"].astype(str)

    if "airline_sentiment" in normalized.columns:
        normalized["airline_sentiment"] = (
            normalized["airline_sentiment"].astype(str).str.lower()
        )

    return normalized


def load_normalized_dataset(path: Path | str = Path("data/raw/Tweets.csv"), *, encoding: str = "utf-8") -> pd.DataFrame:
    """Load the dataset and return a normalised dataframe."""

    raw_df = load_raw_dataset(path=path, encoding=encoding)
    return normalize_dataframe(raw_df)
