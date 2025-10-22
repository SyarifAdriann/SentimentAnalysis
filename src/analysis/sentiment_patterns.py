"""Sentiment analysis and pattern discovery utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from wordcloud import STOPWORDS, WordCloud


def sentiment_by_airline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sentiment counts per airline."""

    pivot = (
        df.pivot_table(
            index="airline",
            columns="airline_sentiment",
            values="tweet_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    total = pivot.drop(columns=["airline"]).sum(axis=1)
    pivot["total"] = total
    for column in pivot.columns:
        if column not in {"airline", "total"}:
            pivot[f"{column}_pct"] = (pivot[column] / total * 100).round(2)
    return pivot


def negative_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count negative reasons overall and per airline."""

    subset = df.dropna(subset=["negativereason"])
    counts = (
        subset.groupby(["airline", "negativereason"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return counts


def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    subset = df.dropna(subset=["negativereason"])
    transactions: list[list[str]] = []
    for row in subset.itertuples(index=False):
        transaction = [f"airline={row.airline}", f"reason={row.negativereason}"]
        transaction.append(f"sentiment={row.airline_sentiment}")
        transactions.append(transaction)
    return transactions


def mine_association_rules(
    df: pd.DataFrame,
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.3,
    max_rules: int = 20,
) -> pd.DataFrame:
    """Generate association rules linking airlines and negative reasons."""

    transactions = build_transactions(df)
    if not transactions:
        return pd.DataFrame()

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(encoded, columns=encoder.columns_)

    frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()

    columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    rules = rules[columns]
    rules = rules.sort_values("confidence", ascending=False).head(max_rules)
    rules["antecedents"] = rules["antecedents"].apply(lambda items: ", ".join(sorted(items)))
    rules["consequents"] = rules["consequents"].apply(lambda items: ", ".join(sorted(items)))
    return rules.reset_index(drop=True)


def generate_wordcloud(
    df: pd.DataFrame,
    *,
    sentiment: str = "negative",
    output_path: Path,
    max_words: int = 200,
    width: int = 1200,
    height: int = 800,
    additional_stopwords: Optional[set[str]] = None,
) -> Path:
    """Generate and save a word cloud for tweets with the given sentiment."""

    subset = df[df["airline_sentiment"].str.lower() == sentiment.lower()]
    text_blob = " ".join(subset["text"].dropna().astype(str))
    if not text_blob:
        raise ValueError(f"No tweets found for sentiment '{sentiment}'")

    stopwords = set(STOPWORDS)
    if additional_stopwords:
        stopwords.update(additional_stopwords)

    cloud = WordCloud(
        background_color="white",
        width=width,
        height=height,
        stopwords=stopwords,
        collocations=False,
        max_words=max_words,
    ).generate(text_blob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_file(str(output_path))
    return output_path
