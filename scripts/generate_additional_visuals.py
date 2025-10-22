"""Generate extended visualization assets for the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.analysis import sentiment_patterns  # noqa: E402
from src.visualization import charts  # noqa: E402

DATA_PROCESSED = ROOT / "data" / "processed"
VISUALIZATIONS = ROOT / "visualizations"


def build_daily_sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.assign(date=df["tweet_created"].dt.date)
        .groupby(["date", "airline_sentiment"], dropna=False)["tweet_id"]
        .count()
        .reset_index(name="count")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(columns={"airline_sentiment": "sentiment"})
    return daily


def main() -> None:
    normalized_path = DATA_PROCESSED / "tweets_normalized.parquet"
    sentiment_airline_path = DATA_PROCESSED / "sentiment_by_airline.csv"
    negative_reason_path = DATA_PROCESSED / "negative_reason_by_airline.csv"

    df = pd.read_parquet(normalized_path)
    sentiment_airline_df = pd.read_csv(sentiment_airline_path)
    negative_reason_df = pd.read_csv(negative_reason_path)

    VISUALIZATIONS.mkdir(parents=True, exist_ok=True)

    charts.plot_sentiment_share_per_airline(
        sentiment_airline_df,
        VISUALIZATIONS / "sentiment_share_per_airline.png",
    )

    charts.plot_negative_ratio_by_airline(
        sentiment_airline_df,
        VISUALIZATIONS / "negative_ratio_per_airline.png",
    )

    charts.plot_negative_reason_heatmap(
        negative_reason_df,
        VISUALIZATIONS / "negative_reason_heatmap.png",
        top_reasons=8,
    )

    daily_sentiment_df = build_daily_sentiment_counts(df)
    charts.plot_daily_sentiment_trend(
        daily_sentiment_df,
        VISUALIZATIONS / "daily_sentiment_trend.png",
    )

    for sentiment in ("positive", "neutral"):
        sentiment_patterns.generate_wordcloud(
            df,
            sentiment=sentiment,
            output_path=VISUALIZATIONS / f"wordcloud_{sentiment}.png",
            additional_stopwords={"http", "https", "co", "amp"},
        )

    print("Additional visualizations generated in", VISUALIZATIONS)


if __name__ == "__main__":
    main()
