"""Command-line entrypoint for executing project phases."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable

import pandas as pd

from src.data import database, loaders, validation
from src.analysis import eda, sentiment_patterns
from src.models import training
from src.visualization import charts

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
VISUALIZATIONS = ROOT / "visualizations"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

for path in (DATA_PROCESSED, VISUALIZATIONS, REPORTS_DIR, LOGS_DIR):
    path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run_phase1() -> None:
    logging.info("Loading raw dataset")
    df_raw = loaders.load_raw_dataset()
    issues = loaders.detect_column_issues(df_raw)
    df_norm = loaders.normalize_dataframe(df_raw)

    snapshot = validation.build_snapshot(df_norm)
    quality_path = REPORTS_DIR / "data_quality_report.md"

    logging.info("Saving normalized dataset and quality report")
    df_norm.to_parquet(DATA_PROCESSED / "tweets_normalized.parquet", index=False)
    df_norm.to_csv(DATA_PROCESSED / "tweets_normalized.csv", index=False)

    markdown = snapshot.to_markdown()
    markdown += "\n\n## Column Comparison\n"
    markdown += f"- Missing columns: {issues['missing']}\n"
    markdown += f"- Unexpected columns: {issues['unexpected']}\n"
    quality_path.write_text(markdown, encoding="utf-8")

    report_data = {
        "row_count": snapshot.row_count,
        "column_count": snapshot.column_count,
        "duplicate_rows": snapshot.duplicate_rows,
        "missing_values": snapshot.missing_values,
        "column_issues": issues,
    }
    (DATA_PROCESSED / "phase1_summary.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    logging.info("Phase 1 completed: data quality report generated")


def run_phase2() -> None:
    logging.info("Loading normalized dataset for EDA")
    data_path = DATA_PROCESSED / "tweets_normalized.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Normalized dataset missing. Run phase1 before phase2."
        )
    df = pd.read_parquet(data_path)

    logging.info("Computing EDA summaries")
    sentiment_df = eda.sentiment_distribution(df)
    airline_df = eda.airline_volume(df)
    timeline_df = eda.temporal_trend(df)
    negative_df = eda.top_negative_reasons(df)

    eda.export_dataframe(sentiment_df, DATA_PROCESSED / "sentiment_distribution.csv")
    eda.export_dataframe(airline_df, DATA_PROCESSED / "airline_tweet_volume.csv")
    eda.export_dataframe(timeline_df, DATA_PROCESSED / "timeline_daily.csv")
    eda.export_dataframe(negative_df, DATA_PROCESSED / "top_negative_reasons.csv")

    markdown = eda.summarize_for_markdown(df)
    (REPORTS_DIR / "eda_summary.md").write_text(markdown, encoding="utf-8")

    logging.info("Phase 2 completed: EDA artifacts saved")


def run_phase3() -> None:
    logging.info("Loading dataset for sentiment pattern analysis")
    data_path = DATA_PROCESSED / "tweets_normalized.parquet"
    df = pd.read_parquet(data_path)

    logging.info("Computing sentiment metrics per airline")
    sentiment_airline_df = sentiment_patterns.sentiment_by_airline(df)
    negative_summary_df = sentiment_patterns.negative_reason_summary(df)
    sentiment_airline_df.to_csv(
        DATA_PROCESSED / "sentiment_by_airline.csv", index=False
    )
    negative_summary_df.to_csv(
        DATA_PROCESSED / "negative_reason_by_airline.csv", index=False
    )

    logging.info("Mining association rules")
    rules_df = sentiment_patterns.mine_association_rules(
        df,
        min_support=0.01,
        min_confidence=0.3,
        max_rules=25,
    )
    if len(rules_df) < 5:
        logging.warning(
            "Fewer than 5 rules discovered with default thresholds; relaxing parameters"
        )
        relaxed_rules_df = sentiment_patterns.mine_association_rules(
            df,
            min_support=0.005,
            min_confidence=0.2,
            max_rules=25,
        )
        if not relaxed_rules_df.empty:
            rules_df = relaxed_rules_df

    if rules_df.empty:
        logging.warning("Association rule mining did not yield any results")
    else:
        rules_df.to_csv(DATA_PROCESSED / "association_rules.csv", index=False)

    logging.info("Generating negative sentiment word cloud")
    wordcloud_path = sentiment_patterns.generate_wordcloud(
        df,
        sentiment="negative",
        output_path=VISUALIZATIONS / "wordcloud_negative.png",
        additional_stopwords={"http", "https", "co", "amp"},
    )

    markdown_lines = ["# Sentiment Pattern Analysis", ""]
    markdown_lines.append("## Sentiment by Airline")
    markdown_lines.append(sentiment_airline_df.to_markdown(index=False))
    markdown_lines.append("")
    markdown_lines.append("## Negative Reasons by Airline (Top 20)")
    markdown_lines.append(
        negative_summary_df.head(20).to_markdown(index=False)
    )
    markdown_lines.append("")
    if rules_df.empty:
        markdown_lines.append("No association rules discovered with the selected thresholds.")
    else:
        markdown_lines.append("## Association Rules")
        markdown_lines.append(rules_df.head(20).to_markdown(index=False))
    markdown_lines.append("")
    markdown_lines.append("## Word Cloud")
    markdown_lines.append(f"![Negative sentiment word cloud]({wordcloud_path.as_posix()})")

    (REPORTS_DIR / "sentiment_analysis.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    logging.info("Phase 3 completed: sentiment analysis artifacts ready")


def run_phase4() -> training.ModelResult:
    logging.info("Training sentiment classification models")
    data_path = DATA_PROCESSED / "tweets_normalized.parquet"
    df = pd.read_parquet(data_path)

    result = training.select_and_train_best(df)

    charts.plot_confusion_matrix(
        result.confusion_matrix,
        result.labels,
        VISUALIZATIONS / "model_confusion_matrix.png",
    )

    logging.info(
        "Phase 4 completed: %s accuracy %.3f",
        result.name,
        result.accuracy,
    )
    return result


def run_phase5() -> None:
    logging.info("Generating visualization assets")

    sentiment_df = pd.read_csv(DATA_PROCESSED / "sentiment_distribution.csv")
    airline_df = pd.read_csv(DATA_PROCESSED / "airline_tweet_volume.csv")
    timeline_df = pd.read_csv(DATA_PROCESSED / "timeline_daily.csv")
    negative_df = pd.read_csv(DATA_PROCESSED / "top_negative_reasons.csv")
    sentiment_airline_df = pd.read_csv(DATA_PROCESSED / "sentiment_by_airline.csv")

    generated = []
    generated.append(
        charts.plot_sentiment_distribution(
            sentiment_df, VISUALIZATIONS / "sentiment_distribution.png"
        )
    )
    generated.append(
        charts.create_plotly_sentiment_distribution(
            sentiment_df, VISUALIZATIONS / "sentiment_distribution.html"
        )
    )
    generated.append(
        charts.plot_airline_sentiment_heatmap(
            sentiment_airline_df, VISUALIZATIONS / "airline_sentiment_heatmap.png"
        )
    )
    generated.append(
        charts.plot_timeline(timeline_df, VISUALIZATIONS / "tweet_volume_timeline.png")
    )
    generated.append(
        charts.plot_negative_reasons(
            negative_df,
            VISUALIZATIONS / "top_negative_reasons.png",
            top_n=10,
        )
    )

    summary_lines = ["# Visualization Assets", ""]
    for path in generated:
        summary_lines.append(f"- {Path(path).name}")
    summary_lines.append("- wordcloud_negative.png")
    summary_lines.append("- model_confusion_matrix.png")

    (REPORTS_DIR / "visualization_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    logging.info("Phase 5 completed: visualization files saved")


def run_phase6() -> None:
    logging.info("Generating Markdown reports")
    from src.reports.generator import generate_reports  # local import to avoid cycle

    generate_reports()
    logging.info("Phase 6 completed: reports generated")


def run_phase7() -> None:
    logging.info("Initializing MySQL database and tables")
    database.initialize_database()
    logging.info("Phase 7 completed: database ready")

def run_phase8() -> None:
    logging.info("Validating Flask web application setup")
    from app import create_app  # local import to avoid circular dependency

    app = create_app()
    with app.app_context():
        logging.info("Flask app context initialized with routes: %s", list(app.url_map.iter_rules()))
    logging.info("Phase 8 completed: web dashboard configured")





PHASE_FUNCTIONS: Dict[str, Callable[[], object]] = {
    "phase1": run_phase1,
    "phase2": run_phase2,
    "phase3": run_phase3,
    "phase4": run_phase4,
    "phase5": run_phase5,
    "phase6": run_phase6,
    "phase7": run_phase7,
    "phase8": run_phase8,
}

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute specified project phase")
    parser.add_argument(
        "--phase",
        required=True,
        choices=sorted(PHASE_FUNCTIONS.keys()),
        help="Phase identifier to execute",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    logging.info("Starting %s", args.phase)
    PHASE_FUNCTIONS[args.phase]()
    logging.info("Completed %s", args.phase)


if __name__ == "__main__":
    main()



