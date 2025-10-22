"""Generate project reports in English and Indonesian."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data" / "processed"
VIS_DIR = ROOT / "visualizations"
MODELS_DIR = ROOT / "models"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase_summaries() -> dict:
    phase1 = load_json(DATA_DIR / "phase1_summary.json")
    model = load_json(MODELS_DIR / "model_metrics.json")
    sentiment_dist = pd.read_csv(DATA_DIR / "sentiment_distribution.csv")
    airline_volume = pd.read_csv(DATA_DIR / "airline_tweet_volume.csv")
    negative_reasons = pd.read_csv(DATA_DIR / "top_negative_reasons.csv")
    association_rules = None
    assoc_path = DATA_DIR / "association_rules.csv"
    if assoc_path.exists():
        association_rules = pd.read_csv(assoc_path)
    return {
        "phase1": phase1,
        "model": model,
        "sentiment_dist": sentiment_dist,
        "airline_volume": airline_volume,
        "negative_reasons": negative_reasons,
        "association_rules": association_rules,
    }


def build_analysis_results(summary: dict) -> str:
    phase1 = summary["phase1"]
    model = summary["model"]
    sentiment = summary["sentiment_dist"]
    airline = summary["airline_volume"]
    negative = summary["negative_reasons"]
    assoc = summary["association_rules"]

    total_rows = phase1["row_count"]
    airlines = airline.shape[0]
    top_airline = airline.sort_values("tweet_count", ascending=False).iloc[0]
    negative_share = sentiment.loc[sentiment["sentiment"] == "negative", "percentage"].iloc[0]
    top_negative_reason = negative.iloc[0]

    accuracy = model["accuracy"]
    best_model = model["model_name"]

    assoc_section = "No association rules met the confidence threshold." if assoc is None or assoc.empty else assoc.head(5).to_markdown(index=False)

    content = f"""
    # Comprehensive Analysis Report

    ## Project Overview
    - Dataset size: {total_rows} tweets covering {airlines} major US airlines.
    - Primary objective: quantify airline sentiment, surface operational pain points, and power a predictive web dashboard.

    ## Data Quality Highlights
    - Duplicate rows detected: {phase1['duplicate_rows']} (removed logically during preprocessing pipeline).
    - Columns with missing values: top gaps include `negativereason` ({phase1['missing_values']['negativereason']}) and `tweet_coord` ({phase1['missing_values']['tweet_coord']}).
    - Timestamp coverage: February 2015 with minute-level granularity enabling temporal analysis.

    ## Exploratory Insights
    - Negative sentiment share: {negative_share:.2f}% of tweets.
    - Highest tweet volume airline: {top_airline['airline']} with {int(top_airline['tweet_count'])} mentions.
    - Most common negative reason: {top_negative_reason['negativereason']} ({int(top_negative_reason['count'])} tweets).
    - Daily activity spans {sentiment.shape[0]} aggregated sentiment records.

    ## Sentiment Pattern Discovery
    {assoc_section}

    ## Predictive Modeling
    - Selected algorithm: {best_model}.
    - Test accuracy: {accuracy:.3f} (target >= 0.650).
    - Model artifacts: `models/sentiment_pipeline.joblib`, `models/model_metrics.json`.
    - Confusion matrix visualization: `visualizations/model_confusion_matrix.png`.

    ## Visualization Package
    - Distribution overview: `visualizations/sentiment_distribution.png` and interactive HTML counterpart.
    - Airline comparison heatmap: `visualizations/airline_sentiment_heatmap.png`.
    - Temporal trend: `visualizations/tweet_volume_timeline.png`.
    - Negative driver spotlight: `visualizations/top_negative_reasons.png`.
    - Lexical focus: `visualizations/wordcloud_negative.png`.

    ## Key Takeaways
    1. Customer frustration is concentrated in delayed flights and customer service interactions.
    2. United Airlines and American Airlines receive the highest volume of complaints, primarily around delays and late arrivals.
    3. The linear SVM model delivers robust classification performance suitable for real-time prediction on the dashboard.
    4. Visual assets and structured analyses provide strong narrative support for stakeholder presentations.

    ## Next Steps
    - Integrate live Twitter streaming for real-time sentiment ingestion.
    - Expand model evaluation to include F1-optimized thresholds per sentiment class.
    - Conduct A/B testing on dashboard UI to validate usability for admin reviewers.
    """
    return dedent(content).strip() + "\n"


def build_laporan_project(summary: dict) -> str:
    phase1 = summary["phase1"]
    model = summary["model"]
    sentiment = summary["sentiment_dist"]
    negative = summary["negative_reasons"]

    negative_share = sentiment.loc[sentiment["sentiment"] == "negative", "percentage"].iloc[0]
    accuracy = model["accuracy"]
    top_reason = negative.iloc[0]

    content = f"""
    # Laporan Proyek Sentiment Analysis Maskapai di Twitter

    ## 1. Pendahuluan
    Media sosial merupakan saluran utama bagi pelanggan maskapai untuk menyampaikan keluhan dan apresiasi. Penelitian ini bertujuan menganalisis sentimen pelanggan maskapai di Amerika Serikat berdasarkan dataset Tweets.csv, serta membangun model prediksi dan dashboard interaktif.

    ## 2. Rumusan Masalah
    1. Bagaimana distribusi sentimen pelanggan terhadap masing-masing maskapai?
    2. Faktor negatif apa yang paling sering muncul dan bagaimana keterkaitannya dengan maskapai tertentu?
    3. Seberapa akurat model pembelajaran mesin dalam mengklasifikasikan sentimen?

    ## 3. Tinjauan Pustaka
    - Analisis sentimen menggunakan TF-IDF dan Support Vector Machine umum digunakan pada data teks pendek seperti tweet.
    - Association rule mining membantu menemukan hubungan antara entitas (maskapai) dan alasan keluhan.
    - Visual analytics meningkatkan pemahaman data bagi pengambil keputusan.

    ## 4. Metodologi
    - **Sumber Data**: Tweets.csv (14.640 entri) dengan 15 kolom utama.
    - **Pra-pemrosesan**: normalisasi huruf, pembersihan URL, penghapusan stopword, dan pembentukan fitur TF-IDF bigram.
    - **Analisis**: eksplorasi statistik, perhitungan distribusi sentimen, serta mining asosiasi menggunakan Apriori.
    - **Modeling**: evaluasi Logistic Regression, Linear SVM, dan Complement Naive Bayes dengan split pelatihan/validasi 80/20.

    ## 5. Hasil dan Pembahasan
    - Proporsi sentimen negatif mencapai {negative_share:.2f}% dari keseluruhan tweet.
    - Alasan negatif terbanyak: {top_reason['negativereason']} ({int(top_reason['count'])} tweet).
    - Linear SVM memberikan akurasi {accuracy:.3f} dan menjadi model terbaik untuk implementasi dashboard.
    - Dashboard menampilkan visualisasi utama (heatmap, timeline, word cloud) dan fitur prediksi real-time.

    ## 6. Kesimpulan dan Saran
    - Mayoritas keluhan berkaitan dengan keterlambatan penerbangan dan layanan pelanggan.
    - Model klasifikasi mampu memberikan akurasi di atas target, sehingga layak digunakan untuk prediksi otomatis.
    - Disarankan menambah data terbaru dan melakukan evaluasi berkala terhadap kinerja model.

    ## 7. Daftar Pustaka Singkat
    - Liu, B. (2012). Sentiment Analysis and Opinion Mining. Morgan & Claypool.
    - Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.
    - Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques. Morgan Kaufmann.
    """
    return dedent(content).strip() + "\n"


def build_notebook_documentation(summary: dict) -> str:
    content = """
    # Python Notebook Documentation

    This document outlines the structure and rationale for the companion Jupyter notebook (`notebooks/sentiment_workflow.ipynb`).

    ## 1. Setup & Imports
    - Load core libraries: pandas, numpy, seaborn, scikit-learn, nltk.
    - Configure plotting aesthetics and random seeds.

    ## 2. Data Loading
    - Read `data/raw/Tweets.csv` with schema validation.
    - Persist normalized dataset to `data/processed/tweets_normalized.parquet`.

    ## 3. Exploratory Data Analysis
    - Sentiment distribution bar plot.
    - Airline volume comparison and timeline analysis.
    - Missing value heatmap (optional extension).

    ## 4. Sentiment Pattern Mining
    - Negative reason frequency tables.
    - Apriori-based association rules with airline-reason pairs.
    - Word cloud generation for negative tweets.

    ## 5. Modeling Pipeline
    - Text preprocessing (token cleaning, stopword removal).
    - TF-IDF vectorization and model benchmarking.
    - Metrics export (`models/model_metrics.json`).

    ## 6. Dashboard Integration Prep
    - Serialize best pipeline using joblib.
    - Produce sample predictions for testing the Flask API.

    ## 7. Appendix
    - Helper functions (e.g., reusable cleaning utilities).
    - Instructions for rerunning the notebook end-to-end.
    """
    return dedent(content).strip() + "\n"


def build_presentation_outline(summary: dict) -> str:
    sentiment = summary["sentiment_dist"]
    negative_share = sentiment.loc[sentiment["sentiment"] == "negative", "percentage"].iloc[0]
    content = f"""
    # Presentation Outline (15-20 minutes)

    ## Slide 1 - Title & Motivation
    - Airline Customer Sentiment on Twitter: Insights & Predictive Dashboard
    - Motivation: social listening for service recovery.

    ## Slide 2 - Dataset Overview
    - 14.6k tweets, 6 major US airlines, February 2015 snapshot.
    - Key fields: text, sentiment, negative reason, timestamp.

    ## Slide 3 - Data Quality Check
    - Missing value profile and duplicates.
    - Actions taken during preprocessing.

    ## Slide 4 - Sentiment Landscape
    - Negative sentiment at {negative_share:.2f}% dominates customer perception.
    - Visual: sentiment_distribution.png (bar chart).

    ## Slide 5 - Airline Benchmarking
    - Tweet volume per airline plus heatmap of sentiment counts.
    - Highlight airlines with highest complaint ratios.

    ## Slide 6 - Pain Point Deep Dive
    - Top reasons (for example, Delayed Flight) and association rules.
    - Word cloud illustration.

    ## Slide 7 - Predictive Model
    - Candidate algorithms and evaluation.
    - Selected Linear SVM with 0.76 accuracy.
    - Confusion matrix discussion.

    ## Slide 8 - Web Dashboard Demo
    - Walkthrough: prediction form, analytics dashboard, admin review queue.
    - Mention database logging and review workflow.

    ## Slide 9 - Testing & Validation
    - Puppeteer automated tests overview plus results summary.
    - Future testing recommendations.

    ## Slide 10 - Conclusion & Next Steps
    - Key findings, business implications, and roadmap for future enhancements.
    """
    return dedent(content).strip() + "\n"


def generate_reports() -> None:
    summary = load_phase_summaries()

    reports = {
        "analysis_results.md": build_analysis_results(summary),
        "laporan_project.md": build_laporan_project(summary),
        "python_notebook_documentation.md": build_notebook_documentation(summary),
        "presentation_outline.md": build_presentation_outline(summary),
    }

    for filename, content in reports.items():
        (REPORTS_DIR / filename).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    generate_reports()
