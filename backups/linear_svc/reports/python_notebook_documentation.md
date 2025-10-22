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
