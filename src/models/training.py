"""Model training utilities for sentiment classification."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelResult:
    """Container for trained model artefacts."""

    name: str
    accuracy: float
    report: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    labels: Sequence[str]
    pipeline: Pipeline


NLP_RESOURCES = {"stopwords": "stopwords", "punkt": "punkt"}
_STOPWORDS_CACHE: Optional[set[str]] = None


def ensure_nltk_resources() -> None:
    for _, resource in NLP_RESOURCES.items():
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            LOGGER.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource)


def _get_stopwords_cache() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        ensure_nltk_resources()
        _STOPWORDS_CACHE = set(stopwords.words("english"))
    return _STOPWORDS_CACHE


def preprocess_text(text: str, cached_stopwords: Optional[set[str]] = None) -> str:
    if cached_stopwords is None:
        cached_stopwords = _get_stopwords_cache()
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in cached_stopwords]
    return " ".join(tokens)


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cached_stopwords = _get_stopwords_cache()

    df = df.copy()
    df = df[df["airline_sentiment"].notna() & df["text"].notna()]
    df["clean_text"] = df["text"].apply(lambda value: preprocess_text(str(value), cached_stopwords))
    df = df[df["clean_text"].str.len() > 0]

    X = df["clean_text"]
    y = df["airline_sentiment"].astype(str)
    return X, y


def build_pipelines() -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_features=6000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        multi_class="auto",
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000),
                ),
                (
                    "clf",
                    LinearSVC(class_weight="balanced"),
                ),
            ]
        ),
        "complement_nb": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000),
                ),
                (
                    "clf",
                    ComplementNB(),
                ),
            ]
        ),
    }


def evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelResult:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(y_test.unique())
    report_dict = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    return ModelResult(
        name=name,
        accuracy=accuracy,
        report=report_dict,
        confusion_matrix=matrix.tolist(),
        labels=labels,
        pipeline=pipeline,
    )


def train_pipeline(
    model_name: str,
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    pipelines = build_pipelines()
    if model_name not in pipelines:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(pipelines)}")

    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    LOGGER.info("Training model: %s", model_name)
    pipeline = pipelines[model_name]
    return evaluate_pipeline(model_name, pipeline, X_train, X_test, y_train, y_test)


def select_and_train_best(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    pipelines = build_pipelines()
    results: List[ModelResult] = []

    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    for name, pipeline in pipelines.items():
        LOGGER.info("Training model: %s", name)
        results.append(
            evaluate_pipeline(name, pipeline, X_train, X_test, y_train, y_test)
        )

    best = max(results, key=lambda item: item.accuracy)

    LOGGER.info("Best model '%s' achieved accuracy %.4f", best.name, best.accuracy)
    save_trained_model(best)
    save_metrics(best)

    return best


def save_trained_model(
    result: ModelResult,
    *,
    directory: Path | None = None,
    filename: str = "sentiment_pipeline.joblib",
) -> Path:
    target_dir = Path(directory) if directory else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    joblib.dump(result.pipeline, path)
    return path


def save_metrics(
    result: ModelResult,
    *,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
    confusion_matrix_path: Path | None = None,
) -> None:
    metrics_file = Path(metrics_path) if metrics_path else MODEL_DIR / "model_metrics.json"
    report_file = Path(report_path) if report_path else REPORTS_DIR / "model_performance.md"
    cm_file = (
        Path(confusion_matrix_path)
        if confusion_matrix_path
        else DATA_PROCESSED_DIR / "confusion_matrix.csv"
    )

    metadata = {
        "model_name": result.name,
        "accuracy": result.accuracy,
        "labels": list(result.labels),
        "classification_report": result.report,
        "confusion_matrix": result.confusion_matrix,
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    markdown_lines = ["# Model Performance", ""]
    markdown_lines.append(f"- Selected model: {result.name}")
    markdown_lines.append(f"- Accuracy: {result.accuracy:.4f}")
    markdown_lines.append("")

    header = "| Class | Precision | Recall | F1-score | Support |"
    markdown_lines.append(header)
    markdown_lines.append("| --- | --- | --- | --- | --- |")
    for label, metrics in result.report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0.0)
        support = metrics.get("support", 0)
        markdown_lines.append(
            f"| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {int(support)} |"
        )

    markdown_lines.append("")
    markdown_lines.append("## Aggregate Metrics")
    for key in ("macro avg", "weighted avg"):
        metrics = result.report.get(key)
        if not metrics:
            continue
        markdown_lines.append(
            f"- {key.title()}: precision {metrics['precision']:.3f}, "
            f"recall {metrics['recall']:.3f}, f1 {metrics['f1-score']:.3f}"
        )

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(markdown_lines), encoding="utf-8")

    cm_df = pd.DataFrame(result.confusion_matrix, columns=result.labels, index=result.labels)
    cm_file.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(cm_file)


def load_trained_model(path: Path | None = None) -> Pipeline:
    if path is None:
        path = MODEL_DIR / "sentiment_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}")
    return joblib.load(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def retrain_with_new_data(
    new_data_rows: List[Dict[str, Any]],
    model_name: str = "linear_svc",
) -> Tuple[Pipeline, Dict[str, Any]]:
    from src.data.loaders import load_normalized_dataset

    if not new_data_rows:
        raise ValueError("No new training data provided")

    original_df = load_normalized_dataset()

    new_df = pd.DataFrame(new_data_rows)
    if new_df.empty:
        raise ValueError("New training data is empty")

    new_df = new_df.rename(
        columns={"tweet_text": "text", "sentiment": "airline_sentiment"}
    )
    missing_columns = {"text", "airline_sentiment"} - set(new_df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns in new data: {missing_columns}")

    combined_df = pd.concat(
        [
            original_df[["text", "airline_sentiment"]],
            new_df[["text", "airline_sentiment"]],
        ],
        ignore_index=True,
    )
    combined_df = combined_df.dropna(subset=["text", "airline_sentiment"])

    X, y = prepare_dataset(combined_df)
    if len(X) == 0:
        raise ValueError("No valid training data after preprocessing")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipelines = build_pipelines()
    if model_name not in pipelines:
        raise ValueError(
            f"Model {model_name} not found. Choose from {list(pipelines.keys())}"
        )

    pipeline = pipelines[model_name]
    LOGGER.info(
        "Retraining %s with %d samples (including %d new)",
        model_name,
        len(X_train),
        len(new_data_rows),
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    results: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": pipeline.classes_.tolist(),
        "total_training_samples": int(len(X_train)),
        "new_samples_added": int(len(new_data_rows)),
        "test_samples": int(len(X_test)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    LOGGER.info("Retraining complete. New accuracy: %.4f", accuracy)

    return pipeline, results


def save_model_version(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    version_number: int,
) -> Tuple[str, str]:
    models_dir = MODEL_DIR
    version_dir = models_dir / f"version_{version_number}"
    if version_dir.exists():
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = version_dir / "metrics.json"
    metrics_clean = json.loads(json.dumps(metrics, default=_json_default))
    metrics_clean["saved_at"] = datetime.utcnow().isoformat() + "Z"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_clean, handle, indent=2)

    LOGGER.info("Saved model version %s to %s", version_number, version_dir)

    return str(model_path), str(metrics_path)


def get_feature_importance(
    pipeline: Pipeline,
    text: str,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    cleaned = preprocess_text(text)
    tfidf = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("clf")
    if tfidf is None or clf is None:
        return []

    vector = tfidf.transform([cleaned])

    if hasattr(clf, "coef_"):
        prediction = pipeline.predict([cleaned])[0]
        classes = list(pipeline.classes_)
        pred_idx = classes.index(prediction)
        coef = clf.coef_[pred_idx]
        feature_names = tfidf.get_feature_names_out()
        nonzero_indices = vector.nonzero()[1]

        contributions: List[Tuple[str, float]] = []
        for idx in nonzero_indices:
            weight = float(coef[idx] * vector[0, idx])
            contributions.append((feature_names[idx], weight))

        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        return contributions[:top_n]

    return []


def explain_prediction(pipeline: Pipeline, text: str) -> Dict[str, Any]:
    cleaned = preprocess_text(text)
    prediction = pipeline.predict([cleaned])[0]

    tfidf = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("clf")
    confidence: Optional[float] = None

    if tfidf is not None and clf is not None:
        vector = tfidf.transform([cleaned])
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(vector)[0]
            classes = list(pipeline.classes_)
            confidence = float(proba[classes.index(prediction)])
        elif hasattr(clf, "decision_function"):
            decision = clf.decision_function(vector)
            from scipy.special import expit

            decision_arr = np.atleast_2d(decision)
            probs = expit(decision_arr)
            classes = list(pipeline.classes_)
            pred_idx = classes.index(prediction)
            confidence = float(probs[0][pred_idx])

    top_features = get_feature_importance(pipeline, text, top_n=5)

    positive_features = [feature for feature, weight in top_features if weight > 0]
    negative_features = [feature for feature, weight in top_features if weight < 0]

    reasoning_parts: List[str] = []
    if positive_features:
        reasoning_parts.append(
            f"Key indicators: {', '.join(positive_features[:3])}"
        )
    if not reasoning_parts and top_features:
        reasoning_parts.append("Based on weighted feature contributions")

    explanation = {
        "sentiment": prediction,
        "confidence": confidence,
        "top_features": [feature for feature, _ in top_features],
        "feature_weights": {feature: weight for feature, weight in top_features},
        "reasoning": " | ".join(reasoning_parts)
        if reasoning_parts
        else "Classification based on learned patterns",
    }

    return explanation

def batch_explain_predictions(pipeline: Pipeline, texts: list[str]) -> list[Dict[str, Any]]:
    """Generate explanations for multiple texts efficiently."""
    if not texts:
        return []

    explanations: list[Dict[str, Any]] = []
    for text in texts:
        explanations.append(explain_prediction(pipeline, text))
    return explanations
