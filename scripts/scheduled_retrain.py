"""Scheduled retraining script for the continuous learning system."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import schedule

from src.data import database
from src.models.training import retrain_with_new_data, save_model_version as persist_model_version

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

RETRAIN_THRESHOLD = 100
AUTO_ACTIVATE_THRESHOLD = 0.02


def _serialise_training_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {
            "tweet_text": row.get("tweet_text"),
            "sentiment": row.get("sentiment"),
        }
        for row in rows
        if row.get("tweet_text") and row.get("sentiment")
    ]


def run_retrain_cycle() -> None:
    available = database.count_approved_with_true_sentiment()
    if available < RETRAIN_THRESHOLD:
        LOGGER.info("Skipping retrain: only %s labelled reviews", available)
        return

    LOGGER.info("Starting scheduled retrain with %s labelled reviews", available)
    training_rows = database.fetch_training_data()
    serialised = _serialise_training_rows(training_rows)
    latest = database.get_latest_model_version()
    old_accuracy = float(latest.get("accuracy", 0)) if latest else 0.0

    pipeline, results = retrain_with_new_data(serialised, model_name="linear_svc")
    all_versions = database.get_all_model_versions()
    next_version = max((row["version_number"] for row in all_versions), default=0) + 1

    model_path, metrics_path = persist_model_version(pipeline, results, next_version)
    database.save_model_version(
        version_num=next_version,
        metrics=results["classification_report"],
        model_path=model_path,
        metrics_path=metrics_path,
        training_samples=results["total_training_samples"],
        notes=f"Scheduled retrain with {results['new_samples_added']} new samples",
        is_active=False,
    )

    new_accuracy = float(results.get("accuracy", 0))
    improvement = new_accuracy - old_accuracy
    LOGGER.info(
        "Retrain complete. Old accuracy: %.4f, New accuracy: %.4f (Δ %.4f)",
        old_accuracy,
        new_accuracy,
        improvement,
    )

    if improvement >= AUTO_ACTIVATE_THRESHOLD:
        database.set_active_model(next_version)
        LOGGER.info("Auto-activated model version %s due to %.2f%% improvement", next_version, improvement * 100)
    else:
        LOGGER.info("New model stored as version %s; awaiting manual activation", next_version)


def main() -> None:
    schedule.every().day.at("02:00").do(run_retrain_cycle)
    LOGGER.info("Scheduled retraining initialised. Press Ctrl+C to stop.")
    run_retrain_cycle()
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        LOGGER.info("Scheduler stopped by user")


if __name__ == "__main__":
    main()
