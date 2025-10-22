"""Celery tasks for background retraining."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from celery import shared_task

from src.data import database
from src.models.training import retrain_with_new_data, save_model_version as persist_model_version

LOGGER = logging.getLogger(__name__)


def perform_retraining(new_data: List[Dict[str, Any]], old_accuracy: float | None = None) -> Dict[str, Any]:
    """Execute the retraining workflow and persist a new model version."""

    serialisable_rows = [
        {"tweet_text": row.get("tweet_text"), "sentiment": row.get("sentiment")}
        for row in new_data
        if row.get("tweet_text") and row.get("sentiment")
    ]

    if not serialisable_rows:
        raise ValueError("No valid training rows supplied for retraining")

    pipeline, results = retrain_with_new_data(serialisable_rows, model_name="linear_svc")

    all_versions = database.get_all_model_versions()
    next_version = max((version["version_number"] for version in all_versions), default=0) + 1

    model_path, metrics_path = persist_model_version(pipeline, results, next_version)

    database.save_model_version(
        version_num=next_version,
        metrics=results["classification_report"],
        model_path=model_path,
        metrics_path=metrics_path,
        training_samples=results["total_training_samples"],
        notes=f"Retrained with {results['new_samples_added']} new samples",
        is_active=False,
    )

    new_accuracy = float(results.get("accuracy", 0))
    baseline_accuracy = float(old_accuracy or 0)
    accuracy_diff = new_accuracy - baseline_accuracy
    improvement_pct = (accuracy_diff / baseline_accuracy * 100) if baseline_accuracy > 0 else 0.0

    payload = {
        "success": True,
        "version": next_version,
        "old_accuracy": baseline_accuracy,
        "new_accuracy": new_accuracy,
        "accuracy_difference": accuracy_diff,
        "improvement_percentage": improvement_pct,
        "training_samples": int(results.get("total_training_samples", 0)),
        "new_samples": int(results.get("new_samples_added", 0)),
        "message": f"Model version {next_version} trained successfully",
    }

    LOGGER.info(
        "Retraining finished for version %s (accuracy %.4f, delta %.4f)",
        next_version,
        new_accuracy,
        accuracy_diff,
    )

    return payload


@shared_task(bind=True)
def retrain_model_task(self, new_data: List[Dict[str, Any]], old_accuracy: float | None = None) -> Dict[str, Any]:
    """Background task that retrains the sentiment model."""
    return perform_retraining(new_data, old_accuracy)
