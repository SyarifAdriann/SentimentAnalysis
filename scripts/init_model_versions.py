"""Initialise database tables and migrate existing model into version history."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.database import (
    get_connection,
    initialize_database,
    save_model_version,
)


def migrate_current_model() -> None:
    """Ensure database schema exists and migrate current model as version 1."""

    print("Initialising database schema...")
    initialize_database()

    metrics_file = Path("models/model_metrics.json")
    if not metrics_file.exists():
        print("No existing model found. Run initial training before migration.")
        return

    with metrics_file.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT 1 FROM model_versions WHERE version_number = 1 LIMIT 1")
    existing = cursor.fetchone()
    cursor.close()
    conn.close()

    if existing:
        print("Model version 1 already exists. Migration complete.")
        return

    classification_report = metrics.get("classification_report", {})
    estimated_training_samples = int(metrics.get("test_samples", 0) * 5)

    save_model_version(
        version_num=1,
        metrics=classification_report,
        model_path="models/sentiment_pipeline.joblib",
        metrics_path=str(metrics_file),
        training_samples=estimated_training_samples,
        notes="Initial model migrated from existing training run",
        is_active=True,
    )

    accuracy = metrics.get("accuracy", 0)
    print("Migrated existing model to version 1 (active)")
    print(f"  Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    migrate_current_model()
