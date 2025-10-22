"""Train SVM and Naive Bayes models and export dedicated artefacts."""

from __future__ import annotations

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.models import training  # noqa: E402
from src.visualization import charts  # noqa: E402

DATA_PROCESSED = ROOT / "data" / "processed"
VISUALIZATIONS = ROOT / "visualizations"
REPORTS = ROOT / "reports"

TARGET_MODELS = {
    "linear_svc": "Support Vector Machine (Linear SVC)",
    "complement_nb": "Complement Naive Bayes",
}


def run() -> None:
    dataset_path = DATA_PROCESSED / "tweets_normalized.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("Normalized dataset not found. Run phase1 before training models.")

    df = pd.read_parquet(dataset_path)
    VISUALIZATIONS.mkdir(parents=True, exist_ok=True)

    summary: dict[str, float] = {}

    for model_key, model_name in TARGET_MODELS.items():
        result = training.train_pipeline(model_key, df)
        summary[model_key] = result.accuracy

        model_dir = ROOT / "models" / model_key
        training.save_trained_model(
            result,
            directory=model_dir,
            filename="model.joblib",
        )
        training.save_metrics(
            result,
            metrics_path=model_dir / "metrics.json",
            report_path=REPORTS / f"model_performance_{model_key}.md",
            confusion_matrix_path=DATA_PROCESSED / f"confusion_matrix_{model_key}.csv",
        )

        charts.plot_confusion_matrix(
            result.confusion_matrix,
            result.labels,
            VISUALIZATIONS / f"confusion_matrix_{model_key}.png",
        )

        print(f"{model_name} accuracy: {result.accuracy:.3f}")

    summary_path = ROOT / "models" / "model_accuracy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
