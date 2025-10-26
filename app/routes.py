"""Flask routes and view logic."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import csv
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from io import StringIO
from typing import Any, Dict, List

import pandas as pd
from flask import (
    Blueprint,
    Flask,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

from app.extensions import limiter
from app.forms import AdminLoginForm, PredictionForm
from src.data import database
from src.models.training import (
    explain_prediction,
    load_trained_model,
)
from tasks.retraining import perform_retraining, retrain_model_task

LOGGER = logging.getLogger(__name__)
try:
    from kombu.exceptions import OperationalError as KombuOperationalError
except Exception:  # pragma: no cover - kombu is provided with Celery
    KombuOperationalError = None  # type: ignore[assignment]

try:
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - redis extra may be missing
    RedisError = None  # type: ignore[assignment]

BROKER_EXCEPTION_TYPES = tuple(
    exc for exc in (KombuOperationalError, RedisError) if exc is not None
)
BROKER_ERROR_MARKERS = (
    "connection refused",
    "cannot connect",
    "connection error",
    "timed out",
    "error 111",
    "broken pipe",
    "no module named 'redis'",
    "broker unavailable",
)


def _is_broker_unavailable(exc: Exception) -> bool:
    """Return True when Celery cannot reach its message broker."""

    if BROKER_EXCEPTION_TYPES and isinstance(exc, BROKER_EXCEPTION_TYPES):
        return True

    message = str(exc).lower()
    return any(marker in message for marker in BROKER_ERROR_MARKERS)

bp = Blueprint("main", __name__)

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
VISUALIZATIONS = PROJECT_ROOT / "visualizations"
STATIC_IMG = APP_ROOT / "static" / "img"
STATIC_HTML = APP_ROOT / "static" / "html"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PIPELINE = None
MODEL_LABELS: List[str] | None = None
MODEL_METADATA: Dict[str, Any] | None = None
AIRLINES: List[str] = []
SENTIMENT_OVERVIEW: Dict[str, Any] | None = None


def ensure_static_assets() -> None:
    STATIC_IMG.mkdir(parents=True, exist_ok=True)
    STATIC_HTML.mkdir(parents=True, exist_ok=True)

    assets = {
        "sentiment_distribution.png": STATIC_IMG / "sentiment_distribution.png",
        "airline_sentiment_heatmap.png": STATIC_IMG / "airline_sentiment_heatmap.png",
        "tweet_volume_timeline.png": STATIC_IMG / "tweet_volume_timeline.png",
        "top_negative_reasons.png": STATIC_IMG / "top_negative_reasons.png",
        "wordcloud_negative.png": STATIC_IMG / "wordcloud_negative.png",
        "model_confusion_matrix.png": STATIC_IMG / "model_confusion_matrix.png",
    }

    for filename, destination in assets.items():
        source = VISUALIZATIONS / filename
        if source.exists():
            shutil.copyfile(source, destination)

    html_assets = {"sentiment_distribution.html": STATIC_HTML / "sentiment_distribution.html"}
    for filename, destination in html_assets.items():
        source = VISUALIZATIONS / filename
        if source.exists():
            shutil.copyfile(source, destination)


def _load_metrics_file(metrics_file: Path) -> Dict[str, Any]:
    if not metrics_file.exists():
        return {}
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse metrics file %s: %s", metrics_file, exc)
        return {}


@lru_cache(maxsize=1000)
def _cached_prediction(version_key: str, text_hash: str, raw_text: str) -> Dict[str, Any]:
    return explain_prediction(MODEL_PIPELINE, raw_text)


def clear_prediction_cache() -> None:
    _cached_prediction.cache_clear()


def load_model() -> None:
    global MODEL_PIPELINE, MODEL_LABELS, MODEL_METADATA

    latest_version = database.get_latest_model_version()
    metrics: Dict[str, Any] = {}

    if latest_version:
        model_path = Path(latest_version["model_path"])
        metrics_path_value = latest_version.get("metrics_path")
        metrics_path = Path(metrics_path_value) if metrics_path_value else None

        if model_path.exists():
            MODEL_PIPELINE = load_trained_model(model_path)
            if metrics_path:
                metrics = _load_metrics_file(metrics_path)
        else:
            LOGGER.warning("Versioned model %s missing. Falling back to default pipeline.", model_path)
            MODEL_PIPELINE = load_trained_model()
    else:
        MODEL_PIPELINE = load_trained_model()

    if MODEL_PIPELINE is None:
        raise RuntimeError("Unable to load sentiment model")

    MODEL_LABELS = MODEL_PIPELINE.classes_.tolist()

    base_metrics_file = MODELS_DIR / "model_metrics.json"
    if not metrics:
        metrics = _load_metrics_file(base_metrics_file)

    MODEL_METADATA = dict(metrics) if metrics else {}
    if latest_version:
        MODEL_METADATA.update(
            {
                "version_number": latest_version.get("version_number"),
                "accuracy": float(latest_version.get("accuracy") or metrics.get("accuracy", 0)),
                "training_samples": latest_version.get("training_samples"),
                "created_at": latest_version.get("created_at"),
                "is_active": bool(latest_version.get("is_active")),
                "model_path": latest_version.get("model_path"),
                "metrics_path": latest_version.get("metrics_path"),
            }
        )

    clear_prediction_cache()


def load_airlines() -> None:
    global AIRLINES

    path = DATA_PROCESSED / "airline_tweet_volume.csv"
    if path.exists():
        df = pd.read_csv(path)
        AIRLINES = sorted(df["airline"].dropna().unique().tolist())
    else:
        AIRLINES = []


def load_sentiment_overview() -> None:
    global SENTIMENT_OVERVIEW

    path = DATA_PROCESSED / "sentiment_distribution.csv"
    if not path.exists():
        SENTIMENT_OVERVIEW = None
        return

    df = pd.read_csv(path)
    summary = {}
    for row in df.to_dict(orient="records"):
        summary[row["sentiment"]] = {
            "count": int(row["count"]),
            "percentage": float(row.get("percentage", 0.0)),
        }

    SENTIMENT_OVERVIEW = {
        "total": int(df["count"].sum()),
        "summary": summary,
        "negative_pct": summary.get("negative", {}).get("percentage"),
        "neutral_pct": summary.get("neutral", {}).get("percentage"),
        "positive_pct": summary.get("positive", {}).get("percentage"),
    }


def load_dashboard_context() -> Dict[str, Any]:
    processed_dir = DATA_PROCESSED

    context: Dict[str, Any] = {
        "sentiment_summary": [],
        "airline_volume": [],
        "negative_reasons": [],
        "model_accuracy": 0,
        "model_name": "Unknown",
        "visuals": {},
    }

    sentiment_file = processed_dir / "sentiment_distribution.csv"
    if sentiment_file.exists():
        df = pd.read_csv(sentiment_file)
        context["sentiment_summary"] = df.to_dict("records")

    airline_file = processed_dir / "airline_tweet_volume.csv"
    if airline_file.exists():
        df = pd.read_csv(airline_file)
        context["airline_volume"] = df.to_dict("records")

    reasons_file = processed_dir / "top_negative_reasons.csv"
    if reasons_file.exists():
        df = pd.read_csv(reasons_file)
        context["negative_reasons"] = df.to_dict("records")

    metrics_file = MODELS_DIR / "model_metrics.json"
    if metrics_file.exists():
        metrics = _load_metrics_file(metrics_file)
        context["model_accuracy"] = metrics.get("accuracy", 0)
        context["model_name"] = metrics.get("model_name", "Unknown")

    context["visuals"] = {
        "sentiment": url_for("static", filename="img/sentiment_distribution.png"),
        "timeline": url_for("static", filename="img/tweet_volume_timeline.png"),
        "negative": url_for("static", filename="img/top_negative_reasons.png"),
        "heatmap": url_for("static", filename="img/airline_sentiment_heatmap.png"),
        "wordcloud": url_for("static", filename="img/wordcloud_negative.png"),
        "model_cm": url_for("static", filename="img/model_confusion_matrix.png"),
        "sentiment_html": url_for("static", filename="html/sentiment_distribution.html"),
    }

    timeline_file = processed_dir / "timeline_daily.csv"
    if timeline_file.exists():
        df = pd.read_csv(timeline_file)
        context["timeline_data"] = df.to_dict("records")


    context["top_airline"] = context["airline_volume"][0] if context["airline_volume"] else None
    context["top_negative_reason"] = context["negative_reasons"][0] if context["negative_reasons"] else None
    negative_summary = next((row for row in context["sentiment_summary"] if row.get("sentiment") == "negative"), None)
    context["negative_share"] = negative_summary.get("percentage") if negative_summary else None

    return context


def _predict_with_cache(text: str) -> Dict[str, Any]:
    version_key = str((MODEL_METADATA or {}).get("version_number") or "baseline")
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return _cached_prediction(version_key, text_hash, text)


@bp.route("/", methods=["GET", "POST"])
@limiter.limit("10 per minute")

def index() -> str:
    form = PredictionForm()
    form.set_airline_choices(AIRLINES)
    prediction_result = None
    batch_results = None
    recent = database.fetch_recent_submissions(limit=5)

    if request.method == "POST" and MODEL_PIPELINE is None:
        load_model()

    if form.validate_on_submit():
        uploaded_file = (
            form.batch_file.data
            if form.batch_file.data and getattr(form.batch_file.data, "filename", "")
            else None
        )
        text = (form.tweet_text.data or "").strip()
        assigned_airline = form.airline.data or None

        if uploaded_file:
            try:
                uploaded_file.stream.seek(0)
                raw_bytes = uploaded_file.stream.read()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to read uploaded CSV")
                flash("Could not read the uploaded file.", "danger")
                return render_template(
                    "index.html",
                    form=form,
                    prediction=prediction_result,
                    batch_results=batch_results,
                    recent_submissions=recent,
                )

            try:
                decoded = raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                flash("CSV file must be UTF-8 encoded.", "danger")
                return render_template(
                    "index.html",
                    form=form,
                    prediction=prediction_result,
                    batch_results=batch_results,
                    recent_submissions=recent,
                )

            reader = csv.DictReader(StringIO(decoded))
            if not reader.fieldnames:
                flash("CSV file is empty.", "danger")
                return render_template(
                    "index.html",
                    form=form,
                    prediction=prediction_result,
                    batch_results=batch_results,
                    recent_submissions=recent,
                )

            field_lookup = {name.lower(): name for name in reader.fieldnames if name}
            tweet_col = field_lookup.get("tweet_text")
            airline_col = field_lookup.get("airline")

            if not tweet_col:
                flash("CSV header must include a 'tweet_text' column.", "danger")
                return render_template(
                    "index.html",
                    form=form,
                    prediction=prediction_result,
                    batch_results=batch_results,
                    recent_submissions=recent,
                )

            max_rows = 200
            batch_results = []
            skipped_rows: list[str] = []
            truncated = False

            for row_index, row in enumerate(reader, start=1):
                if row_index > max_rows:
                    truncated = True
                    break

                row_text = (row.get(tweet_col) or "").strip()
                if not row_text:
                    skipped_rows.append(f"Row {row_index}: Missing tweet_text.")
                    continue
                if len(row_text) < 10 or len(row_text) > 500:
                    skipped_rows.append(
                        f"Row {row_index}: Tweet must be between 10 and 500 characters."
                    )
                    continue

                explanation = _predict_with_cache(row_text)
                row_airline = (row.get(airline_col) or "").strip() if airline_col else None
                row_airline = row_airline or assigned_airline

                database.insert_submission(
                    tweet_text=row_text,
                    predicted_sentiment=explanation["sentiment"],
                    prediction_confidence=explanation["confidence"],
                    assigned_airline=row_airline,
                )

                batch_results.append(
                    {
                        "row_index": row_index,
                        "text": row_text,
                        "sentiment": explanation["sentiment"],
                        "confidence": explanation["confidence"],
                        "reasoning": explanation["reasoning"],
                        "top_features": explanation["top_features"],
                        "airline": row_airline,
                    }
                )

            if not batch_results:
                flash("No valid rows found in CSV.", "warning")
            else:
                flash(f"Processed {len(batch_results)} tweets from CSV.", "success")
                if truncated:
                    flash("Only the first 200 rows were processed.", "info")
                if skipped_rows:
                    preview = "; ".join(skipped_rows[:3])
                    if len(skipped_rows) > 3:
                        preview = f"{preview} (and {len(skipped_rows) - 3} more)"
                    flash(f"Skipped rows: {preview}", "warning")
                recent = database.fetch_recent_submissions(limit=5)
            return render_template(
                "index.html",
                form=form,
                prediction=prediction_result,
                batch_results=batch_results,
                recent_submissions=recent,
            )

        if not text:
            flash("Tweet text cannot be empty.", "danger")
            return render_template(
                "index.html",
                form=form,
                prediction=prediction_result,
                batch_results=batch_results,
                recent_submissions=recent,
            )

        explanation = _predict_with_cache(text)

        database.insert_submission(
            tweet_text=text,
            predicted_sentiment=explanation["sentiment"],
            prediction_confidence=explanation["confidence"],
            assigned_airline=assigned_airline,
        )

        prediction_result = {
            "sentiment": explanation["sentiment"],
            "confidence": explanation["confidence"],
            "reasoning": explanation["reasoning"],
            "top_features": explanation["top_features"],
            "airline": assigned_airline,
        }
        flash("Sentiment prediction saved for review.", "success")
        recent = database.fetch_recent_submissions(limit=5)

    return render_template(
        "index.html",
        form=form,
        prediction=prediction_result,
        batch_results=batch_results,
        recent_submissions=recent,
    )


@bp.route("/dashboard")
def dashboard() -> str:
    context = load_dashboard_context()
    approved_count = database.count_approved_with_true_sentiment()
    context["approved_training_count"] = approved_count
    context["current_model"] = MODEL_METADATA
    context["all_versions"] = database.get_all_model_versions()
    return render_template("dashboard.html", **context)


@bp.route("/api/live-metrics")
def live_metrics() -> Any:
    accuracy = float((MODEL_METADATA or {}).get("accuracy", 0) or 0)
    pending = database.count_pending_submissions()
    return jsonify({
        "accuracy": accuracy,
        "pending_count": pending,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@bp.route("/admin", methods=["GET", "POST"])
def admin() -> str:
    if not session.get("is_admin"):
        form = AdminLoginForm()
        if form.validate_on_submit():
            username = (form.username.data or "").strip()
            password = form.password.data or ""
            expected_username = current_app.config["ADMIN_USERNAME"]
            password_hash = current_app.config["ADMIN_PASSWORD_HASH"]

            if username.lower() == expected_username.lower() and check_password_hash(password_hash, password):
                session["is_admin"] = True
                session["admin_username"] = expected_username
                flash("Welcome back, admin.", "success")
                return redirect(url_for("main.admin"))

            flash("Invalid admin credentials.", "danger")
        return render_template("admin_login.html", form=form)

    submissions = database.fetch_all_submissions_grouped(limit=500)
    return render_template("admin.html", submissions=submissions)


@bp.route("/admin/review/<int:submission_id>", methods=["POST"])
def review_submission(submission_id: int):
    if not session.get("is_admin"):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("main.admin"))

    submission = database.get_submission_by_id(submission_id)
    action = request.form.get("action")
    true_sentiment = (request.form.get("true_sentiment") or "").strip() or None
    admin_comment = (request.form.get("admin_comment") or "").strip() or None

    if action not in {"approve", "reject"}:
        flash("Invalid action.", "danger")
        return redirect(url_for("main.admin"))

    status = "approved" if action == "approve" else "rejected"

    database.update_submission_status(
        submission_id=submission_id,
        status=status,
        true_sentiment=true_sentiment,
        admin_comment=admin_comment,
    )

    if status == "approved" and submission and true_sentiment:
        predicted = submission.get("predicted_sentiment")
        model_version = (MODEL_METADATA or {}).get("version_number")
        if predicted and model_version:
            was_correct = predicted == true_sentiment
            database.log_prediction_performance(
                submission_id=submission_id,
                model_version=int(model_version),
                was_correct=was_correct,
            )

    flash("Submission updated.", "success")
    return redirect(url_for("main.admin"))


@bp.route("/admin/retrain", methods=["POST"])
def retrain_model():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        new_data = database.fetch_training_data()
        if len(new_data) < 50:
            return (
                jsonify(
                    {
                        "error": (
                            "Insufficient training data. Need at least 50 approved reviews "
                            f"with true sentiment set. Currently have {len(new_data)}."
                        )
                    }
                ),
                400,
            )

        serialisable_data = [
            {
                "tweet_text": row.get("tweet_text"),
                "sentiment": row.get("sentiment"),
            }
            for row in new_data
        ]

        old_accuracy = float((MODEL_METADATA or {}).get("accuracy", 0) or 0)

        try:
            task = retrain_model_task.delay(serialisable_data, old_accuracy)
        except Exception as exc:  # noqa: BLE001
            if _is_broker_unavailable(exc):
                LOGGER.warning(
                    "Celery broker unavailable; running retraining synchronously: %s",
                    exc,
                )
                try:
                    payload = perform_retraining(serialisable_data, old_accuracy)
                except Exception as inline_exc:  # noqa: BLE001
                    LOGGER.exception("Synchronous retraining failed")
                    return jsonify({"error": str(inline_exc)}), 500
                return jsonify(payload)
            raise

        return jsonify({"task_id": task.id})
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to enqueue retraining task")
        return jsonify({"error": str(exc)}), 500


@bp.route("/admin/retrain/status/<task_id>")
def retrain_status(task_id: str):
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 403

    task = retrain_model_task.AsyncResult(task_id)
    if not task.ready():
        return jsonify({"status": "pending"})

    if task.failed():
        return jsonify({"status": "failed", "error": str(task.result)}), 500

    result = task.result or {}
    return jsonify({"status": "complete", "result": result})


@bp.route("/admin/model/<int:version>/activate", methods=["POST"])
def activate_model(version: int):
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        database.set_active_model(version)
        load_model()
        flash(f"Model version {version} is now active.", "success")
        return jsonify({"success": True, "message": f"Model version {version} activated"})
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to activate model version %s", version)
        return jsonify({"error": str(exc)}), 500


@bp.route("/logout")
def logout():
    session.pop("is_admin", None)
    session.pop("admin_username", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("main.index"))


def init_app(app: Flask) -> None:
    ensure_static_assets()
    load_model()
    load_airlines()
    load_sentiment_overview()
    app.register_blueprint(bp)


ensure_static_assets()
load_model()
load_airlines()
load_sentiment_overview()

