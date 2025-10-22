# Continuous Learning System Implementation Guide

## Overview
This guide implements a human-in-the-loop continuous learning system with:
- Complete review history display (unmarked shown first)
- Manual model retraining from dashboard
- Training progress indicators
- Model comparison & rollback capability
- Prediction explanations using top influential features

---

## Part 1: Database Schema Updates

### 1.1 Update `src/data/database.py`

Add new table for model versioning and update the submissions table:

```python
# Add this function after initialize_database()

def create_model_versions_table(cursor):
    """Create table to track model versions and metrics"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            version_number INT NOT NULL,
            accuracy DECIMAL(6,4) NOT NULL,
            precision_macro DECIMAL(6,4),
            recall_macro DECIMAL(6,4),
            f1_macro DECIMAL(6,4),
            training_samples INT NOT NULL,
            model_path VARCHAR(255) NOT NULL,
            metrics_path VARCHAR(255),
            is_active BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE KEY unique_version (version_number)
        )
    """)

def initialize_database() -> None:
    """Initialize database and required tables"""
    config = DatabaseConfig.from_env()
    
    # Create database if not exists
    conn = get_connection(include_database=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
    cursor.close()
    conn.close()
    
    # Create tables
    conn = get_connection()
    cursor = conn.cursor()
    
    # Original submissions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tweet_text TEXT NOT NULL,
            predicted_sentiment VARCHAR(20) NOT NULL,
            prediction_confidence DECIMAL(5,4),
            assigned_airline VARCHAR(50),
            true_sentiment VARCHAR(20),
            review_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
            admin_comment TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_review_status (review_status),
            INDEX idx_created_at (created_at)
        )
    """)
    
    # New model versions table
    create_model_versions_table(cursor)
    
    conn.commit()
    cursor.close()
    conn.close()
```

### 1.2 Add New Database Functions

```python
# Add these functions to src/data/database.py

def fetch_all_submissions_grouped(limit: int = 1000) -> dict:
    """Fetch submissions grouped by review status"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Pending submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'pending'
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    pending = cursor.fetchall()
    
    # Approved submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'approved'
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))
    approved = cursor.fetchall()
    
    # Rejected submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'rejected'
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))
    rejected = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {
        'pending': pending,
        'approved': approved,
        'rejected': rejected
    }

def count_approved_with_true_sentiment() -> int:
    """Count approved submissions with true_sentiment set"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM submissions 
        WHERE review_status = 'approved' 
        AND true_sentiment IS NOT NULL 
        AND true_sentiment != ''
    """)
    count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    return count

def fetch_training_data():
    """Fetch approved submissions with true_sentiment for retraining"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT tweet_text, true_sentiment as sentiment
        FROM submissions 
        WHERE review_status = 'approved' 
        AND true_sentiment IS NOT NULL 
        AND true_sentiment != ''
        AND tweet_text IS NOT NULL
        ORDER BY updated_at ASC
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_model_version(version_num: int, metrics: dict, model_path: str, 
                       training_samples: int, notes: str = None, is_active: bool = False):
    """Save model version metadata"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO model_versions 
        (version_number, accuracy, precision_macro, recall_macro, f1_macro, 
         training_samples, model_path, is_active, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        version_num,
        metrics.get('accuracy', 0),
        metrics.get('macro avg', {}).get('precision', 0),
        metrics.get('macro avg', {}).get('recall', 0),
        metrics.get('macro avg', {}).get('f1-score', 0),
        training_samples,
        model_path,
        is_active,
        notes
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_latest_model_version():
    """Get the latest active model version info"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM model_versions 
        WHERE is_active = TRUE
        ORDER BY version_number DESC 
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

def set_active_model(version_number: int):
    """Set a specific version as active, deactivating others"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Deactivate all models
    cursor.execute("UPDATE model_versions SET is_active = FALSE")
    
    # Activate selected model
    cursor.execute("""
        UPDATE model_versions 
        SET is_active = TRUE 
        WHERE version_number = %s
    """, (version_number,))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_all_model_versions():
    """Get all model versions ordered by version number"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM model_versions 
        ORDER BY version_number DESC
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results
```

---

## Part 2: Enhanced Training Module

### 2.1 Update `src/models/training.py`

Add functions for retraining and feature importance:

```python
# Add these imports at the top
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# Add this function after select_and_train_best()

def retrain_with_new_data(new_data_rows: list, model_name: str = "linear_svc") -> dict:
    """
    Retrain model with original dataset + new approved submissions
    
    Args:
        new_data_rows: List of dicts with 'tweet_text' and 'sentiment' keys
        model_name: Which model to retrain (linear_svc, logistic_regression, complement_nb)
    
    Returns:
        dict with training results and comparison metrics
    """
    from src.data.loaders import load_normalized_dataset
    
    # Load original dataset
    original_df = load_normalized_dataset()
    
    # Create dataframe from new data
    if not new_data_rows:
        raise ValueError("No new training data provided")
    
    new_df = pd.DataFrame(new_data_rows)
    new_df.columns = ['text', 'airline_sentiment']
    
    # Combine datasets
    combined_df = pd.concat([
        original_df[['text', 'airline_sentiment']], 
        new_df
    ], ignore_index=True)
    
    # Prepare dataset
    X, y = prepare_dataset(combined_df)
    
    if len(X) == 0:
        raise ValueError("No valid training data after preprocessing")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and train the specified model
    pipelines = build_pipelines()
    if model_name not in pipelines:
        raise ValueError(f"Model {model_name} not found. Choose from {list(pipelines.keys())}")
    
    pipeline = pipelines[model_name]
    
    logger.info(f"Retraining {model_name} with {len(X_train)} samples (including {len(new_data_rows)} new)")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'labels': pipeline.classes_.tolist(),
        'total_training_samples': len(X_train),
        'new_samples_added': len(new_data_rows),
        'test_samples': len(X_test)
    }
    
    logger.info(f"Retraining complete. New accuracy: {accuracy:.4f}")
    
    return pipeline, results

def save_model_version(pipeline, metrics: dict, version_number: int):
    """Save a versioned model"""
    models_dir = Path("models")
    version_dir = models_dir / f"version_{version_number}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = version_dir / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    
    # Save metrics
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved model version {version_number} to {version_dir}")
    
    return str(model_path), str(metrics_path)

def get_feature_importance(pipeline, text: str, top_n: int = 5) -> list:
    """
    Extract top N features that influenced the prediction
    
    Args:
        pipeline: Trained sklearn pipeline
        text: Input tweet text
        top_n: Number of top features to return
    
    Returns:
        List of tuples (feature_name, weight)
    """
    # Preprocess text
    cleaned = preprocess_text(text)
    
    # Get feature vector
    tfidf_vector = pipeline.named_steps['tfidf'].transform([cleaned])
    
    # Get classifier coefficients
    classifier = pipeline.named_steps['clf']
    
    # For Linear SVC
    if hasattr(classifier, 'coef_'):
        # Get prediction
        prediction = pipeline.predict([cleaned])[0]
        pred_idx = list(pipeline.classes_).index(prediction)
        
        # Get coefficients for predicted class
        coef = classifier.coef_[pred_idx]
        
        # Get feature names
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Get non-zero features from the vector
        nonzero_indices = tfidf_vector.nonzero()[1]
        
        # Calculate feature contributions
        contributions = []
        for idx in nonzero_indices:
            feature_name = feature_names[idx]
            feature_weight = coef[idx] * tfidf_vector[0, idx]
            contributions.append((feature_name, float(feature_weight)))
        
        # Sort by absolute weight
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:top_n]
    
    # For Naive Bayes or other models
    return []

def explain_prediction(pipeline, text: str) -> dict:
    """
    Generate human-readable explanation for prediction
    
    Returns:
        dict with sentiment, confidence, and reasoning
    """
    prediction = pipeline.predict([text])[0]
    
    # Get confidence if available
    confidence = None
    if hasattr(pipeline.named_steps['clf'], 'decision_function'):
        decision = pipeline.named_steps['clf'].decision_function([preprocess_text(text)])
        # Convert to pseudo-probability
        from scipy.special import expit
        probs = expit(decision)[0]
        pred_idx = list(pipeline.classes_).index(prediction)
        confidence = float(probs[pred_idx]) if len(probs.shape) > 0 else float(probs)
    
    # Get top features
    top_features = get_feature_importance(pipeline, text, top_n=5)
    
    # Build explanation
    if top_features:
        positive_features = [f for f, w in top_features if w > 0]
        negative_features = [f for f, w in top_features if w < 0]
        
        reasoning_parts = []
        if positive_features:
            reasoning_parts.append(f"Key indicators: {', '.join(positive_features[:3])}")
        
        explanation = {
            'sentiment': prediction,
            'confidence': confidence,
            'top_features': [f for f, w in top_features],
            'feature_weights': {f: w for f, w in top_features},
            'reasoning': ' | '.join(reasoning_parts) if reasoning_parts else "Based on overall text patterns"
        }
    else:
        explanation = {
            'sentiment': prediction,
            'confidence': confidence,
            'top_features': [],
            'feature_weights': {},
            'reasoning': "Classification based on learned patterns"
        }
    
    return explanation
```

---

## Part 3: Update Flask Routes

### 3.1 Update `app/routes.py`

Replace the entire file with enhanced version:

```python
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from app.forms import PredictionForm, AdminLoginForm
from src.data import database
from src.models.training import load_trained_model, explain_prediction, retrain_with_new_data, save_model_version
import os
from pathlib import Path
import json
import pandas as pd
import shutil
from datetime import datetime

bp = Blueprint('main', __name__)

# Global model variables
MODEL_PIPELINE = None
MODEL_LABELS = None
MODEL_METADATA = None

def load_model():
    """Load the trained model and metadata"""
    global MODEL_PIPELINE, MODEL_LABELS, MODEL_METADATA
    
    # Check for versioned model first
    latest_version = database.get_latest_model_version()
    
    if latest_version:
        model_path = Path(latest_version['model_path'])
        if model_path.exists():
            MODEL_PIPELINE = load_trained_model(str(model_path))
            MODEL_METADATA = latest_version
        else:
            # Fall back to default
            MODEL_PIPELINE = load_trained_model()
            MODEL_METADATA = None
    else:
        MODEL_PIPELINE = load_trained_model()
        MODEL_METADATA = None
    
    if MODEL_PIPELINE:
        MODEL_LABELS = MODEL_PIPELINE.classes_.tolist()
    
    # Load metrics
    metrics_file = Path("models/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            if MODEL_METADATA is None:
                MODEL_METADATA = metrics

# Airline options
AIRLINES = [
    ("", "-- Select Airline (Optional) --"),
    ("United", "United Airlines"),
    ("US Airways", "US Airways"),
    ("American", "American Airlines"),
    ("Southwest", "Southwest Airlines"),
    ("Delta", "Delta Airlines"),
    ("Virgin America", "Virgin America")
]

def ensure_static_assets():
    """Copy visualization assets to static directory"""
    static_img = Path("app/static/img")
    static_html = Path("app/static/html")
    static_img.mkdir(parents=True, exist_ok=True)
    static_html.mkdir(parents=True, exist_ok=True)
    
    vis_dir = Path("visualizations")
    if vis_dir.exists():
        for png_file in vis_dir.glob("*.png"):
            shutil.copy(png_file, static_img / png_file.name)
        for html_file in vis_dir.glob("*.html"):
            shutil.copy(html_file, static_html / html_file.name)

@bp.route("/", methods=["GET", "POST"])
def index():
    form = PredictionForm()
    form.set_airline_choices(AIRLINES)
    prediction_result = None
    recent = database.fetch_recent_submissions(limit=5)
    
    if form.validate_on_submit():
        text = form.tweet_text.data.strip()
        if not text:
            flash("Tweet text cannot be empty.", "danger")
            return render_template("index.html", form=form, prediction=prediction_result, recent_submissions=recent)
        
        # Get prediction with explanation
        explanation = explain_prediction(MODEL_PIPELINE, text)
        
        assigned_airline = form.airline.data or None
        
        # Insert to database
        database.insert_submission(
            tweet_text=text,
            predicted_sentiment=explanation['sentiment'],
            prediction_confidence=explanation['confidence'],
            assigned_airline=assigned_airline,
        )
        
        prediction_result = {
            'sentiment': explanation['sentiment'],
            'confidence': explanation['confidence'],
            'reasoning': explanation['reasoning'],
            'top_features': explanation['top_features'],
            'airline': assigned_airline
        }
        
        flash("Sentiment prediction saved for review.", "success")
        recent = database.fetch_recent_submissions(limit=5)
    
    return render_template("index.html", form=form, prediction=prediction_result, recent_submissions=recent)

@bp.route("/dashboard")
def dashboard():
    context = load_dashboard_context()
    
    # Add training data count
    approved_count = database.count_approved_with_true_sentiment()
    context['approved_training_count'] = approved_count
    
    # Add model version info
    context['current_model'] = MODEL_METADATA
    context['all_versions'] = database.get_all_model_versions()
    
    return render_template("dashboard.html", **context)

@bp.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST" and not session.get("is_admin"):
        form = AdminLoginForm()
        if form.validate_on_submit():
            from flask import current_app
            if form.password.data == current_app.config.get("ADMIN_PASSWORD"):
                session["is_admin"] = True
                flash("Login successful!", "success")
                return redirect(url_for("main.admin"))
            else:
                flash("Invalid password.", "danger")
        return render_template("admin_login.html", form=form)
    
    if not session.get("is_admin"):
        form = AdminLoginForm()
        return render_template("admin_login.html", form=form)
    
    # Fetch all submissions grouped by status
    submissions = database.fetch_all_submissions_grouped(limit=500)
    
    return render_template("admin.html", submissions=submissions)

@bp.route("/admin/review/<int:submission_id>", methods=["POST"])
def review_submission(submission_id: int):
    if not session.get("is_admin"):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("main.admin"))
    
    action = request.form.get("action")
    true_sentiment = request.form.get("true_sentiment")
    admin_comment = request.form.get("admin_comment")
    
    if action == "approve":
        status = "approved"
        flash(f"Submission {submission_id} approved.", "success")
    elif action == "reject":
        status = "rejected"
        flash(f"Submission {submission_id} rejected.", "info")
    else:
        flash("Invalid action.", "danger")
        return redirect(url_for("main.admin"))
    
    database.update_submission_status(
        submission_id=submission_id,
        status=status,
        true_sentiment=true_sentiment if true_sentiment else None,
        admin_comment=admin_comment if admin_comment else None
    )
    
    return redirect(url_for("main.admin"))

@bp.route("/admin/retrain", methods=["POST"])
def retrain_model():
    """Trigger model retraining with approved data"""
    if not session.get("is_admin"):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get training data
        new_data = database.fetch_training_data()
        
        if len(new_data) < 50:
            return jsonify({
                'error': f'Insufficient training data. Need at least 50 approved reviews with true_sentiment set. Currently have {len(new_data)}.'
            }), 400
        
        # Get current model accuracy
        old_accuracy = MODEL_METADATA.get('accuracy', 0) if MODEL_METADATA else 0
        
        # Retrain model
        new_pipeline, results = retrain_with_new_data(new_data, model_name="linear_svc")
        
        # Get next version number
        all_versions = database.get_all_model_versions()
        next_version = max([v['version_number'] for v in all_versions], default=0) + 1
        
        # Save as new version
        model_path, metrics_path = save_model_version(new_pipeline, results, next_version)
        
        # Save to database
        database.save_model_version(
            version_num=next_version,
            metrics=results['classification_report'],
            model_path=model_path,
            training_samples=results['total_training_samples'],
            notes=f"Retrained with {results['new_samples_added']} new samples",
            is_active=False  # Don't activate yet, let admin choose
        )
        
        new_accuracy = results['accuracy']
        accuracy_diff = new_accuracy - old_accuracy
        
        return jsonify({
            'success': True,
            'version': next_version,
            'old_accuracy': float(old_accuracy),
            'new_accuracy': float(new_accuracy),
            'accuracy_difference': float(accuracy_diff),
            'improvement_percentage': float((accuracy_diff / old_accuracy * 100) if old_accuracy > 0 else 0),
            'training_samples': results['total_training_samples'],
            'new_samples': results['new_samples_added'],
            'message': f'Model version {next_version} trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/admin/model/<int:version>/activate", methods=["POST"])
def activate_model(version: int):
    """Activate a specific model version"""
    if not session.get("is_admin"):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        database.set_active_model(version)
        
        # Reload model globally
        load_model()
        
        flash(f"Model version {version} is now active.", "success")
        return jsonify({'success': True, 'message': f'Model version {version} activated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/logout")
def logout():
    session.pop("is_admin", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("main.index"))

def load_dashboard_context() -> dict:
    """Load all data needed for dashboard"""
    processed_dir = Path("data/processed")
    
    context = {
        "sentiment_summary": [],
        "airline_volume": [],
        "negative_reasons": [],
        "model_accuracy": 0,
        "model_name": "Unknown",
        "visuals": {}
    }
    
    # Load sentiment distribution
    sentiment_file = processed_dir / "sentiment_distribution.csv"
    if sentiment_file.exists():
        df = pd.read_csv(sentiment_file)
        context["sentiment_summary"] = df.to_dict('records')
    
    # Load airline volume
    airline_file = processed_dir / "airline_tweet_volume.csv"
    if airline_file.exists():
        df = pd.read_csv(airline_file)
        context["airline_volume"] = df.to_dict('records')
    
    # Load negative reasons
    reasons_file = processed_dir / "top_negative_reasons.csv"
    if reasons_file.exists():
        df = pd.read_csv(reasons_file)
        context["negative_reasons"] = df.to_dict('records')
    
    # Load model metrics
    metrics_file = Path("models/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            context["model_accuracy"] = metrics.get("accuracy", 0)
            context["model_name"] = metrics.get("model_name", "Unknown")
    
    # Visual assets
    context["visuals"] = {
        "sentiment": "/static/img/sentiment_distribution.png",
        "timeline": "/static/img/tweet_volume_timeline.png",
        "negative": "/static/img/top_negative_reasons.png",
        "heatmap": "/static/img/airline_sentiment_heatmap.png",
        "wordcloud": "/static/img/wordcloud_negative.png",
        "model_cm": "/static/img/model_confusion_matrix.png",
        "sentiment_html": "/static/html/sentiment_distribution.html",
    }
    
    return context

# Initialize on import
ensure_static_assets()
load_model()
```

---

## Part 4: Update Templates

### 4.1 Update `app/templates/admin.html`

Replace with enhanced version showing all submissions:

```html
{% extends "base.html" %}

{% block title %}Admin Panel - Review Queue{% endblock %}

{% block content %}
<div class="admin-container">
    <div class="admin-header">
        <h1>Admin Review Panel</h1>
        <a href="{{ url_for('main.logout') }}" class="btn btn-secondary">Logout</a>
    </div>

    <!-- Training Section -->
    <div class="card training-section">
        <h2>Model Training</h2>
        <div class="training-stats">
            <p><strong>Approved Reviews Available for Training:</strong> 
                {{ submissions.approved|selectattr('true_sentiment')|list|length }}</p>
            <p class="help-text">Minimum 50 reviews with true sentiment required for retraining</p>
        </div>
        
        <button id="retrainBtn" class="btn btn-primary" onclick="startRetraining()">
            <span id="retrainText">Retrain Model</span>
            <span id="retrainSpinner" class="spinner" style="display:none;">⟳ Training...</span>
        </button>
        
        <div id="trainingResults" style="display:none; margin-top: 20px;">
            <!-- Results will be injected here -->
        </div>
    </div>

    <!-- Model Comparison Modal -->
    <div id="comparisonModal" class="modal" style="display:none;">
        <div class="modal-content">
            <h2>Model Training Complete</h2>
            <div id="comparisonDetails"></div>
            <div class="modal-actions">
                <button onclick="activateNewModel()" class="btn btn-primary">Use New Model</button>
                <button onclick="keepOldModel()" class="btn btn-secondary">Keep Current Model</button>
            </div>
        </div>
    </div>

    <!-- Submissions Tabs -->
    <div class="submissions-tabs">
        <button class="tab-btn active" onclick="showTab('pending')">
            Pending ({{ submissions.pending|length }})
        </button>
        <button class="tab-btn" onclick="showTab('approved')">
            Approved ({{ submissions.approved|length }})
        </button>
        <button class="tab-btn" onclick="showTab('rejected')">
            Rejected ({{ submissions.rejected|length }})
        </button>
    </div>

    <!-- Pending Submissions -->
    <div id="pending-tab" class="tab-content active">
        <h2>Pending Reviews</h2>
        {% if submissions.pending %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>Confidence</th>
                            <th>Airline</th>
                            <th>Date</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.pending %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>
                                {% if submission.prediction_confidence %}
                                    {{ "%.1f"|format(submission.prediction_confidence * 100) }}%
                                {% else %}
                                    N/A
                                {% endif %}
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td>{{ submission.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>
                                <form method="POST" action="{{ url_for('main.review_submission', submission_id=submission.id) }}" class="inline-form">
                                    <select name="true_sentiment" required>
                                        <option value="">True Sentiment</option>
                                        <option value="negative">Negative</option>
                                        <option value="neutral">Neutral</option>
                                        <option value="positive">Positive</option>
                                    </select>
                                    <input type="text" name="admin_comment" placeholder="Comment (optional)" class="comment-input">
                                    <button type="submit" name="action" value="approve" class="btn btn-sm btn-success">Approve</button>
                                    <button type="submit" name="action" value="reject" class="btn btn-sm btn-danger">Reject</button>
                                </form>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No pending submissions</p>
        {% endif %}
    </div>

    <!-- Approved Submissions -->
    <div id="approved-tab" class="tab-content">
        <h2>Approved Reviews</h2>
        {% if submissions.approved %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>True Sentiment</th>
                            <th>Match</th>
                            <th>Airline</th>
                            <th>Comment</th>
                            <th>Reviewed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.approved %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>
                                {% if submission.true_sentiment %}
                                    <span class="sentiment-badge sentiment-{{ submission.true_sentiment }}">
                                        {{ submission.true_sentiment }}
                                    </span>
                                {% else %}
                                    <span class="text-muted">Not set</span>
                                {% endif %}
                            </td>
                            <td>
                                {% if submission.true_sentiment %}
                                    {% if submission.predicted_sentiment == submission.true_sentiment %}
                                        <span class="badge-success">✓ Match</span>
                                    {% else %}
                                        <span class="badge-error">✗ Mismatch</span>
                                    {% endif %}
                                {% else %}
                                    -
                                {% endif %}
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td class="comment-cell">{{ submission.admin_comment or '-' }}</td>
                            <td>{{ submission.updated_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No approved submissions yet</p>
        {% endif %}
    </div>

    <!-- Rejected Submissions -->
    <div id="rejected-tab" class="tab-content">
        <h2>Rejected Reviews</h2>
        {% if submissions.rejected %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>Airline</th>
                            <th>Comment</th>
                            <th>Rejected</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.rejected %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td class="comment-cell">{{ submission.admin_comment or '-' }}</td>
                            <td>{{ submission.updated_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No rejected submissions</p>
        {% endif %}
    </div>
</div>

<script>
let newModelVersion = null;

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

async function startRetraining() {
    const btn = document.getElementById('retrainBtn');
    const text = document.getElementById('retrainText');
    const spinner = document.getElementById('retrainSpinner');
    const results = document.getElementById('trainingResults');
    
    // Disable button and show spinner
    btn.disabled = true;
    text.style.display = 'none';
    spinner.style.display = 'inline';
    results.style.display = 'none';
    
    try {
        const response = await fetch('/admin/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.error) {
            results.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
            results.style.display = 'block';
        } else {
            newModelVersion = data.version;
            showComparisonModal(data);
        }
    } catch (error) {
        results.innerHTML = `<div class="alert alert-error">Training failed: ${error.message}</div>`;
        results.style.display = 'block';
    } finally {
        // Re-enable button
        btn.disabled = false;
        text.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

function showComparisonModal(data) {
    const modal = document.getElementById('comparisonModal');
    const details = document.getElementById('comparisonDetails');
    
    const improvementText = data.accuracy_difference > 0 
        ? `<span style="color: green;">+${(data.improvement_percentage).toFixed(2)}% improvement</span>`
        : `<span style="color: red;">${(data.improvement_percentage).toFixed(2)}% decrease</span>`;
    
    details.innerHTML = `
        <div class="comparison-grid">
            <div class="comparison-item">
                <h3>Current Model</h3>
                <p class="metric-value">${(data.old_accuracy * 100).toFixed(2)}%</p>
                <p class="metric-label">Accuracy</p>
            </div>
            <div class="comparison-item">
                <h3>New Model (v${data.version})</h3>
                <p class="metric-value">${(data.new_accuracy * 100).toFixed(2)}%</p>
                <p class="metric-label">Accuracy</p>
            </div>
        </div>
        <div class="comparison-summary">
            <p><strong>Change:</strong> ${improvementText}</p>
            <p><strong>Training samples:</strong> ${data.training_samples} (${data.new_samples} new)</p>
            <p class="help-text">Choose whether to activate the new model or keep using the current one.</p>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function activateNewModel() {
    try {
        const response = await fetch(`/admin/model/${newModelVersion}/activate`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('New model activated! The page will reload.');
            location.reload();
        } else {
            alert('Failed to activate model: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function keepOldModel() {
    document.getElementById('comparisonModal').style.display = 'none';
    const results = document.getElementById('trainingResults');
    results.innerHTML = `<div class="alert alert-info">New model version ${newModelVersion} saved but not activated. Current model remains in use.</div>`;
    results.style.display = 'block';
}
</script>

<style>
.admin-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.admin-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.training-section {
    margin-bottom: 30px;
    padding: 20px;
}

.training-stats {
    margin-bottom: 15px;
}

.help-text {
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
}

.spinner {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.submissions-tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
}

.tab-btn {
    padding: 10px 20px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 16px;
    color: #666;
    border-bottom: 3px solid transparent;
    transition: all 0.3s;
}

.tab-btn:hover {
    color: #0a3d62;
}

.tab-btn.active {
    color: #0a3d62;
    border-bottom-color: #0a3d62;
    font-weight: bold;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.data-table-container {
    overflow-x: auto;
}

.tweet-cell {
    max-width: 300px;
    word-wrap: break-word;
}

.comment-cell {
    max-width: 200px;
    font-style: italic;
    color: #666;
}

.sentiment-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: bold;
}

.sentiment-negative {
    background-color: #fee;
    color: #c00;
}

.sentiment-neutral {
    background-color: #ffc;
    color: #660;
}

.sentiment-positive {
    background-color: #efe;
    color: #060;
}

.badge-success {
    color: green;
    font-weight: bold;
}

.badge-error {
    color: red;
    font-weight: bold;
}

.inline-form {
    display: flex;
    gap: 5px;
    align-items: center;
}

.comment-input {
    padding: 4px 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 0.9em;
}

.btn-sm {
    padding: 4px 12px;
    font-size: 0.85em;
}

.btn-success {
    background-color: #28a745;
    color: white;
}

.btn-danger {
    background-color: #dc3545;
    color: white;
}

.empty-state {
    text-align: center;
    padding: 40px;
    color: #999;
    font-style: italic;
}

.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal-content {
    background: white;
    padding: 30px;
    border-radius: 8px;
    max-width: 600px;
    width: 90%;
}

.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}

.comparison-item {
    text-align: center;
    padding: 20px;
    border: 2px solid #ddd;
    border-radius: 8px;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #0a3d62;
    margin: 10px 0;
}

.metric-label {
    color: #666;
    font-size: 0.9em;
}

.comparison-summary {
    margin-top: 20px;
    padding: 15px;
    background: #f5f5f5;
    border-radius: 4px;
}

.modal-actions {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 20px;
}

.alert {
    padding: 15px;
    border-radius: 4px;
    margin-top: 15px;
}

.alert-error {
    background-color: #fee;
    color: #c00;
    border: 1px solid #fcc;
}

.alert-info {
    background-color: #e7f3ff;
    color: #004085;
    border: 1px solid #b8daff;
}

.text-muted {
    color: #999;
}
</style>
{% endblock %}
---

## Part 5: Database Migration

### 5.1 Run Database Initialization

After updating the code, run this Python script to initialize the new tables:

```python
# scripts/init_model_versions.py

from src.data.database import initialize_database, save_model_version, get_connection
from pathlib import Path
import json

def migrate_current_model():
    """Migrate existing model to version 1"""
    print("Initializing database schema...")
    initialize_database()
    
    # Check if model metrics exist
    metrics_file = Path("models/model_metrics.json")
    if not metrics_file.exists():
        print("No existing model found. Run Phase 4 first.")
        return
    
    # Load current model metrics
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Check if version 1 already exists
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM model_versions WHERE version_number = 1")
    existing = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if existing:
        print("Version 1 already exists. Migration complete.")
        return
    
    # Save as version 1
    classification_report = metrics.get('classification_report', {})
    
    save_model_version(
        version_num=1,
        metrics=classification_report,
        model_path="models/sentiment_pipeline.joblib",
        training_samples=metrics.get('test_samples', 0) * 5,  # Estimate training samples
        notes="Initial model from Phase 4 training",
        is_active=True
    )
    
    print("✓ Migrated existing model to version 1 (active)")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")

if __name__ == "__main__":
    migrate_current_model()
```

**Run it:**
```bash
python scripts/init_model_versions.py
```

---

## Part 6: Testing the System

### 6.1 Manual Testing Checklist

1. **Test Prediction with Explanation:**
   - Go to `http://localhost:5000/`
   - Enter: "This flight is terrible! Lost my bags and no help from staff"
   - Submit and verify you see:
     - Sentiment prediction
     - Confidence score
     - Reasoning explanation
     - Key words detected

2. **Test Admin Review Workflow:**
   - Login to admin panel
   - Review pending submissions
   - Approve 5-10 submissions and set true_sentiment
   - Check they appear in "Approved" tab
   - Verify match/mismatch indicator shows correctly

3. **Test Model Retraining:**
   - Ensure you have at least 50 approved reviews with true_sentiment
   - Click "Retrain Model" button
   - Wait for progress spinner
   - Verify comparison modal appears showing:
     - Old accuracy vs new accuracy
     - Accuracy difference
     - Training sample count
   - Try both "Use New Model" and "Keep Current Model"

4. **Test Model Switching:**
   - After retraining, dashboard should show model versions
   - Try activating different versions
   - Verify predictions use the active model

---

## Part 7: Additional Enhancements (Optional)

### 7.1 Add Dashboard Metrics Card

Update `app/templates/dashboard.html` to show model version info:

```html
<!-- Add this after the existing cards -->
<div class="card">
    <h2>Model Information</h2>
    <div class="model-info-grid">
        <div class="info-item">
            <h3>Current Model</h3>
            {% if current_model %}
                <p><strong>Version:</strong> {{ current_model.version_number }}</p>
                <p><strong>Accuracy:</strong> {{ "%.2f"|format(current_model.accuracy * 100) }}%</p>
                <p><strong>Training Samples:</strong> {{ current_model.training_samples }}</p>
                <p><strong>Created:</strong> {{ current_model.created_at.strftime('%Y-%m-%d') }}</p>
            {% else %}
                <p>Version 1 (Original)</p>
            {% endif %}
        </div>
        
        <div class="info-item">
            <h3>Available for Training</h3>
            <p class="big-number">{{ approved_training_count }}</p>
            <p class="label">Approved reviews with true sentiment</p>
        </div>
    </div>
    
    {% if all_versions and all_versions|length > 1 %}
    <div class="version-history">
        <h3>Training History</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Version</th>
                    <th>Accuracy</th>
                    <th>Samples</th>
                    <th>Created</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for version in all_versions %}
                <tr>
                    <td>v{{ version.version_number }}</td>
                    <td>{{ "%.2f"|format(version.accuracy * 100) }}%</td>
                    <td>{{ version.training_samples }}</td>
                    <td>{{ version.created_at.strftime('%Y-%m-%d') }}</td>
                    <td>
                        {% if version.is_active %}
                            <span class="badge-active">Active</span>
                        {% else %}
                            <span class="badge-inactive">Inactive</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
</div>

<style>
.model-info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.info-item {
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}

.big-number {
    font-size: 3em;
    font-weight: bold;
    color: #0a3d62;
    margin: 10px 0;
}

.label {
    color: #666;
    font-size: 0.9em;
}

.version-history {
    margin-top: 30px;
}

.badge-active {
    background-color: #28a745;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: bold;
}

.badge-inactive {
    background-color: #6c757d;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
}
</style>
```

### 7.2 Add Auto-Refresh for Training Status

Add to `app/templates/admin.html` inside `<script>` tags:

```javascript
// Poll for training completion if needed
let trainingPollInterval = null;

function pollTrainingStatus() {
    // Implementation for checking training status via AJAX
    // This would require a new endpoint that tracks training progress
    // For now, training is synchronous
}
```

---

## Part 8: Deployment Steps

### 8.1 Step-by-Step Deployment

1. **Backup existing database:**
```bash
mysqldump -u root airline_sentiment > backup_before_upgrade.sql
```

2. **Update Python code:**
   - Copy all updated files from Part 1-4 into your project
   - Ensure `src/data/database.py` has the new functions
   - Ensure `src/models/training.py` has retraining functions
   - Replace `app/routes.py` completely
   - Replace `app/templates/admin.html` and `index.html`

3. **Initialize new database schema:**
```bash
python scripts/init_model_versions.py
```

4. **Restart Flask application:**
```bash
python run.py
```

5. **Verify functionality:**
   - Visit homepage and test prediction
   - Login to admin panel
   - Check all three tabs (Pending, Approved, Rejected)
   - Approve some submissions with true_sentiment
   - Test retraining once you have 50+ approved reviews

6. **Monitor logs:**
```bash
# Watch pipeline logs for errors
tail -f logs/pipeline.log
```

---

## Part 9: Troubleshooting Guide

### 9.1 Common Issues and Solutions

**Issue: "Table model_versions doesn't exist"**
```bash
# Solution: Run migration script
python scripts/init_model_versions.py
```

**Issue: "Insufficient training data" when retraining**
```python
# Solution: Check how many approved reviews have true_sentiment set
# Add this to a test script:
from src.data.database import count_approved_with_true_sentiment
count = count_approved_with_true_sentiment()
print(f"Available for training: {count}")
# You need at least 50
```

**Issue: Model comparison modal doesn't appear**
```
# Solution: Check browser console for JavaScript errors
# Verify the modal HTML exists in admin.html
# Check that newModelVersion is being set correctly
```

**Issue: Feature importance shows empty**
```python
# Solution: This happens with non-linear models
# Linear SVC should work. Verify model is loaded:
from app.routes import MODEL_PIPELINE
print(type(MODEL_PIPELINE.named_steps['clf']))
# Should show LinearSVC
```

**Issue: Database connection errors**
```bash
# Solution: Verify MySQL is running
# Windows (XAMPP):
# Start XAMPP Control Panel > Start MySQL

# Check .env file has correct credentials:
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=airline_sentiment
```

**Issue: Retraining takes too long**
```python
# Solution: The model retrains in ~8 seconds for 14k samples
# If it's taking longer, check:
# 1. System resources (CPU usage)
# 2. Dataset size (query the database)
# 3. Consider running in background with Celery (advanced)
```

### 9.2 Validation Queries

Run these in MySQL to validate data:

```sql
-- Check submissions by status
SELECT review_status, COUNT(*) as count 
FROM submissions 
GROUP BY review_status;

-- Check approved with true_sentiment
SELECT COUNT(*) as training_ready
FROM submissions 
WHERE review_status = 'approved' 
AND true_sentiment IS NOT NULL;

-- Check model versions
SELECT version_number, accuracy, is_active, created_at 
FROM model_versions 
ORDER BY version_number DESC;

-- Check prediction accuracy
SELECT 
    predicted_sentiment,
    true_sentiment,
    COUNT(*) as count
FROM submissions
WHERE review_status = 'approved' 
AND true_sentiment IS NOT NULL
GROUP BY predicted_sentiment, true_sentiment;
```

---

## Part 10: Advanced Features (Future Enhancements)

### 10.1 Async Training with Celery

For production systems with large datasets, implement background training:

```python
# requirements.txt additions:
# celery==5.3.4
# redis==5.0.1

# celery_app.py
from celery import Celery

celery = Celery('sentiment_tasks',
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/0')

@celery.task
def retrain_model_async(new_data):
    from src.models.training import retrain_with_new_data
    pipeline, results = retrain_with_new_data(new_data)
    return results

# In routes.py, change retrain endpoint to:
@bp.route("/admin/retrain", methods=["POST"])
def retrain_model():
    new_data = database.fetch_training_data()
    task = retrain_model_async.delay(new_data)
    return jsonify({'task_id': task.id})

@bp.route("/admin/retrain/status/<task_id>")
def retrain_status(task_id):
    task = retrain_model_async.AsyncResult(task_id)
    if task.ready():
        return jsonify({'status': 'complete', 'result': task.result})
    return jsonify({'status': 'pending'})
```

### 10.2 A/B Testing Models

Compare multiple models in production:

```python
# Add to database.py
def log_prediction_performance(submission_id, model_version, was_correct):
    """Track which model versions perform better"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_logs 
        (submission_id, model_version, was_correct, logged_at)
        VALUES (%s, %s, %s, NOW())
    """, (submission_id, model_version, was_correct))
    conn.commit()
    cursor.close()
    conn.close()
```

### 10.3 Real-time Metrics Dashboard

Add live accuracy tracking:

```javascript
// Add to dashboard.html
function updateLiveMetrics() {
    fetch('/api/live-metrics')
        .then(r => r.json())
        .then(data => {
            document.getElementById('live-accuracy').textContent = 
                (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('pending-count').textContent = 
                data.pending_count;
        });
}

setInterval(updateLiveMetrics, 30000); // Update every 30 seconds
```

### 10.4 Automated Retraining Schedule

Set up automatic retraining:

```python
# scripts/scheduled_retrain.py
import schedule
import time
from src.data.database import count_approved_with_true_sentiment, fetch_training_data
from src.models.training import retrain_with_new_data, save_model_version

def auto_retrain():
    count = count_approved_with_true_sentiment()
    if count >= 100:  # Threshold for auto-retraining
        print(f"Starting auto-retrain with {count} samples...")
        new_data = fetch_training_data()
        pipeline, results = retrain_with_new_data(new_data)
        # Auto-activate if accuracy improves by 2%
        # Implementation here...

schedule.every().day.at("02:00").do(auto_retrain)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Part 11: Performance Optimization

### 11.1 Database Indexing

Add these indexes for better query performance:

```sql
-- Add indexes to submissions table
ALTER TABLE submissions ADD INDEX idx_review_status_true_sentiment (review_status, true_sentiment);
ALTER TABLE submissions ADD INDEX idx_updated_at (updated_at);

-- Add indexes to model_versions table
ALTER TABLE model_versions ADD INDEX idx_is_active (is_active);
```

### 11.2 Caching Predictions

Implement prediction caching for repeated tweets:

```python
# Add to routes.py
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_predict(text_hash):
    """Cache predictions by text hash"""
    # Actual prediction happens here
    pass

@bp.route("/", methods=["POST"])
def index():
    text = form.tweet_text.data.strip()
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache first
    cached = cached_predict(text_hash)
    if cached:
        prediction_result = cached
    else:
        # Do actual prediction
        explanation = explain_prediction(MODEL_PIPELINE, text)
        prediction_result = explanation
```

### 11.3 Batch Processing

Process multiple predictions efficiently:

```python
# Add to training.py
def batch_explain_predictions(texts: list) -> list:
    """Process multiple texts in one batch"""
    predictions = MODEL_PIPELINE.predict(texts)
    explanations = []
    
    for text, pred in zip(texts, predictions):
        explanations.append(explain_prediction(MODEL_PIPELINE, text))
    
    return explanations
```

---

## Part 12: Security Enhancements

### 12.1 Rate Limiting

Protect the prediction endpoint:

```python
# requirements.txt addition:
# Flask-Limiter==3.5.0

# In app/__init__.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# In routes.py
@bp.route("/", methods=["POST"])
@limiter.limit("10 per minute")
def index():
    # existing code...
```

### 12.2 Input Sanitization

Add text validation:

```python
# Add to forms.py
from wtforms.validators import Regexp

class PredictionForm(FlaskForm):
    tweet_text = TextAreaField(
        'Tweet Text',
        validators=[
            DataRequired(),
            Length(min=10, max=500),
            Regexp(
                r'^[a-zA-Z0-9\s\.,!?@#\-\'\"]+# Continuous Learning System Implementation Guide

## Overview
This guide implements a human-in-the-loop continuous learning system with:
- Complete review history display (unmarked shown first)
- Manual model retraining from dashboard
- Training progress indicators
- Model comparison & rollback capability
- Prediction explanations using top influential features

---

## Part 1: Database Schema Updates

### 1.1 Update `src/data/database.py`

Add new table for model versioning and update the submissions table:

```python
# Add this function after initialize_database()

def create_model_versions_table(cursor):
    """Create table to track model versions and metrics"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            version_number INT NOT NULL,
            accuracy DECIMAL(6,4) NOT NULL,
            precision_macro DECIMAL(6,4),
            recall_macro DECIMAL(6,4),
            f1_macro DECIMAL(6,4),
            training_samples INT NOT NULL,
            model_path VARCHAR(255) NOT NULL,
            metrics_path VARCHAR(255),
            is_active BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE KEY unique_version (version_number)
        )
    """)

def initialize_database() -> None:
    """Initialize database and required tables"""
    config = DatabaseConfig.from_env()
    
    # Create database if not exists
    conn = get_connection(include_database=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
    cursor.close()
    conn.close()
    
    # Create tables
    conn = get_connection()
    cursor = conn.cursor()
    
    # Original submissions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tweet_text TEXT NOT NULL,
            predicted_sentiment VARCHAR(20) NOT NULL,
            prediction_confidence DECIMAL(5,4),
            assigned_airline VARCHAR(50),
            true_sentiment VARCHAR(20),
            review_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
            admin_comment TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_review_status (review_status),
            INDEX idx_created_at (created_at)
        )
    """)
    
    # New model versions table
    create_model_versions_table(cursor)
    
    conn.commit()
    cursor.close()
    conn.close()
```

### 1.2 Add New Database Functions

```python
# Add these functions to src/data/database.py

def fetch_all_submissions_grouped(limit: int = 1000) -> dict:
    """Fetch submissions grouped by review status"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Pending submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'pending'
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    pending = cursor.fetchall()
    
    # Approved submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'approved'
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))
    approved = cursor.fetchall()
    
    # Rejected submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'rejected'
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))
    rejected = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {
        'pending': pending,
        'approved': approved,
        'rejected': rejected
    }

def count_approved_with_true_sentiment() -> int:
    """Count approved submissions with true_sentiment set"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM submissions 
        WHERE review_status = 'approved' 
        AND true_sentiment IS NOT NULL 
        AND true_sentiment != ''
    """)
    count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    return count

def fetch_training_data():
    """Fetch approved submissions with true_sentiment for retraining"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT tweet_text, true_sentiment as sentiment
        FROM submissions 
        WHERE review_status = 'approved' 
        AND true_sentiment IS NOT NULL 
        AND true_sentiment != ''
        AND tweet_text IS NOT NULL
        ORDER BY updated_at ASC
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_model_version(version_num: int, metrics: dict, model_path: str, 
                       training_samples: int, notes: str = None, is_active: bool = False):
    """Save model version metadata"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO model_versions 
        (version_number, accuracy, precision_macro, recall_macro, f1_macro, 
         training_samples, model_path, is_active, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        version_num,
        metrics.get('accuracy', 0),
        metrics.get('macro avg', {}).get('precision', 0),
        metrics.get('macro avg', {}).get('recall', 0),
        metrics.get('macro avg', {}).get('f1-score', 0),
        training_samples,
        model_path,
        is_active,
        notes
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_latest_model_version():
    """Get the latest active model version info"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM model_versions 
        WHERE is_active = TRUE
        ORDER BY version_number DESC 
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

def set_active_model(version_number: int):
    """Set a specific version as active, deactivating others"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Deactivate all models
    cursor.execute("UPDATE model_versions SET is_active = FALSE")
    
    # Activate selected model
    cursor.execute("""
        UPDATE model_versions 
        SET is_active = TRUE 
        WHERE version_number = %s
    """, (version_number,))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_all_model_versions():
    """Get all model versions ordered by version number"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM model_versions 
        ORDER BY version_number DESC
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results
```

---

## Part 2: Enhanced Training Module

### 2.1 Update `src/models/training.py`

Add functions for retraining and feature importance:

```python
# Add these imports at the top
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# Add this function after select_and_train_best()

def retrain_with_new_data(new_data_rows: list, model_name: str = "linear_svc") -> dict:
    """
    Retrain model with original dataset + new approved submissions
    
    Args:
        new_data_rows: List of dicts with 'tweet_text' and 'sentiment' keys
        model_name: Which model to retrain (linear_svc, logistic_regression, complement_nb)
    
    Returns:
        dict with training results and comparison metrics
    """
    from src.data.loaders import load_normalized_dataset
    
    # Load original dataset
    original_df = load_normalized_dataset()
    
    # Create dataframe from new data
    if not new_data_rows:
        raise ValueError("No new training data provided")
    
    new_df = pd.DataFrame(new_data_rows)
    new_df.columns = ['text', 'airline_sentiment']
    
    # Combine datasets
    combined_df = pd.concat([
        original_df[['text', 'airline_sentiment']], 
        new_df
    ], ignore_index=True)
    
    # Prepare dataset
    X, y = prepare_dataset(combined_df)
    
    if len(X) == 0:
        raise ValueError("No valid training data after preprocessing")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and train the specified model
    pipelines = build_pipelines()
    if model_name not in pipelines:
        raise ValueError(f"Model {model_name} not found. Choose from {list(pipelines.keys())}")
    
    pipeline = pipelines[model_name]
    
    logger.info(f"Retraining {model_name} with {len(X_train)} samples (including {len(new_data_rows)} new)")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'labels': pipeline.classes_.tolist(),
        'total_training_samples': len(X_train),
        'new_samples_added': len(new_data_rows),
        'test_samples': len(X_test)
    }
    
    logger.info(f"Retraining complete. New accuracy: {accuracy:.4f}")
    
    return pipeline, results

def save_model_version(pipeline, metrics: dict, version_number: int):
    """Save a versioned model"""
    models_dir = Path("models")
    version_dir = models_dir / f"version_{version_number}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = version_dir / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    
    # Save metrics
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved model version {version_number} to {version_dir}")
    
    return str(model_path), str(metrics_path)

def get_feature_importance(pipeline, text: str, top_n: int = 5) -> list:
    """
    Extract top N features that influenced the prediction
    
    Args:
        pipeline: Trained sklearn pipeline
        text: Input tweet text
        top_n: Number of top features to return
    
    Returns:
        List of tuples (feature_name, weight)
    """
    # Preprocess text
    cleaned = preprocess_text(text)
    
    # Get feature vector
    tfidf_vector = pipeline.named_steps['tfidf'].transform([cleaned])
    
    # Get classifier coefficients
    classifier = pipeline.named_steps['clf']
    
    # For Linear SVC
    if hasattr(classifier, 'coef_'):
        # Get prediction
        prediction = pipeline.predict([cleaned])[0]
        pred_idx = list(pipeline.classes_).index(prediction)
        
        # Get coefficients for predicted class
        coef = classifier.coef_[pred_idx]
        
        # Get feature names
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Get non-zero features from the vector
        nonzero_indices = tfidf_vector.nonzero()[1]
        
        # Calculate feature contributions
        contributions = []
        for idx in nonzero_indices:
            feature_name = feature_names[idx]
            feature_weight = coef[idx] * tfidf_vector[0, idx]
            contributions.append((feature_name, float(feature_weight)))
        
        # Sort by absolute weight
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:top_n]
    
    # For Naive Bayes or other models
    return []

def explain_prediction(pipeline, text: str) -> dict:
    """
    Generate human-readable explanation for prediction
    
    Returns:
        dict with sentiment, confidence, and reasoning
    """
    prediction = pipeline.predict([text])[0]
    
    # Get confidence if available
    confidence = None
    if hasattr(pipeline.named_steps['clf'], 'decision_function'):
        decision = pipeline.named_steps['clf'].decision_function([preprocess_text(text)])
        # Convert to pseudo-probability
        from scipy.special import expit
        probs = expit(decision)[0]
        pred_idx = list(pipeline.classes_).index(prediction)
        confidence = float(probs[pred_idx]) if len(probs.shape) > 0 else float(probs)
    
    # Get top features
    top_features = get_feature_importance(pipeline, text, top_n=5)
    
    # Build explanation
    if top_features:
        positive_features = [f for f, w in top_features if w > 0]
        negative_features = [f for f, w in top_features if w < 0]
        
        reasoning_parts = []
        if positive_features:
            reasoning_parts.append(f"Key indicators: {', '.join(positive_features[:3])}")
        
        explanation = {
            'sentiment': prediction,
            'confidence': confidence,
            'top_features': [f for f, w in top_features],
            'feature_weights': {f: w for f, w in top_features},
            'reasoning': ' | '.join(reasoning_parts) if reasoning_parts else "Based on overall text patterns"
        }
    else:
        explanation = {
            'sentiment': prediction,
            'confidence': confidence,
            'top_features': [],
            'feature_weights': {},
            'reasoning': "Classification based on learned patterns"
        }
    
    return explanation
```

---

## Part 3: Update Flask Routes

### 3.1 Update `app/routes.py`

Replace the entire file with enhanced version:

```python
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from app.forms import PredictionForm, AdminLoginForm
from src.data import database
from src.models.training import load_trained_model, explain_prediction, retrain_with_new_data, save_model_version
import os
from pathlib import Path
import json
import pandas as pd
import shutil
from datetime import datetime

bp = Blueprint('main', __name__)

# Global model variables
MODEL_PIPELINE = None
MODEL_LABELS = None
MODEL_METADATA = None

def load_model():
    """Load the trained model and metadata"""
    global MODEL_PIPELINE, MODEL_LABELS, MODEL_METADATA
    
    # Check for versioned model first
    latest_version = database.get_latest_model_version()
    
    if latest_version:
        model_path = Path(latest_version['model_path'])
        if model_path.exists():
            MODEL_PIPELINE = load_trained_model(str(model_path))
            MODEL_METADATA = latest_version
        else:
            # Fall back to default
            MODEL_PIPELINE = load_trained_model()
            MODEL_METADATA = None
    else:
        MODEL_PIPELINE = load_trained_model()
        MODEL_METADATA = None
    
    if MODEL_PIPELINE:
        MODEL_LABELS = MODEL_PIPELINE.classes_.tolist()
    
    # Load metrics
    metrics_file = Path("models/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            if MODEL_METADATA is None:
                MODEL_METADATA = metrics

# Airline options
AIRLINES = [
    ("", "-- Select Airline (Optional) --"),
    ("United", "United Airlines"),
    ("US Airways", "US Airways"),
    ("American", "American Airlines"),
    ("Southwest", "Southwest Airlines"),
    ("Delta", "Delta Airlines"),
    ("Virgin America", "Virgin America")
]

def ensure_static_assets():
    """Copy visualization assets to static directory"""
    static_img = Path("app/static/img")
    static_html = Path("app/static/html")
    static_img.mkdir(parents=True, exist_ok=True)
    static_html.mkdir(parents=True, exist_ok=True)
    
    vis_dir = Path("visualizations")
    if vis_dir.exists():
        for png_file in vis_dir.glob("*.png"):
            shutil.copy(png_file, static_img / png_file.name)
        for html_file in vis_dir.glob("*.html"):
            shutil.copy(html_file, static_html / html_file.name)

@bp.route("/", methods=["GET", "POST"])
def index():
    form = PredictionForm()
    form.set_airline_choices(AIRLINES)
    prediction_result = None
    recent = database.fetch_recent_submissions(limit=5)
    
    if form.validate_on_submit():
        text = form.tweet_text.data.strip()
        if not text:
            flash("Tweet text cannot be empty.", "danger")
            return render_template("index.html", form=form, prediction=prediction_result, recent_submissions=recent)
        
        # Get prediction with explanation
        explanation = explain_prediction(MODEL_PIPELINE, text)
        
        assigned_airline = form.airline.data or None
        
        # Insert to database
        database.insert_submission(
            tweet_text=text,
            predicted_sentiment=explanation['sentiment'],
            prediction_confidence=explanation['confidence'],
            assigned_airline=assigned_airline,
        )
        
        prediction_result = {
            'sentiment': explanation['sentiment'],
            'confidence': explanation['confidence'],
            'reasoning': explanation['reasoning'],
            'top_features': explanation['top_features'],
            'airline': assigned_airline
        }
        
        flash("Sentiment prediction saved for review.", "success")
        recent = database.fetch_recent_submissions(limit=5)
    
    return render_template("index.html", form=form, prediction=prediction_result, recent_submissions=recent)

@bp.route("/dashboard")
def dashboard():
    context = load_dashboard_context()
    
    # Add training data count
    approved_count = database.count_approved_with_true_sentiment()
    context['approved_training_count'] = approved_count
    
    # Add model version info
    context['current_model'] = MODEL_METADATA
    context['all_versions'] = database.get_all_model_versions()
    
    return render_template("dashboard.html", **context)

@bp.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST" and not session.get("is_admin"):
        form = AdminLoginForm()
        if form.validate_on_submit():
            from flask import current_app
            if form.password.data == current_app.config.get("ADMIN_PASSWORD"):
                session["is_admin"] = True
                flash("Login successful!", "success")
                return redirect(url_for("main.admin"))
            else:
                flash("Invalid password.", "danger")
        return render_template("admin_login.html", form=form)
    
    if not session.get("is_admin"):
        form = AdminLoginForm()
        return render_template("admin_login.html", form=form)
    
    # Fetch all submissions grouped by status
    submissions = database.fetch_all_submissions_grouped(limit=500)
    
    return render_template("admin.html", submissions=submissions)

@bp.route("/admin/review/<int:submission_id>", methods=["POST"])
def review_submission(submission_id: int):
    if not session.get("is_admin"):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("main.admin"))
    
    action = request.form.get("action")
    true_sentiment = request.form.get("true_sentiment")
    admin_comment = request.form.get("admin_comment")
    
    if action == "approve":
        status = "approved"
        flash(f"Submission {submission_id} approved.", "success")
    elif action == "reject":
        status = "rejected"
        flash(f"Submission {submission_id} rejected.", "info")
    else:
        flash("Invalid action.", "danger")
        return redirect(url_for("main.admin"))
    
    database.update_submission_status(
        submission_id=submission_id,
        status=status,
        true_sentiment=true_sentiment if true_sentiment else None,
        admin_comment=admin_comment if admin_comment else None
    )
    
    return redirect(url_for("main.admin"))

@bp.route("/admin/retrain", methods=["POST"])
def retrain_model():
    """Trigger model retraining with approved data"""
    if not session.get("is_admin"):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get training data
        new_data = database.fetch_training_data()
        
        if len(new_data) < 50:
            return jsonify({
                'error': f'Insufficient training data. Need at least 50 approved reviews with true_sentiment set. Currently have {len(new_data)}.'
            }), 400
        
        # Get current model accuracy
        old_accuracy = MODEL_METADATA.get('accuracy', 0) if MODEL_METADATA else 0
        
        # Retrain model
        new_pipeline, results = retrain_with_new_data(new_data, model_name="linear_svc")
        
        # Get next version number
        all_versions = database.get_all_model_versions()
        next_version = max([v['version_number'] for v in all_versions], default=0) + 1
        
        # Save as new version
        model_path, metrics_path = save_model_version(new_pipeline, results, next_version)
        
        # Save to database
        database.save_model_version(
            version_num=next_version,
            metrics=results['classification_report'],
            model_path=model_path,
            training_samples=results['total_training_samples'],
            notes=f"Retrained with {results['new_samples_added']} new samples",
            is_active=False  # Don't activate yet, let admin choose
        )
        
        new_accuracy = results['accuracy']
        accuracy_diff = new_accuracy - old_accuracy
        
        return jsonify({
            'success': True,
            'version': next_version,
            'old_accuracy': float(old_accuracy),
            'new_accuracy': float(new_accuracy),
            'accuracy_difference': float(accuracy_diff),
            'improvement_percentage': float((accuracy_diff / old_accuracy * 100) if old_accuracy > 0 else 0),
            'training_samples': results['total_training_samples'],
            'new_samples': results['new_samples_added'],
            'message': f'Model version {next_version} trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/admin/model/<int:version>/activate", methods=["POST"])
def activate_model(version: int):
    """Activate a specific model version"""
    if not session.get("is_admin"):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        database.set_active_model(version)
        
        # Reload model globally
        load_model()
        
        flash(f"Model version {version} is now active.", "success")
        return jsonify({'success': True, 'message': f'Model version {version} activated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/logout")
def logout():
    session.pop("is_admin", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("main.index"))

def load_dashboard_context() -> dict:
    """Load all data needed for dashboard"""
    processed_dir = Path("data/processed")
    
    context = {
        "sentiment_summary": [],
        "airline_volume": [],
        "negative_reasons": [],
        "model_accuracy": 0,
        "model_name": "Unknown",
        "visuals": {}
    }
    
    # Load sentiment distribution
    sentiment_file = processed_dir / "sentiment_distribution.csv"
    if sentiment_file.exists():
        df = pd.read_csv(sentiment_file)
        context["sentiment_summary"] = df.to_dict('records')
    
    # Load airline volume
    airline_file = processed_dir / "airline_tweet_volume.csv"
    if airline_file.exists():
        df = pd.read_csv(airline_file)
        context["airline_volume"] = df.to_dict('records')
    
    # Load negative reasons
    reasons_file = processed_dir / "top_negative_reasons.csv"
    if reasons_file.exists():
        df = pd.read_csv(reasons_file)
        context["negative_reasons"] = df.to_dict('records')
    
    # Load model metrics
    metrics_file = Path("models/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            context["model_accuracy"] = metrics.get("accuracy", 0)
            context["model_name"] = metrics.get("model_name", "Unknown")
    
    # Visual assets
    context["visuals"] = {
        "sentiment": "/static/img/sentiment_distribution.png",
        "timeline": "/static/img/tweet_volume_timeline.png",
        "negative": "/static/img/top_negative_reasons.png",
        "heatmap": "/static/img/airline_sentiment_heatmap.png",
        "wordcloud": "/static/img/wordcloud_negative.png",
        "model_cm": "/static/img/model_confusion_matrix.png",
        "sentiment_html": "/static/html/sentiment_distribution.html",
    }
    
    return context

# Initialize on import
ensure_static_assets()
load_model()
```

---

## Part 4: Update Templates

### 4.1 Update `app/templates/admin.html`

Replace with enhanced version showing all submissions:

```html
{% extends "base.html" %}

{% block title %}Admin Panel - Review Queue{% endblock %}
```

### 4.2 Update `app/templates/index.html`

Add prediction explanation display:

```html
{% extends "base.html" %}

{% block title %}Predict Sentiment{% endblock %}

{% block content %}
<div class="hero">
    <h1>Twitter Airline Sentiment Analyzer</h1>
    <p>Analyze sentiment of airline-related tweets with AI-powered predictions</p>
</div>

<div class="container">
    <div class="card">
        <h2>Enter a Tweet</h2>
        <form method="POST" action="{{ url_for('main.index') }}">
            {{ form.hidden_tag() }}
            
            <div class="form-group">
                {{ form.tweet_text.label }}
                {{ form.tweet_text(class="form-control", rows=4, placeholder="Example: Flight delayed again! Terrible customer service.") }}
                {% if form.tweet_text.errors %}
                    <span class="error">{{ form.tweet_text.errors[0] }}</span>
                {% endif %}
            </div>
            
            <div class="form-group">
                {{ form.airline.label }}
                {{ form.airline(class="form-control") }}
            </div>
            
            <button type="submit" class="btn btn-primary">Analyze Sentiment</button>
        </form>
    </div>

    {% if prediction %}
    <div class="card prediction-result">
        <h2>Prediction Result</h2>
        <div class="result-grid">
            <div class="result-item">
                <h3>Sentiment</h3>
                <p class="sentiment-badge sentiment-{{ prediction.sentiment }}">
                    {{ prediction.sentiment|upper }}
                </p>
            </div>
            
            <div class="result-item">
                <h3>Confidence</h3>
                <p class="confidence-value">
                    {% if prediction.confidence %}
                        {{ "%.1f"|format(prediction.confidence * 100) }}%
                    {% else %}
                        N/A
                    {% endif %}
                </p>
            </div>
        </div>
        
        {% if prediction.reasoning %}
        <div class="explanation-section">
            <h3>Why This Prediction?</h3>
            <p class="reasoning-text">{{ prediction.reasoning }}</p>
            
            {% if prediction.top_features %}
            <div class="features-list">
                <h4>Key Words Detected:</h4>
                <ul>
                    {% for feature in prediction.top_features[:5] %}
                    <li><code>{{ feature }}</code></li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if prediction.airline %}
        <p class="airline-info"><strong>Airline:</strong> {{ prediction.airline }}</p>
        {% endif %}
        
        <p class="review-note">This prediction has been saved for admin review.</p>
    </div>
    {% endif %}

    {% if recent_submissions %}
    <div class="card">
        <h2>Recent Predictions</h2>
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Tweet</th>
                        <th>Predicted</th>
                        <th>Confidence</th>
                        <th>Status</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    {% for sub in recent_submissions %}
                    <tr>
                        <td class="tweet-snippet">{{ sub.tweet_text[:80] }}...</td>
                        <td>
                            <span class="sentiment-badge sentiment-{{ sub.predicted_sentiment }}">
                                {{ sub.predicted_sentiment }}
                            </span>
                        </td>
                        <td>
                            {% if sub.prediction_confidence %}
                                {{ "%.1f"|format(sub.prediction_confidence * 100) }}%
                            {% else %}
                                N/A
                            {% endif %}
                        </td>
                        <td>
                            <span class="status-badge status-{{ sub.review_status }}">
                                {{ sub.review_status }}
                            </span>
                        </td>
                        <td>{{ sub.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}
</div>

<style>
.prediction-result {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-top: 20px;
}

.prediction-result h2,
.prediction-result h3,
.prediction-result h4 {
    color: white;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.result-item {
    text-align: center;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}

.result-item h3 {
    font-size: 1em;
    margin-bottom: 10px;
    opacity: 0.9;
}

.sentiment-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 25px;
    font-size: 1.5em;
    font-weight: bold;
}

.confidence-value {
    font-size: 2em;
    font-weight: bold;
    margin: 0;
}

.explanation-section {
    margin-top: 30px;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}

.reasoning-text {
    font-size: 1.1em;
    line-height: 1.6;
    margin: 15px 0;
}

.features-list {
    margin-top: 20px;
}

.features-list h4 {
    font-size: 1em;
    margin-bottom: 10px;
}

.features-list ul {
    list-style: none;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.features-list li {
    background: rgba(255,255,255,0.2);
    padding: 5px 15px;
    border-radius: 15px;
}

.features-list code {
    color: white;
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
}

.airline-info {
    margin-top: 20px;
    font-size: 1.1em;
}

.review-note {
    margin-top: 15px;
    font-style: italic;
    opacity: 0.8;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: bold;
}

.status-pending {
    background-color: #ffc107;
    color: #000;
}

.status-approved {
    background-color: #28a745;
    color: white;
}

.status-rejected {
    background-color: #dc3545;
    color: white;
}

.tweet-snippet {
    max-width: 300px;
}
</style>
{% endblock %}

{% block content %}
<div class="admin-container">
    <div class="admin-header">
        <h1>Admin Review Panel</h1>
        <a href="{{ url_for('main.logout') }}" class="btn btn-secondary">Logout</a>
    </div>

    <!-- Training Section -->
    <div class="card training-section">
        <h2>Model Training</h2>
        <div class="training-stats">
            <p><strong>Approved Reviews Available for Training:</strong> 
                {{ submissions.approved|selectattr('true_sentiment')|list|length }}</p>
            <p class="help-text">Minimum 50 reviews with true sentiment required for retraining</p>
        </div>
        
        <button id="retrainBtn" class="btn btn-primary" onclick="startRetraining()">
            <span id="retrainText">Retrain Model</span>
            <span id="retrainSpinner" class="spinner" style="display:none;">⟳ Training...</span>
        </button>
        
        <div id="trainingResults" style="display:none; margin-top: 20px;">
            <!-- Results will be injected here -->
        </div>
    </div>

    <!-- Model Comparison Modal -->
    <div id="comparisonModal" class="modal" style="display:none;">
        <div class="modal-content">
            <h2>Model Training Complete</h2>
            <div id="comparisonDetails"></div>
            <div class="modal-actions">
                <button onclick="activateNewModel()" class="btn btn-primary">Use New Model</button>
                <button onclick="keepOldModel()" class="btn btn-secondary">Keep Current Model</button>
            </div>
        </div>
    </div>

    <!-- Submissions Tabs -->
    <div class="submissions-tabs">
        <button class="tab-btn active" onclick="showTab('pending')">
            Pending ({{ submissions.pending|length }})
        </button>
        <button class="tab-btn" onclick="showTab('approved')">
            Approved ({{ submissions.approved|length }})
        </button>
        <button class="tab-btn" onclick="showTab('rejected')">
            Rejected ({{ submissions.rejected|length }})
        </button>
    </div>

    <!-- Pending Submissions -->
    <div id="pending-tab" class="tab-content active">
        <h2>Pending Reviews</h2>
        {% if submissions.pending %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>Confidence</th>
                            <th>Airline</th>
                            <th>Date</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.pending %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>
                                {% if submission.prediction_confidence %}
                                    {{ "%.1f"|format(submission.prediction_confidence * 100) }}%
                                {% else %}
                                    N/A
                                {% endif %}
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td>{{ submission.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>
                                <form method="POST" action="{{ url_for('main.review_submission', submission_id=submission.id) }}" class="inline-form">
                                    <select name="true_sentiment" required>
                                        <option value="">True Sentiment</option>
                                        <option value="negative">Negative</option>
                                        <option value="neutral">Neutral</option>
                                        <option value="positive">Positive</option>
                                    </select>
                                    <input type="text" name="admin_comment" placeholder="Comment (optional)" class="comment-input">
                                    <button type="submit" name="action" value="approve" class="btn btn-sm btn-success">Approve</button>
                                    <button type="submit" name="action" value="reject" class="btn btn-sm btn-danger">Reject</button>
                                </form>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No pending submissions</p>
        {% endif %}
    </div>

    <!-- Approved Submissions -->
    <div id="approved-tab" class="tab-content">
        <h2>Approved Reviews</h2>
        {% if submissions.approved %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>True Sentiment</th>
                            <th>Match</th>
                            <th>Airline</th>
                            <th>Comment</th>
                            <th>Reviewed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.approved %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>
                                {% if submission.true_sentiment %}
                                    <span class="sentiment-badge sentiment-{{ submission.true_sentiment }}">
                                        {{ submission.true_sentiment }}
                                    </span>
                                {% else %}
                                    <span class="text-muted">Not set</span>
                                {% endif %}
                            </td>
                            <td>
                                {% if submission.true_sentiment %}
                                    {% if submission.predicted_sentiment == submission.true_sentiment %}
                                        <span class="badge-success">✓ Match</span>
                                    {% else %}
                                        <span class="badge-error">✗ Mismatch</span>
                                    {% endif %}
                                {% else %}
                                    -
                                {% endif %}
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td class="comment-cell">{{ submission.admin_comment or '-' }}</td>
                            <td>{{ submission.updated_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No approved submissions yet</p>
        {% endif %}
    </div>

    <!-- Rejected Submissions -->
    <div id="rejected-tab" class="tab-content">
        <h2>Rejected Reviews</h2>
        {% if submissions.rejected %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>Airline</th>
                            <th>Comment</th>
                            <th>Rejected</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.rejected %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td class="comment-cell">{{ submission.admin_comment or '-' }}</td>
                            <td>{{ submission.updated_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No rejected submissions</p>
        {% endif %}
    </div>
</div>

<script>
let newModelVersion = null;

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

async function startRetraining() {
    const btn = document.getElementById('retrainBtn');
    const text = document.getElementById('retrainText');
    const spinner = document.getElementById('retrainSpinner');
    const results = document.getElementById('trainingResults');
    
    // Disable button and show spinner
    btn.disabled = true;
    text.style.display = 'none';
    spinner.style.display = 'inline';
    results.style.display = 'none';
    
    try {
        const response = await fetch('/admin/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.error) {
            results.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
            results.style.display = 'block';
        } else {
            newModelVersion = data.version;
            showComparisonModal(data);
        }
    } catch (error) {
        results.innerHTML = `<div class="alert alert-error">Training failed: ${error.message}</div>`;
        results.style.display = 'block';
    } finally {
        // Re-enable button
        btn.disabled = false;
        text.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

function showComparisonModal(data) {
    const modal = document.getElementById('comparisonModal');
    const details = document.getElementById('comparisonDetails');
    
    const improvementText = data.accuracy_difference > 0 
        ? `<span style="color: green;">+${(data.improvement_percentage).toFixed(2)}% improvement</span>`
        : `<span style="color: red;">${(data.improvement_percentage).toFixed(2)}% decrease</span>`;
    
    details.innerHTML = `
        <div class="comparison-grid">
            <div class="comparison-item">
                <h3>Current Model</h3>
                <p class="metric-value">${(data.old_accuracy * 100).toFixed(2)}%</p>
                <p class="metric-label">Accuracy</p>
            </div>
            <div class="comparison-item">
                <h3>New Model (v${data.version})</h3>
                <p class="metric-value">${(data.new_accuracy * 100).toFixed(2)}%</p>
                <p class="metric-label">Accuracy</p>
            </div>
        </div>
        <div class="comparison-summary">
            <p><strong>Change:</strong> ${improvementText}</p>
            <p><strong>Training samples:</strong> ${data.training_samples} (${data.new_samples} new)</p>
            <p class="help-text">Choose whether to activate the new model or keep using the current one.</p>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function activateNewModel() {
    try {
        const response = await fetch(`/admin/model/${newModelVersion}/activate`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('New model activated! The page will reload.');
            location.reload();
        } else {
            alert('Failed to activate model: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function keepOldModel() {
    document.getElementById('comparisonModal').style.display = 'none';
    const results = document.getElementById('trainingResults');
    results.innerHTML = `<div class="alert alert-info">New model version ${newModelVersion} saved but not activated. Current model remains in use.</div>`;
    results.style.display = 'block';
}
</script>

<style>
.admin-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.admin-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.training-section {
    margin-bottom: 30px;
    padding: 20px;
}

.training-stats {
    margin-bottom: 15px;
}

.help-text {
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
}

.spinner {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.submissions-tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
}

.tab-btn {
    padding: 10px 20px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 16px;
    color: #666;
    border-bottom: 3px solid transparent;
    transition: all 0.3s;
}

.tab-btn:hover {
    color: #0a3d62;
}

.tab-btn.active {
    color: #0a3d62;
    border-bottom-color: #0a3d62;
    font-weight: bold;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.data-table-container {
    overflow-x: auto;
}

.tweet-cell {
    max-width: 300px;
    word-wrap: break-word;
}

.comment-cell {
    max-width: 200px;
    font-style: italic;
    color: #666;
}

.sentiment-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: bold;
}

.sentiment-negative {
    background-color: #fee;
    color: #c00;
}

.sentiment-neutral {
    background-color: #ffc;
    color: #660;
}

.sentiment-positive {
    background-color: #efe;
    color: #060;
}

.badge-success {
    color: green;
    font-weight: bold;
}

.badge-error {
    color: red;
    font-weight: bold;
}

.inline-form {
    display: flex;
    gap: 5px;
    align-items: center;
}

.comment-input {
    padding: 4px 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 0.9em;
}

.btn-sm {
    padding: 4px 12px;
    font-size: 0.85em;
}

.btn-success {
    background-color: #28a745;
    color: white;
}

.btn-danger {
    background-color: #dc3545;
    color: white;
}

.empty-state {
    text-align: center;
    padding: 40px;
    color: #999;
    font-style: italic;
}

.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal-content {
    background: white;
    padding: 30px;
    border-radius: 8px;
    max-width: 600px;
    width: 90%;
}

.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}

.comparison-item {
    text-align: center;
    padding: 20px;
    border: 2px solid #ddd;
    border-radius: 8px;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #0a3d62;
    margin: 10px 0;
}

.metric-label {
    color: #666;
    font-size: 0.9em;
}

.comparison-summary {
    margin-top: 20px;
    padding: 15px;
    background: #f5f5f5;
    border-radius: 4px;
}

.modal-actions {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 20px;
}

.alert {
    padding: 15px;
    border-radius: 4px;
    margin-top: 15px;
}

.alert-error {
    background-color: #fee;
    color: #c00;
    border: 1px solid #fcc;
}

.alert-info {
    background-color: #e7f3ff;
    color: #004085;
    border: 1px solid #b8daff;
}

.text-muted {
    color: #999;
}
</style>
{% endblock %},
                message="Tweet contains invalid characters"
            )
        ]
    )
```

### 12.3 CSRF Token Verification

Ensure all forms are protected:

```html
<!-- Already implemented in templates, but verify: -->
<form method="POST">
    {{ form.hidden_tag() }}  <!-- This generates CSRF token -->
    <!-- rest of form -->
</form>
```

---

## Part 13: Documentation Updates

### 13.1 Update PROJECT_SUMMARY.md

Add these sections:

```markdown
## Continuous Learning Features

### Human-in-the-Loop Training
- Admins review all predictions and mark true sentiment
- Approved reviews with true_sentiment become training data
- System requires minimum 50 approved reviews before retraining

### Model Versioning
- Each training run creates a new versioned model
- Models are stored with full metrics and metadata
- Admins can compare old vs new accuracy before activating
- Previous models preserved for rollback capability

### Prediction Explanations
- Every prediction shows key influential words
- Feature importance extracted from Linear SVC coefficients
- Users see reasoning behind sentiment classification

### Admin Dashboard Enhancements
- Three tabs: Pending, Approved, Rejected reviews
- Match/mismatch indicators for approved reviews
- Manual "Retrain Model" button with progress indicator
- Model comparison modal before activation
- Full training history with version tracking
```

### 13.2 Create User Guide

Create `docs/USER_GUIDE.md`:

```markdown
# User Guide - Continuous Learning System

## For Regular Users

### Making Predictions
1. Go to homepage
2. Enter tweet text (10-500 characters)
3. Optionally select airline
4. Click "Analyze Sentiment"
5. Review prediction, confidence, and reasoning

### Understanding Results
- **Sentiment**: negative, neutral, or positive
- **Confidence**: 0-100% (higher = more certain)
- **Key Words**: Most influential words for this prediction
- **Reasoning**: Brief explanation of classification

## For Administrators

### Reviewing Submissions
1. Login to admin panel (password from .env)
2. Click "Pending" tab to see unreviewed predictions
3. For each submission:
   - Select true sentiment from dropdown
   - Optionally add comment
   - Click "Approve" or "Reject"

### Training the Model
1. Ensure 50+ approved reviews with true_sentiment
2. Click "Retrain Model" button
3. Wait for training to complete (~10-30 seconds)
4. Review accuracy comparison modal:
   - Old model accuracy
   - New model accuracy
   - Percentage change
5. Choose "Use New Model" or "Keep Current Model"

### Best Practices
- Always set true_sentiment for approved reviews
- Add comments explaining complex cases
- Retrain after every 50-100 new approvals
- Monitor match/mismatch rate in Approved tab
- Keep old model if new accuracy is lower
```

---

## Part 14: Testing Strategy

### 14.1 Create Test Script

Create `tests/test_continuous_learning.py`:

```python
import pytest
from src.data.database import (
    initialize_database, 
    insert_submission,
    count_approved_with_true_sentiment,
    fetch_training_data,
    save_model_version
)
from src.models.training import retrain_with_new_data

def test_database_schema():
    """Test that all tables exist"""
    initialize_database()
    # Add assertions to check tables exist

def test_submission_workflow():
    """Test full submission lifecycle"""
    # Insert test submission
    # Approve it
    # Verify it appears in training data

def test_retraining():
    """Test model retraining process"""
    # Create mock approved submissions
    # Trigger retraining
    # Verify new model is created

def test_feature_importance():
    """Test prediction explanations"""
    from src.models.training import explain_prediction, load_trained_model
    pipeline = load_trained_model()
    
    explanation = explain_prediction(pipeline, "Great flight!")
    assert explanation['sentiment'] is not None
    assert 'reasoning' in explanation
    assert len(explanation['top_features']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 14.2 Integration Test# Continuous Learning System Implementation Guide

## Overview
This guide implements a human-in-the-loop continuous learning system with:
- Complete review history display (unmarked shown first)
- Manual model retraining from dashboard
- Training progress indicators
- Model comparison & rollback capability
- Prediction explanations using top influential features

---

## Part 1: Database Schema Updates

### 1.1 Update `src/data/database.py`

Add new table for model versioning and update the submissions table:

```python
# Add this function after initialize_database()

def create_model_versions_table(cursor):
    """Create table to track model versions and metrics"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            version_number INT NOT NULL,
            accuracy DECIMAL(6,4) NOT NULL,
            precision_macro DECIMAL(6,4),
            recall_macro DECIMAL(6,4),
            f1_macro DECIMAL(6,4),
            training_samples INT NOT NULL,
            model_path VARCHAR(255) NOT NULL,
            metrics_path VARCHAR(255),
            is_active BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE KEY unique_version (version_number)
        )
    """)

def initialize_database() -> None:
    """Initialize database and required tables"""
    config = DatabaseConfig.from_env()
    
    # Create database if not exists
    conn = get_connection(include_database=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
    cursor.close()
    conn.close()
    
    # Create tables
    conn = get_connection()
    cursor = conn.cursor()
    
    # Original submissions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tweet_text TEXT NOT NULL,
            predicted_sentiment VARCHAR(20) NOT NULL,
            prediction_confidence DECIMAL(5,4),
            assigned_airline VARCHAR(50),
            true_sentiment VARCHAR(20),
            review_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
            admin_comment TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_review_status (review_status),
            INDEX idx_created_at (created_at)
        )
    """)
    
    # New model versions table
    create_model_versions_table(cursor)
    
    conn.commit()
    cursor.close()
    conn.close()
```

### 1.2 Add New Database Functions

```python
# Add these functions to src/data/database.py

def fetch_all_submissions_grouped(limit: int = 1000) -> dict:
    """Fetch submissions grouped by review status"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Pending submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'pending'
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    pending = cursor.fetchall()
    
    # Approved submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'approved'
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))
    approved = cursor.fetchall()
    
    # Rejected submissions
    cursor.execute("""
        SELECT * FROM submissions 
        WHERE review_status = 'rejected'
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))
    rejected = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {
        'pending': pending,
        'approved': approved,
        'rejected': rejected
    }

def count_approved_with_true_sentiment() -> int:
    """Count approved submissions with true_sentiment set"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM submissions 
        WHERE review_status = 'approved' 
        AND true_sentiment IS NOT NULL 
        AND true_sentiment != ''
    """)
    count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    return count

def fetch_training_data():
    """Fetch approved submissions with true_sentiment for retraining"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT tweet_text, true_sentiment as sentiment
        FROM submissions 
        WHERE review_status = 'approved' 
        AND true_sentiment IS NOT NULL 
        AND true_sentiment != ''
        AND tweet_text IS NOT NULL
        ORDER BY updated_at ASC
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_model_version(version_num: int, metrics: dict, model_path: str, 
                       training_samples: int, notes: str = None, is_active: bool = False):
    """Save model version metadata"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO model_versions 
        (version_number, accuracy, precision_macro, recall_macro, f1_macro, 
         training_samples, model_path, is_active, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        version_num,
        metrics.get('accuracy', 0),
        metrics.get('macro avg', {}).get('precision', 0),
        metrics.get('macro avg', {}).get('recall', 0),
        metrics.get('macro avg', {}).get('f1-score', 0),
        training_samples,
        model_path,
        is_active,
        notes
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_latest_model_version():
    """Get the latest active model version info"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM model_versions 
        WHERE is_active = TRUE
        ORDER BY version_number DESC 
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

def set_active_model(version_number: int):
    """Set a specific version as active, deactivating others"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Deactivate all models
    cursor.execute("UPDATE model_versions SET is_active = FALSE")
    
    # Activate selected model
    cursor.execute("""
        UPDATE model_versions 
        SET is_active = TRUE 
        WHERE version_number = %s
    """, (version_number,))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_all_model_versions():
    """Get all model versions ordered by version number"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM model_versions 
        ORDER BY version_number DESC
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results
```

---

## Part 2: Enhanced Training Module

### 2.1 Update `src/models/training.py`

Add functions for retraining and feature importance:

```python
# Add these imports at the top
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# Add this function after select_and_train_best()

def retrain_with_new_data(new_data_rows: list, model_name: str = "linear_svc") -> dict:
    """
    Retrain model with original dataset + new approved submissions
    
    Args:
        new_data_rows: List of dicts with 'tweet_text' and 'sentiment' keys
        model_name: Which model to retrain (linear_svc, logistic_regression, complement_nb)
    
    Returns:
        dict with training results and comparison metrics
    """
    from src.data.loaders import load_normalized_dataset
    
    # Load original dataset
    original_df = load_normalized_dataset()
    
    # Create dataframe from new data
    if not new_data_rows:
        raise ValueError("No new training data provided")
    
    new_df = pd.DataFrame(new_data_rows)
    new_df.columns = ['text', 'airline_sentiment']
    
    # Combine datasets
    combined_df = pd.concat([
        original_df[['text', 'airline_sentiment']], 
        new_df
    ], ignore_index=True)
    
    # Prepare dataset
    X, y = prepare_dataset(combined_df)
    
    if len(X) == 0:
        raise ValueError("No valid training data after preprocessing")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and train the specified model
    pipelines = build_pipelines()
    if model_name not in pipelines:
        raise ValueError(f"Model {model_name} not found. Choose from {list(pipelines.keys())}")
    
    pipeline = pipelines[model_name]
    
    logger.info(f"Retraining {model_name} with {len(X_train)} samples (including {len(new_data_rows)} new)")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'labels': pipeline.classes_.tolist(),
        'total_training_samples': len(X_train),
        'new_samples_added': len(new_data_rows),
        'test_samples': len(X_test)
    }
    
    logger.info(f"Retraining complete. New accuracy: {accuracy:.4f}")
    
    return pipeline, results

def save_model_version(pipeline, metrics: dict, version_number: int):
    """Save a versioned model"""
    models_dir = Path("models")
    version_dir = models_dir / f"version_{version_number}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = version_dir / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    
    # Save metrics
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved model version {version_number} to {version_dir}")
    
    return str(model_path), str(metrics_path)

def get_feature_importance(pipeline, text: str, top_n: int = 5) -> list:
    """
    Extract top N features that influenced the prediction
    
    Args:
        pipeline: Trained sklearn pipeline
        text: Input tweet text
        top_n: Number of top features to return
    
    Returns:
        List of tuples (feature_name, weight)
    """
    # Preprocess text
    cleaned = preprocess_text(text)
    
    # Get feature vector
    tfidf_vector = pipeline.named_steps['tfidf'].transform([cleaned])
    
    # Get classifier coefficients
    classifier = pipeline.named_steps['clf']
    
    # For Linear SVC
    if hasattr(classifier, 'coef_'):
        # Get prediction
        prediction = pipeline.predict([cleaned])[0]
        pred_idx = list(pipeline.classes_).index(prediction)
        
        # Get coefficients for predicted class
        coef = classifier.coef_[pred_idx]
        
        # Get feature names
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Get non-zero features from the vector
        nonzero_indices = tfidf_vector.nonzero()[1]
        
        # Calculate feature contributions
        contributions = []
        for idx in nonzero_indices:
            feature_name = feature_names[idx]
            feature_weight = coef[idx] * tfidf_vector[0, idx]
            contributions.append((feature_name, float(feature_weight)))
        
        # Sort by absolute weight
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:top_n]
    
    # For Naive Bayes or other models
    return []

def explain_prediction(pipeline, text: str) -> dict:
    """
    Generate human-readable explanation for prediction
    
    Returns:
        dict with sentiment, confidence, and reasoning
    """
    prediction = pipeline.predict([text])[0]
    
    # Get confidence if available
    confidence = None
    if hasattr(pipeline.named_steps['clf'], 'decision_function'):
        decision = pipeline.named_steps['clf'].decision_function([preprocess_text(text)])
        # Convert to pseudo-probability
        from scipy.special import expit
        probs = expit(decision)[0]
        pred_idx = list(pipeline.classes_).index(prediction)
        confidence = float(probs[pred_idx]) if len(probs.shape) > 0 else float(probs)
    
    # Get top features
    top_features = get_feature_importance(pipeline, text, top_n=5)
    
    # Build explanation
    if top_features:
        positive_features = [f for f, w in top_features if w > 0]
        negative_features = [f for f, w in top_features if w < 0]
        
        reasoning_parts = []
        if positive_features:
            reasoning_parts.append(f"Key indicators: {', '.join(positive_features[:3])}")
        
        explanation = {
            'sentiment': prediction,
            'confidence': confidence,
            'top_features': [f for f, w in top_features],
            'feature_weights': {f: w for f, w in top_features},
            'reasoning': ' | '.join(reasoning_parts) if reasoning_parts else "Based on overall text patterns"
        }
    else:
        explanation = {
            'sentiment': prediction,
            'confidence': confidence,
            'top_features': [],
            'feature_weights': {},
            'reasoning': "Classification based on learned patterns"
        }
    
    return explanation
```

---

## Part 3: Update Flask Routes

### 3.1 Update `app/routes.py`

Replace the entire file with enhanced version:

```python
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from app.forms import PredictionForm, AdminLoginForm
from src.data import database
from src.models.training import load_trained_model, explain_prediction, retrain_with_new_data, save_model_version
import os
from pathlib import Path
import json
import pandas as pd
import shutil
from datetime import datetime

bp = Blueprint('main', __name__)

# Global model variables
MODEL_PIPELINE = None
MODEL_LABELS = None
MODEL_METADATA = None

def load_model():
    """Load the trained model and metadata"""
    global MODEL_PIPELINE, MODEL_LABELS, MODEL_METADATA
    
    # Check for versioned model first
    latest_version = database.get_latest_model_version()
    
    if latest_version:
        model_path = Path(latest_version['model_path'])
        if model_path.exists():
            MODEL_PIPELINE = load_trained_model(str(model_path))
            MODEL_METADATA = latest_version
        else:
            # Fall back to default
            MODEL_PIPELINE = load_trained_model()
            MODEL_METADATA = None
    else:
        MODEL_PIPELINE = load_trained_model()
        MODEL_METADATA = None
    
    if MODEL_PIPELINE:
        MODEL_LABELS = MODEL_PIPELINE.classes_.tolist()
    
    # Load metrics
    metrics_file = Path("models/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            if MODEL_METADATA is None:
                MODEL_METADATA = metrics

# Airline options
AIRLINES = [
    ("", "-- Select Airline (Optional) --"),
    ("United", "United Airlines"),
    ("US Airways", "US Airways"),
    ("American", "American Airlines"),
    ("Southwest", "Southwest Airlines"),
    ("Delta", "Delta Airlines"),
    ("Virgin America", "Virgin America")
]

def ensure_static_assets():
    """Copy visualization assets to static directory"""
    static_img = Path("app/static/img")
    static_html = Path("app/static/html")
    static_img.mkdir(parents=True, exist_ok=True)
    static_html.mkdir(parents=True, exist_ok=True)
    
    vis_dir = Path("visualizations")
    if vis_dir.exists():
        for png_file in vis_dir.glob("*.png"):
            shutil.copy(png_file, static_img / png_file.name)
        for html_file in vis_dir.glob("*.html"):
            shutil.copy(html_file, static_html / html_file.name)

@bp.route("/", methods=["GET", "POST"])
def index():
    form = PredictionForm()
    form.set_airline_choices(AIRLINES)
    prediction_result = None
    recent = database.fetch_recent_submissions(limit=5)
    
    if form.validate_on_submit():
        text = form.tweet_text.data.strip()
        if not text:
            flash("Tweet text cannot be empty.", "danger")
            return render_template("index.html", form=form, prediction=prediction_result, recent_submissions=recent)
        
        # Get prediction with explanation
        explanation = explain_prediction(MODEL_PIPELINE, text)
        
        assigned_airline = form.airline.data or None
        
        # Insert to database
        database.insert_submission(
            tweet_text=text,
            predicted_sentiment=explanation['sentiment'],
            prediction_confidence=explanation['confidence'],
            assigned_airline=assigned_airline,
        )
        
        prediction_result = {
            'sentiment': explanation['sentiment'],
            'confidence': explanation['confidence'],
            'reasoning': explanation['reasoning'],
            'top_features': explanation['top_features'],
            'airline': assigned_airline
        }
        
        flash("Sentiment prediction saved for review.", "success")
        recent = database.fetch_recent_submissions(limit=5)
    
    return render_template("index.html", form=form, prediction=prediction_result, recent_submissions=recent)

@bp.route("/dashboard")
def dashboard():
    context = load_dashboard_context()
    
    # Add training data count
    approved_count = database.count_approved_with_true_sentiment()
    context['approved_training_count'] = approved_count
    
    # Add model version info
    context['current_model'] = MODEL_METADATA
    context['all_versions'] = database.get_all_model_versions()
    
    return render_template("dashboard.html", **context)

@bp.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST" and not session.get("is_admin"):
        form = AdminLoginForm()
        if form.validate_on_submit():
            from flask import current_app
            if form.password.data == current_app.config.get("ADMIN_PASSWORD"):
                session["is_admin"] = True
                flash("Login successful!", "success")
                return redirect(url_for("main.admin"))
            else:
                flash("Invalid password.", "danger")
        return render_template("admin_login.html", form=form)
    
    if not session.get("is_admin"):
        form = AdminLoginForm()
        return render_template("admin_login.html", form=form)
    
    # Fetch all submissions grouped by status
    submissions = database.fetch_all_submissions_grouped(limit=500)
    
    return render_template("admin.html", submissions=submissions)

@bp.route("/admin/review/<int:submission_id>", methods=["POST"])
def review_submission(submission_id: int):
    if not session.get("is_admin"):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("main.admin"))
    
    action = request.form.get("action")
    true_sentiment = request.form.get("true_sentiment")
    admin_comment = request.form.get("admin_comment")
    
    if action == "approve":
        status = "approved"
        flash(f"Submission {submission_id} approved.", "success")
    elif action == "reject":
        status = "rejected"
        flash(f"Submission {submission_id} rejected.", "info")
    else:
        flash("Invalid action.", "danger")
        return redirect(url_for("main.admin"))
    
    database.update_submission_status(
        submission_id=submission_id,
        status=status,
        true_sentiment=true_sentiment if true_sentiment else None,
        admin_comment=admin_comment if admin_comment else None
    )
    
    return redirect(url_for("main.admin"))

@bp.route("/admin/retrain", methods=["POST"])
def retrain_model():
    """Trigger model retraining with approved data"""
    if not session.get("is_admin"):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get training data
        new_data = database.fetch_training_data()
        
        if len(new_data) < 50:
            return jsonify({
                'error': f'Insufficient training data. Need at least 50 approved reviews with true_sentiment set. Currently have {len(new_data)}.'
            }), 400
        
        # Get current model accuracy
        old_accuracy = MODEL_METADATA.get('accuracy', 0) if MODEL_METADATA else 0
        
        # Retrain model
        new_pipeline, results = retrain_with_new_data(new_data, model_name="linear_svc")
        
        # Get next version number
        all_versions = database.get_all_model_versions()
        next_version = max([v['version_number'] for v in all_versions], default=0) + 1
        
        # Save as new version
        model_path, metrics_path = save_model_version(new_pipeline, results, next_version)
        
        # Save to database
        database.save_model_version(
            version_num=next_version,
            metrics=results['classification_report'],
            model_path=model_path,
            training_samples=results['total_training_samples'],
            notes=f"Retrained with {results['new_samples_added']} new samples",
            is_active=False  # Don't activate yet, let admin choose
        )
        
        new_accuracy = results['accuracy']
        accuracy_diff = new_accuracy - old_accuracy
        
        return jsonify({
            'success': True,
            'version': next_version,
            'old_accuracy': float(old_accuracy),
            'new_accuracy': float(new_accuracy),
            'accuracy_difference': float(accuracy_diff),
            'improvement_percentage': float((accuracy_diff / old_accuracy * 100) if old_accuracy > 0 else 0),
            'training_samples': results['total_training_samples'],
            'new_samples': results['new_samples_added'],
            'message': f'Model version {next_version} trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/admin/model/<int:version>/activate", methods=["POST"])
def activate_model(version: int):
    """Activate a specific model version"""
    if not session.get("is_admin"):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        database.set_active_model(version)
        
        # Reload model globally
        load_model()
        
        flash(f"Model version {version} is now active.", "success")
        return jsonify({'success': True, 'message': f'Model version {version} activated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/logout")
def logout():
    session.pop("is_admin", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("main.index"))

def load_dashboard_context() -> dict:
    """Load all data needed for dashboard"""
    processed_dir = Path("data/processed")
    
    context = {
        "sentiment_summary": [],
        "airline_volume": [],
        "negative_reasons": [],
        "model_accuracy": 0,
        "model_name": "Unknown",
        "visuals": {}
    }
    
    # Load sentiment distribution
    sentiment_file = processed_dir / "sentiment_distribution.csv"
    if sentiment_file.exists():
        df = pd.read_csv(sentiment_file)
        context["sentiment_summary"] = df.to_dict('records')
    
    # Load airline volume
    airline_file = processed_dir / "airline_tweet_volume.csv"
    if airline_file.exists():
        df = pd.read_csv(airline_file)
        context["airline_volume"] = df.to_dict('records')
    
    # Load negative reasons
    reasons_file = processed_dir / "top_negative_reasons.csv"
    if reasons_file.exists():
        df = pd.read_csv(reasons_file)
        context["negative_reasons"] = df.to_dict('records')
    
    # Load model metrics
    metrics_file = Path("models/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            context["model_accuracy"] = metrics.get("accuracy", 0)
            context["model_name"] = metrics.get("model_name", "Unknown")
    
    # Visual assets
    context["visuals"] = {
        "sentiment": "/static/img/sentiment_distribution.png",
        "timeline": "/static/img/tweet_volume_timeline.png",
        "negative": "/static/img/top_negative_reasons.png",
        "heatmap": "/static/img/airline_sentiment_heatmap.png",
        "wordcloud": "/static/img/wordcloud_negative.png",
        "model_cm": "/static/img/model_confusion_matrix.png",
        "sentiment_html": "/static/html/sentiment_distribution.html",
    }
    
    return context

# Initialize on import
ensure_static_assets()
load_model()
```

---

## Part 4: Update Templates

### 4.1 Update `app/templates/admin.html`

Replace with enhanced version showing all submissions:

```html
{% extends "base.html" %}

{% block title %}Admin Panel - Review Queue{% endblock %}
```

### 4.2 Update `app/templates/index.html`

Add prediction explanation display:

```html
{% extends "base.html" %}

{% block title %}Predict Sentiment{% endblock %}

{% block content %}
<div class="hero">
    <h1>Twitter Airline Sentiment Analyzer</h1>
    <p>Analyze sentiment of airline-related tweets with AI-powered predictions</p>
</div>

<div class="container">
    <div class="card">
        <h2>Enter a Tweet</h2>
        <form method="POST" action="{{ url_for('main.index') }}">
            {{ form.hidden_tag() }}
            
            <div class="form-group">
                {{ form.tweet_text.label }}
                {{ form.tweet_text(class="form-control", rows=4, placeholder="Example: Flight delayed again! Terrible customer service.") }}
                {% if form.tweet_text.errors %}
                    <span class="error">{{ form.tweet_text.errors[0] }}</span>
                {% endif %}
            </div>
            
            <div class="form-group">
                {{ form.airline.label }}
                {{ form.airline(class="form-control") }}
            </div>
            
            <button type="submit" class="btn btn-primary">Analyze Sentiment</button>
        </form>
    </div>

    {% if prediction %}
    <div class="card prediction-result">
        <h2>Prediction Result</h2>
        <div class="result-grid">
            <div class="result-item">
                <h3>Sentiment</h3>
                <p class="sentiment-badge sentiment-{{ prediction.sentiment }}">
                    {{ prediction.sentiment|upper }}
                </p>
            </div>
            
            <div class="result-item">
                <h3>Confidence</h3>
                <p class="confidence-value">
                    {% if prediction.confidence %}
                        {{ "%.1f"|format(prediction.confidence * 100) }}%
                    {% else %}
                        N/A
                    {% endif %}
                </p>
            </div>
        </div>
        
        {% if prediction.reasoning %}
        <div class="explanation-section">
            <h3>Why This Prediction?</h3>
            <p class="reasoning-text">{{ prediction.reasoning }}</p>
            
            {% if prediction.top_features %}
            <div class="features-list">
                <h4>Key Words Detected:</h4>
                <ul>
                    {% for feature in prediction.top_features[:5] %}
                    <li><code>{{ feature }}</code></li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if prediction.airline %}
        <p class="airline-info"><strong>Airline:</strong> {{ prediction.airline }}</p>
        {% endif %}
        
        <p class="review-note">This prediction has been saved for admin review.</p>
    </div>
    {% endif %}

    {% if recent_submissions %}
    <div class="card">
        <h2>Recent Predictions</h2>
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Tweet</th>
                        <th>Predicted</th>
                        <th>Confidence</th>
                        <th>Status</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    {% for sub in recent_submissions %}
                    <tr>
                        <td class="tweet-snippet">{{ sub.tweet_text[:80] }}...</td>
                        <td>
                            <span class="sentiment-badge sentiment-{{ sub.predicted_sentiment }}">
                                {{ sub.predicted_sentiment }}
                            </span>
                        </td>
                        <td>
                            {% if sub.prediction_confidence %}
                                {{ "%.1f"|format(sub.prediction_confidence * 100) }}%
                            {% else %}
                                N/A
                            {% endif %}
                        </td>
                        <td>
                            <span class="status-badge status-{{ sub.review_status }}">
                                {{ sub.review_status }}
                            </span>
                        </td>
                        <td>{{ sub.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}
</div>

<style>
.prediction-result {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-top: 20px;
}

.prediction-result h2,
.prediction-result h3,
.prediction-result h4 {
    color: white;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.result-item {
    text-align: center;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}

.result-item h3 {
    font-size: 1em;
    margin-bottom: 10px;
    opacity: 0.9;
}

.sentiment-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 25px;
    font-size: 1.5em;
    font-weight: bold;
}

.confidence-value {
    font-size: 2em;
    font-weight: bold;
    margin: 0;
}

.explanation-section {
    margin-top: 30px;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}

.reasoning-text {
    font-size: 1.1em;
    line-height: 1.6;
    margin: 15px 0;
}

.features-list {
    margin-top: 20px;
}

.features-list h4 {
    font-size: 1em;
    margin-bottom: 10px;
}

.features-list ul {
    list-style: none;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.features-list li {
    background: rgba(255,255,255,0.2);
    padding: 5px 15px;
    border-radius: 15px;
}

.features-list code {
    color: white;
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
}

.airline-info {
    margin-top: 20px;
    font-size: 1.1em;
}

.review-note {
    margin-top: 15px;
    font-style: italic;
    opacity: 0.8;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: bold;
}

.status-pending {
    background-color: #ffc107;
    color: #000;
}

.status-approved {
    background-color: #28a745;
    color: white;
}

.status-rejected {
    background-color: #dc3545;
    color: white;
}

.tweet-snippet {
    max-width: 300px;
}
</style>
{% endblock %}

{% block content %}
<div class="admin-container">
    <div class="admin-header">
        <h1>Admin Review Panel</h1>
        <a href="{{ url_for('main.logout') }}" class="btn btn-secondary">Logout</a>
    </div>

    <!-- Training Section -->
    <div class="card training-section">
        <h2>Model Training</h2>
        <div class="training-stats">
            <p><strong>Approved Reviews Available for Training:</strong> 
                {{ submissions.approved|selectattr('true_sentiment')|list|length }}</p>
            <p class="help-text">Minimum 50 reviews with true sentiment required for retraining</p>
        </div>
        
        <button id="retrainBtn" class="btn btn-primary" onclick="startRetraining()">
            <span id="retrainText">Retrain Model</span>
            <span id="retrainSpinner" class="spinner" style="display:none;">⟳ Training...</span>
        </button>
        
        <div id="trainingResults" style="display:none; margin-top: 20px;">
            <!-- Results will be injected here -->
        </div>
    </div>

    <!-- Model Comparison Modal -->
    <div id="comparisonModal" class="modal" style="display:none;">
        <div class="modal-content">
            <h2>Model Training Complete</h2>
            <div id="comparisonDetails"></div>
            <div class="modal-actions">
                <button onclick="activateNewModel()" class="btn btn-primary">Use New Model</button>
                <button onclick="keepOldModel()" class="btn btn-secondary">Keep Current Model</button>
            </div>
        </div>
    </div>

    <!-- Submissions Tabs -->
    <div class="submissions-tabs">
        <button class="tab-btn active" onclick="showTab('pending')">
            Pending ({{ submissions.pending|length }})
        </button>
        <button class="tab-btn" onclick="showTab('approved')">
            Approved ({{ submissions.approved|length }})
        </button>
        <button class="tab-btn" onclick="showTab('rejected')">
            Rejected ({{ submissions.rejected|length }})
        </button>
    </div>

    <!-- Pending Submissions -->
    <div id="pending-tab" class="tab-content active">
        <h2>Pending Reviews</h2>
        {% if submissions.pending %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>Confidence</th>
                            <th>Airline</th>
                            <th>Date</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.pending %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>
                                {% if submission.prediction_confidence %}
                                    {{ "%.1f"|format(submission.prediction_confidence * 100) }}%
                                {% else %}
                                    N/A
                                {% endif %}
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td>{{ submission.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>
                                <form method="POST" action="{{ url_for('main.review_submission', submission_id=submission.id) }}" class="inline-form">
                                    <select name="true_sentiment" required>
                                        <option value="">True Sentiment</option>
                                        <option value="negative">Negative</option>
                                        <option value="neutral">Neutral</option>
                                        <option value="positive">Positive</option>
                                    </select>
                                    <input type="text" name="admin_comment" placeholder="Comment (optional)" class="comment-input">
                                    <button type="submit" name="action" value="approve" class="btn btn-sm btn-success">Approve</button>
                                    <button type="submit" name="action" value="reject" class="btn btn-sm btn-danger">Reject</button>
                                </form>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No pending submissions</p>
        {% endif %}
    </div>

    <!-- Approved Submissions -->
    <div id="approved-tab" class="tab-content">
        <h2>Approved Reviews</h2>
        {% if submissions.approved %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>True Sentiment</th>
                            <th>Match</th>
                            <th>Airline</th>
                            <th>Comment</th>
                            <th>Reviewed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.approved %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>
                                {% if submission.true_sentiment %}
                                    <span class="sentiment-badge sentiment-{{ submission.true_sentiment }}">
                                        {{ submission.true_sentiment }}
                                    </span>
                                {% else %}
                                    <span class="text-muted">Not set</span>
                                {% endif %}
                            </td>
                            <td>
                                {% if submission.true_sentiment %}
                                    {% if submission.predicted_sentiment == submission.true_sentiment %}
                                        <span class="badge-success">✓ Match</span>
                                    {% else %}
                                        <span class="badge-error">✗ Mismatch</span>
                                    {% endif %}
                                {% else %}
                                    -
                                {% endif %}
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td class="comment-cell">{{ submission.admin_comment or '-' }}</td>
                            <td>{{ submission.updated_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No approved submissions yet</p>
        {% endif %}
    </div>

    <!-- Rejected Submissions -->
    <div id="rejected-tab" class="tab-content">
        <h2>Rejected Reviews</h2>
        {% if submissions.rejected %}
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Tweet</th>
                            <th>Predicted</th>
                            <th>Airline</th>
                            <th>Comment</th>
                            <th>Rejected</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for submission in submissions.rejected %}
                        <tr>
                            <td>{{ submission.id }}</td>
                            <td class="tweet-cell">{{ submission.tweet_text[:100] }}...</td>
                            <td>
                                <span class="sentiment-badge sentiment-{{ submission.predicted_sentiment }}">
                                    {{ submission.predicted_sentiment }}
                                </span>
                            </td>
                            <td>{{ submission.assigned_airline or '-' }}</td>
                            <td class="comment-cell">{{ submission.admin_comment or '-' }}</td>
                            <td>{{ submission.updated_at.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <p class="empty-state">No rejected submissions</p>
        {% endif %}
    </div>
</div>

<script>
let newModelVersion = null;

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

async function startRetraining() {
    const btn = document.getElementById('retrainBtn');
    const text = document.getElementById('retrainText');
    const spinner = document.getElementById('retrainSpinner');
    const results = document.getElementById('trainingResults');
    
    // Disable button and show spinner
    btn.disabled = true;
    text.style.display = 'none';
    spinner.style.display = 'inline';
    results.style.display = 'none';
    
    try {
        const response = await fetch('/admin/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.error) {
            results.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
            results.style.display = 'block';
        } else {
            newModelVersion = data.version;
            showComparisonModal(data);
        }
    } catch (error) {
        results.innerHTML = `<div class="alert alert-error">Training failed: ${error.message}</div>`;
        results.style.display = 'block';
    } finally {
        // Re-enable button
        btn.disabled = false;
        text.style.display = 'inline';
        spinner.style.display = 'none';
    }
}

function showComparisonModal(data) {
    const modal = document.getElementById('comparisonModal');
    const details = document.getElementById('comparisonDetails');
    
    const improvementText = data.accuracy_difference > 0 
        ? `<span style="color: green;">+${(data.improvement_percentage).toFixed(2)}% improvement</span>`
        : `<span style="color: red;">${(data.improvement_percentage).toFixed(2)}% decrease</span>`;
    
    details.innerHTML = `
        <div class="comparison-grid">
            <div class="comparison-item">
                <h3>Current Model</h3>
                <p class="metric-value">${(data.old_accuracy * 100).toFixed(2)}%</p>
                <p class="metric-label">Accuracy</p>
            </div>
            <div class="comparison-item">
                <h3>New Model (v${data.version})</h3>
                <p class="metric-value">${(data.new_accuracy * 100).toFixed(2)}%</p>
                <p class="metric-label">Accuracy</p>
            </div>
        </div>
        <div class="comparison-summary">
            <p><strong>Change:</strong> ${improvementText}</p>
            <p><strong>Training samples:</strong> ${data.training_samples} (${data.new_samples} new)</p>
            <p class="help-text">Choose whether to activate the new model or keep using the current one.</p>
        </div>
    `;
    
    modal.style.display = 'flex';
}

async function activateNewModel() {
    try {
        const response = await fetch(`/admin/model/${newModelVersion}/activate`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('New model activated! The page will reload.');
            location.reload();
        } else {
            alert('Failed to activate model: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function keepOldModel() {
    document.getElementById('comparisonModal').style.display = 'none';
    const results = document.getElementById('trainingResults');
    results.innerHTML = `<div class="alert alert-info">New model version ${newModelVersion} saved but not activated. Current model remains in use.</div>`;
    results.style.display = 'block';
}
</script>

<style>
.admin-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.admin-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.training-section {
    margin-bottom: 30px;
    padding: 20px;
}

.training-stats {
    margin-bottom: 15px;
}

.help-text {
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
}

.spinner {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.submissions-tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
}

.tab-btn {
    padding: 10px 20px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 16px;
    color: #666;
    border-bottom: 3px solid transparent;
    transition: all 0.3s;
}

.tab-btn:hover {
    color: #0a3d62;
}

.tab-btn.active {
    color: #0a3d62;
    border-bottom-color: #0a3d62;
    font-weight: bold;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.data-table-container {
    overflow-x: auto;
}

.tweet-cell {
    max-width: 300px;
    word-wrap: break-word;
}

.comment-cell {
    max-width: 200px;
    font-style: italic;
    color: #666;
}

.sentiment-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: bold;
}

.sentiment-negative {
    background-color: #fee;
    color: #c00;
}

.sentiment-neutral {
    background-color: #ffc;
    color: #660;
}

.sentiment-positive {
    background-color: #efe;
    color: #060;
}

.badge-success {
    color: green;
    font-weight: bold;
}

.badge-error {
    color: red;
    font-weight: bold;
}

.inline-form {
    display: flex;
    gap: 5px;
    align-items: center;
}

.comment-input {
    padding: 4px 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 0.9em;
}

.btn-sm {
    padding: 4px 12px;
    font-size: 0.85em;
}

.btn-success {
    background-color: #28a745;
    color: white;
}

.btn-danger {
    background-color: #dc3545;
    color: white;
}

.empty-state {
    text-align: center;
    padding: 40px;
    color: #999;
    font-style: italic;
}

.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal-content {
    background: white;
    padding: 30px;
    border-radius: 8px;
    max-width: 600px;
    width: 90%;
}

.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}

.comparison-item {
    text-align: center;
    padding: 20px;
    border: 2px solid #ddd;
    border-radius: 8px;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #0a3d62;
    margin: 10px 0;
}

.metric-label {
    color: #666;
    font-size: 0.9em;
}

.comparison-summary {
    margin-top: 20px;
    padding: 15px;
    background: #f5f5f5;
    border-radius: 4px;
}

.modal-actions {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 20px;
}

.alert {
    padding: 15px;
    border-radius: 4px;
    margin-top: 15px;
}

.alert-error {
    background-color: #fee;
    color: #c00;
    border: 1px solid #fcc;
}

.alert-info {
    background-color: #e7f3ff;
    color: #004085;
    border: 1px solid #b8daff;
}

.text-muted {
    color: #999;
}
</style>
{% endblock %}