# Project Summary

## Highlights
- Data pipeline processed 14640 tweets with validated schema and normalized outputs.
- Negative sentiment accounts for 62.69% of tweets, while positive sentiment is 16.14%.
- Support Vector Machine (`linear_svc`) accuracy **0.763** (primary dashboard model).
- Complement Naive Bayes accuracy **0.759**, offering a probabilistic baseline for comparative analysis.
- Association rules, visualization assets, and bilingual reports generated for stakeholder consumption.

## Generated Artifacts
- Reports: `analysis_results.md`, `laporan_project.md`, `python_notebook_documentation.md`, `presentation_outline.md`, `model_performance.md`, `data_quality_report.md`, `sentiment_analysis.md`, `visualization_summary.md`.
- Processed data tables: sentiment distributions, airline volumes, negative reason breakdowns, association rules, confusion matrix.
- Visuals: sentiment distribution (PNG + HTML), airline heatmap, timeline, top negative reasons, word cloud, model confusion matrix.
- Serialized model: `models/sentiment_pipeline.joblib` plus metrics JSON.
- Database: MySQL `airline_sentiment` with `submissions` table supporting admin review workflow.

## Automated Testing Summary
- Puppeteer suite executed 7 end-to-end tests. All passed.
  - [PASS] loads home page
  - [PASS] submits sentiment prediction
  - [PASS] recent submissions shows latest entry
  - [PASS] dashboard displays key visuals
  - [PASS] admin login rejects wrong password
  - [PASS] admin login succeeds with valid password
  - [PASS] admin can approve pending submission

## How to Run the Application
1. `python -m venv venv` (first time) and `venv/Script/activate` (Windows) or `source venv/bin/activate` (macOS/Linux).
2. `pip install -r requirements.txt` and `npm install`.
3. Ensure MySQL via XAMPP is running; credentials in `.env`.
4. Initialize artifacts (optional) via `python -m src.pipeline.run_phase --phase phase1` ... `phase8`.
5. Launch web app: `python run.py` and open http://127.0.0.1:5000 .
6. Run automated UI tests: `npm test`.

## Next Recommendations
- Integrate real-time Twitter streaming API for live data ingestion.
- Add user authentication and audit logging for admin actions.
- Deploy production-ready WSGI server (Gunicorn/Nginx) with HTTPS for external demos.
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
