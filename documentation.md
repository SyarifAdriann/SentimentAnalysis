# Twitter Airline Sentiment Analysis & Web Dashboard – Comprehensive Technical Documentation

_Last updated: 2025-10-17_

Authoring context: Automated synthesis based on repository state located at `C:\xampp\htdocs\SentimentAnalysis`.

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Project Context and Objectives](#2-project-context-and-objectives)
- [3. System Architecture Overview](#3-system-architecture-overview)
- [4. Data Sources and Governance](#4-data-sources-and-governance)
- [5. Dataset Profiling and Quality Metrics](#5-dataset-profiling-and-quality-metrics)
- [6. Data Processing Pipeline (Phases 1-3)](#6-data-processing-pipeline-phases-1-3)
- [7. Feature Engineering and NLP Normalisation](#7-feature-engineering-and-nlp-normalisation)
- [8. Exploratory Data Analysis Findings](#8-exploratory-data-analysis-findings)
- [9. Sentiment Pattern Mining and Association Rules](#9-sentiment-pattern-mining-and-association-rules)
- [10. Machine Learning Pipeline (Phase 4)](#10-machine-learning-pipeline-phase-4)
- [11. Model Evaluation and Diagnostics](#11-model-evaluation-and-diagnostics)
- [12. Model Artefacts and Versioning](#12-model-artefacts-and-versioning)
- [13. Visualisation Suite (Phase 5)](#13-visualisation-suite-phase-5)
- [14. Web Application Architecture](#14-web-application-architecture)
- [15. API and Route Documentation](#15-api-and-route-documentation)
- [16. Front-end Implementation](#16-front-end-implementation)
- [17. Database Schema and Persistence Layer](#17-database-schema-and-persistence-layer)
- [18. Infrastructure and Deployment Workflow](#18-infrastructure-and-deployment-workflow)
- [19. Technology Stack and Dependency Inventory](#19-technology-stack-and-dependency-inventory)
- [20. Project Structure and File Inventory](#20-project-structure-and-file-inventory)
- [21. Automated Testing and Quality Assurance](#21-automated-testing-and-quality-assurance)
- [35. Appendix: Deployment Runbook](#35-appendix-deployment-runbook)

---

## 1. Executive Summary
The Twitter Airline Sentiment Analysis & Web Dashboard project ingests the 14,640-row `Tweets.csv` corpus released by Crowdflower to characterise public perception of six US airlines during February 2015. The repository at `C:\xampp\htdocs\SentimentAnalysis` implements an end-to-end analytics platform that normalises the raw CSV dataset, quantifies sentiment distributions, mines negative reason patterns, trains a supervised text classifier, and surfaces insights through a Flask-based dashboard backed by a MySQL persistence layer. A complementary Puppeteer test harness validates the critical user journeys, ensuring the sentiment prediction workflow, analytics experience, and administrative review pipeline operate as designed.

The current build (timestamped 2025-10-17 in `logs/pipeline.log`) completes the following milestones:
- **Data readiness**: Phase 1 normalises the raw CSV into both Parquet (1.50 MB) and CSV (3.27 MB) artefacts, producing a data quality snapshot that records 36 duplicate rows and missing value concentrations for `negativereason` (5,462 nulls) and `tweet_coord` (13,621 nulls).
- **Analytical insights**: Phase 2 generates descriptive statistics showing that negative sentiment dominates at 62.69% (9,178 tweets), followed by neutral (21.17%) and positive (16.14%). United Airlines receives the highest tweet volume (3,822), with Customer Service Issues driving 2,910 negative mentions.
- **Predictive capability**: Phase 4 benchmarks Logistic Regression, Linear SVC, and Complement Naive Bayes pipelines, selecting Linear SVC with an accuracy of 0.7626, macro-average F1 of 0.6986, and detailed confusion matrix persisted to `models/model_metrics.json` and `data/processed/confusion_matrix.csv`.
- **Visual storytelling**: Phase 5 exports 13 static PNG assets plus an interactive Plotly pie chart, covering sentiment distribution, temporal volume, heatmaps of airline-versus-reason intensity, and class-specific word clouds.
- **Operational interface**: Phase 8 delivers a tri-page Flask application featuring (1) a prediction form with confidence scoring, (2) a dashboard embedding static and interactive charts, and (3) an admin queue backed by the MySQL `submissions` table defined in `src/data/database.py`.
- **Automated quality gates**: Phase 10’s Puppeteer suite (`tests/puppeteer/run-tests.js`) runs seven UI flows, including authentication checks and admin approval. Results stored in `logs/puppeteer_results.json` confirm a passing state on 2025-10-17T03:03:57Z.

Collectively, the artefacts position the project as a reproducible academic submission and a reference implementation for sentiment-driven service intelligence. Remaining gaps include aligning generated visuals with the stricter file naming conventions listed in `rules.md`, closing the discrepancy between the expected ten Puppeteer tests versus the seven implemented, and extending the deployment documentation for production-grade HTTP serving.


**2025-10 Continuous Learning Update:** the application now records model versions in MySQL, exposes manual Celery-backed retraining from the admin dashboard, rate-limits public predictions via Flask-Limiter, and publishes a migration playbook (`migrationsteps.md`) to streamline cross-machine setup. A lightweight pytest suite complements the existing Puppeteer E2E coverage.
## 2. Project Context and Objectives
The repository follows the multi-phase mandate set out in `rules.md`, which maps eleven sequential phases spanning environment verification through final launch. Key drivers include:
- Enabling educators to audit every architectural layer without inspecting raw source files.
- Providing analysts with traceable statistics—from raw dataset profiling to model evaluation—anchored on verifiable artefacts inside `data/processed/`, `reports/`, and `models/`.
- Supporting end users (students or stakeholders) with an interactive dashboard and review workflow capable of logging human feedback for model governance.

Business questions answered by the system:
1. **Sentiment landscape** – Quantify how public discourse splits across negative, neutral, and positive sentiment for each airline, enabling prioritisation of customer experience interventions.
2. **Pain point identification** – Surface the most frequent negative reasons and their airline associations to feed operational remediation plans.
3. **Predictive triage** – Employ supervised ML to score new tweets, persist predictions for audit, and allow human-in-the-loop review via the admin queue.
4. **Experience validation** – Ensure the UI flows required for prediction, insight consumption, and admin governance remain reliable through automated browser regression tests.

Project objectives translate into deliverables across the phases. For example, Phase 3’s association rules underpin stakeholder briefings on intertwined causes, while Phase 4’s serialized model (`models/sentiment_pipeline.joblib`, 368 KB) is the linchpin for the Flask inference endpoint. The documentation herein expands each phase-specific output, records quantitative metrics, and captures design intent to satisfy the “single source of truth” requirement.
## 3. System Architecture Overview
The architecture combines offline analytics pipelines with an online inference service. Artefacts and execution traces demonstrate the following layered design:

```
+------------------+        +---------------------+        +--------------------+
|  data/raw/Tweets |  -->   |  Phase 1-3 ETL &    |  -->   |  data/processed/   |
|  .csv (14,640 rows)|      |  analytics routines |        |  (CSV, Parquet,    |
+------------------+        |  (src/data,         |        |  summaries, rules) |
                             |  src/analysis,      |        +---------+----------+
                             |  src/pipeline)      |                  |
                             +----------+----------+                  |
                                        |                             v
                                        v                +--------------------------+
                             +----------+----------+     |  Phase 4 model training  |
                             |  src/models/training | -->|  (Linear SVC pipeline,   |
                             |  (TF-IDF + SVC)      |    |  joblib + metrics JSON)  |
                             +----------+----------+     +--------------------------+
                                        |
                                        v
+----------------------+      +---------+----------+      +--------------------------+
| Flask UI (app/       |<-----| Model inference    |<-----|  models/sentiment_       |
| routes.py, templates)|      | (training.load_    |      |  pipeline.joblib         |
| Serves HTML, forms,  |      | trained_model)     |      +--------------------------+
| dashboards)          |      +---------+----------+
|                      |                |
|                      |                v
|                      |      +---------+----------+
|                      |      | MySQL `submissions`|
|                      |      | table via src/data |
|                      |      | /database.py       |
+----------+-----------+      +---------+----------+
           ^                               |
           |                               v
           |                 +-------------+---------------+
           |                 | Admin review queue &        |
           |                 | Puppeteer regression tests  |
           |                 +-----------------------------+
```

**Execution flow narrative**:
1. `scripts/master_pipeline.py` orchestrates phase execution, delegating to `src/pipeline/run_phase.py`. Phases 1–3 serialise cleaned datasets, EDA outputs, and association rules into `data/processed/` and `reports/`.
2. Phase 4 (`run_phase4`) invokes `training.select_and_train_best`, persisting `models/model_metrics.json`, `models/sentiment_pipeline.joblib`, and confusion matrices reused by both the dashboard and offline reports.
3. Phase 5 (plus `scripts/generate_additional_visuals.py`) produces PNG/HTML assets copied into `app/static/` by `app/routes.ensure_static_assets` so that Flask can serve them without recomputing graphics.
4. The Flask factory in `app/__init__.py` loads environment secrets from `.env`, prepares `PredictionForm` and `AdminLoginForm`, and registers the blueprint defined in `app/routes.py`.
5. Runtime inference uses the joblib serialised Linear SVC pipeline. Predictions and optional confidences are written to the MySQL `submissions` table via `src/data/database.insert_submission`, enabling audit and admin workflows.
6. Puppeteer tests in `tests/puppeteer/run-tests.js` spin up the Flask app (`run.py`) in a child process, execute browser interactions, and log pass/fail entries into `logs/puppeteer_results.json`.

This structure allows offline analytics and online services to evolve semi-independently while sharing exported artefacts through the filesystem.

## 4. Data Sources and Governance
- **Primary dataset**: `data/raw/Tweets.csv` (3.26 MB, 14,640 tweets) is the authoritative input. It retains original columns such as `tweet_id`, `airline_sentiment`, `negativereason`, `tweet_created`, and `user_timezone`. A sample record (tweet 570306133677760513) confirms column naming and timestamp formatting.
- **Processed derivatives**: Phase 1 writes `data/processed/tweets_normalized.parquet` (1.50 MB) and `tweets_normalized.csv` (3.44 MB) to accelerate downstream Pandas operations and maintain reproducible exports. Additional CSVs (e.g., `sentiment_distribution.csv`, `airline_tweet_volume.csv`, `negative_reason_by_airline.csv`) capture summarised metrics for the dashboard.
- **Association artefacts**: `data/processed/association_rules.csv` holds 25 rules derived from MLxtend’s Apriori algorithm. Each rule stores antecedents, consequents, support, confidence, lift, leverage, and conviction. In this dataset, all rules align with perfect confidence (1.0), highlighting the prevalence of negative sentiment when a specific airline or negative reason is involved.
- **Reporting source**: `data/processed/phase1_summary.json` includes machine-readable snapshots of row counts, duplicates, and missing values, providing a canonical reference for data quality statements reused in this documentation.
- **Backups**: `backups/linear_svc/` mirrors processed data, reports, models, and visualisations, demonstrating a manual snapshot taken post Phase 4. It ensures reproducibility if working assets are corrupted.
- **Governance artefacts**: `.env` stores local credentials (`DB_HOST`, `DB_USER`, `DB_NAME`, `ADMIN_PASSWORD`), while `rules.md` documents mandatory deliverables, constraints, and nomenclature expectations across all phases. `PROJECT_SUMMARY.md` serves as a lightweight stakeholder digest and is regenerated when phases complete.
- **Data retention**: No automated purging is implemented. The MySQL table accumulates submissions until truncated manually. Processed CSV/Parquet files can be regenerated idempotently by rerunning phases, as logged in `logs/pipeline.log`.

Ownership and stewardship are implicitly assigned to the project maintainer executing `scripts/master_pipeline.py`. The dataset is public; however, `.env` clarifies default credentials, and documentation should remind operators to rotate the admin password for production contexts.
## 5. Dataset Profiling and Quality Metrics
### 5.1 Structural overview
- **Rows**: 14,640 (`phase1_summary.json` → `row_count`).
- **Columns**: 15 (covering identifiers, sentiment labels, textual content, geospatial hints, and temporal metadata).
- **Duplicate rows**: 36 (reported as `duplicate_rows`). No deduplication is persisted in-place; the analytics pipeline implicitly works on the raw counts, which is acceptable because the duplicate volume represents just 0.25% of the corpus.

### 5.2 Missing value assessment
The following abridged table summarises null prevalence per column (values taken directly from `phase1_summary.json`):

| Column                     | Missing | Note |
|---------------------------|--------:|------|
| `negativereason`          | 5,462   | Most tweets do not provide a structured reason; downstream association rules therefore operate on the 9,178 populated entries.
| `negativereason_confidence` | 4,118 | Confidence scores missing when reasons are blank; not used for modelling.
| `tweet_coord`             | 13,621  | Geo-coordinates largely absent; visualisations do not attempt geospatial mapping.
| `tweet_location`          | 4,733   | Free-text location missing or ambiguous.
| `user_timezone`           | 4,820   | Overlaps with missing location data; timezone not leveraged in current analyses.
| `airline_sentiment_gold`  | 14,600  | Gold labels rarely provided; pipeline ignores this field.
| `negativereason_gold`     | 14,608  | Similar to above.
| `text`, `airline_sentiment`, `airline` | 0 | Core analytical fields fully populated.

### 5.3 Numeric distribution snapshots
- `airline_sentiment_confidence`: mean 0.9002, std 0.1628, min 0.3350, max 1.0.
- `negativereason_confidence`: mean 0.6383, std 0.3304, min 0.0, max 1.0.
- `retweet_count`: mean 0.0827, std 0.7458, min 0, max 44. Values confirm the dataset captures mostly original tweets rather than viral content.

### 5.4 Temporal coverage
`data/processed/timeline_daily.csv` indicates tweets range from 2015-02-16 to 2015-02-24 (Pacific Time). Activity spikes on 2015-02-22 (3,079 tweets) and 2015-02-23 (3,028 tweets), aligning with newsworthy airline disruptions during that period.

## 6. Data Processing Pipeline (Phases 1-3)
### 6.1 Phase 1 – Loading and validation
`src/pipeline/run_phase.py:run_phase1` orchestrates the following steps:
1. `loaders.load_raw_dataset()` reads `data/raw/Tweets.csv` and returns a Pandas DataFrame.
2. `loaders.detect_column_issues` verifies schema alignment with the expected column tuple.
3. `loaders.normalize_dataframe` converts `tweet_created` to pandas datetime, coerces `tweet_id` to string, and normalises `airline_sentiment` to lowercase.
4. `validation.build_snapshot` produces aggregate statistics, serialised to `reports/data_quality_report.md` and `data/processed/phase1_summary.json`.
5. Parquet and CSV exports of the normalised dataset are written to `data/processed/` to support future phases and backup replication.

Log excerpt (`logs/pipeline.log` at 09:39:08) confirms successful completion. Two consecutive invocations show idempotent behaviour.

### 6.2 Phase 2 – Exploratory data analysis
`run_phase2` loads the normalised Parquet file, computes aggregated DataFrames via `src/analysis/eda` helpers, and writes both CSV tables and markdown summary. Key functions:
- `eda.sentiment_distribution`: value counts & percentages across sentiments.
- `eda.airline_volume`: counts by airline.
- `eda.temporal_trend`: daily resampling of tweets, requiring non-null `tweet_created`.
- `eda.top_negative_reasons`: top 10 reasons by frequency.

Outputs land in `data/processed/` and `reports/eda_summary.md`. Phase 2 executes fast (<1 s) according to logs, aided by Parquet I/O.

### 6.3 Phase 3 – Sentiment pattern mining
`run_phase3` extends analysis beyond univariate distributions:
- `sentiment_patterns.sentiment_by_airline`: pivot table capturing per-airline sentiment counts and percentages, exported to `data/processed/sentiment_by_airline.csv`.
- `sentiment_patterns.negative_reason_summary`: counts per airline and reason, saved to `negative_reason_by_airline.csv`.
- `sentiment_patterns.mine_association_rules`: Apriori-based rule mining. The default threshold (support ≥0.01, confidence ≥0.3) yields >5 rules; a fallback reduces thresholds if necessary.
- `sentiment_patterns.generate_wordcloud`: produces `visualizations/wordcloud_negative.png` with custom stopwords.

Phase 3 also renders `reports/sentiment_analysis.md`, embedding markdown tables and linking to the word cloud asset. Completion occurs within ~4 seconds, dominated by word cloud generation (Pillow/wordcloud library file writing observed in logs at 09:40:12).

## 7. Feature Engineering and NLP Normalisation
`src/models/training.py` encapsulates text preprocessing and feature generation:
- **NLTK resources**: `ensure_nltk_resources()` checks for `stopwords` and `punkt`, downloading them if absent (logged during Phase 4 at 09:43:12 when first invoked).
- **Token cleaning**: `preprocess_text` lowercases text, removes hyperlinks, strips Twitter handles, and drops non-alphabetic characters before filtering stopwords. The function leverages regex from `re` and uses a cached stopword set per dataset to avoid repeated lookups.
- **Clean text column**: `prepare_dataset` filters out records lacking sentiment or text, applies `preprocess_text`, and discards rows whose cleaned text becomes empty. The resulting `clean_text` series feeds downstream vectorisers.
- **TF-IDF vectorisation**: `build_pipelines` instantiates `TfidfVectorizer` with bigram support (`ngram_range=(1,2)`). Linear SVC and Logistic Regression limit feature space to 6,000 tokens with `min_df` of 3; Complement NB allows 4,000 features with `min_df=2` to better suit multinomial counts.
- **Train/test split**: Both `train_pipeline` and `select_and_train_best` use an 80/20 stratified split to preserve class ratios. Random state 42 ensures reproducibility.

These transformations convert raw tweets into high-dimensional sparse vectors amenable to linear classifiers. The pipeline purposely avoids lemmatisation to maintain runtime efficiency, and the absence of class rebalancing beyond `class_weight="balanced"` on SVC and Logistic Regression reflects the already skewed distribution (negative sentiment majority).
## 8. Exploratory Data Analysis Findings
### 8.1 Sentiment distribution
`reports/eda_summary.md` and `data/processed/sentiment_distribution.csv` reveal the sentiment composition:

| Sentiment | Count | Percentage |
|-----------|------:|-----------:|
| Negative  | 9,178 | 62.69% |
| Neutral   | 3,099 | 21.17% |
| Positive  | 2,363 | 16.14% |

Interpretation:
- Negative sentiment outweighs neutral and positive, underscoring persistent dissatisfaction.
- Positive tweets form the smallest segment but still represent >2k supportive mentions, useful for comparative tone analysis.

### 8.2 Airline-specific volume

| Airline         | Tweets | Share of Corpus |
|-----------------|-------:|----------------:|
| United          | 3,822  | 26.1% |
| US Airways      | 2,913  | 19.9% |
| American        | 2,759  | 18.8% |
| Southwest       | 2,420  | 16.5% |
| Delta           | 2,222  | 15.2% |
| Virgin America  |   504  | 3.4%  |

United’s dominance indicates stronger brand visibility (and scrutiny). Virgin America’s smaller footprint suggests caution when generalising findings due to sample size.

### 8.3 Negative reason prevalence
Top 10 reasons (counts from `top_negative_reasons.csv`):

1. Customer Service Issue – 2,910
2. Late Flight – 1,665
3. Can't Tell – 1,190
4. Cancelled Flight – 847
5. Lost Luggage – 724
6. Bad Flight – 580
7. Flight Booking Problems – 529
8. Flight Attendant Complaints – 481
9. longlines – 178
10. Damaged Luggage – 74

Customer support frustrations and scheduling issues dominate, suggesting operational improvements in service recovery could impact the majority of complaints.

### 8.4 Temporal trends
Daily tweet counts (excerpt):
- 2015-02-16: 4 tweets (likely dataset warm-up).
- 2015-02-20: 1,500 tweets.
- 2015-02-22: 3,079 tweets (peak volume).
- 2015-02-24: 1,344 tweets (dataset tail).

Spikes coincide with weather-related disruptions and viral incidents recorded in February 2015, corroborating historical airline operations.

## 9. Sentiment Pattern Mining and Association Rules
### 9.1 Per-airline sentiment ratios
`sentiment_by_airline.csv` produces both counts and percentages:
- **US Airways**: 77.69% negative, 13.08% neutral, 9.23% positive (highest negative skew).
- **American Airlines**: 71.04% negative; positive share only 12.18%.
- **Delta**: More balanced, with negative 42.98% and positive 24.48% (best performer among majors).
- **Virgin America**: 35.91% negative, 30.16% positive (smallest sample yet highest positive share).

### 9.2 Airline–reason matrix
`negative_reason_by_airline.csv` spotlights specific challenges:
- United suffers the most “Customer Service Issue” complaints (681) and “Late Flight” mentions (525).
- US Airways mirrors United’s service challenge with 811 customer service complaints and 453 late flight references.
- Delta’s top two issues are “Late Flight” (269) and “Can't Tell” (186), signalling more operational than service deficiencies.

### 9.3 Association rules
`association_rules.csv` enumerates 25 high-confidence rules. Representative examples:

| Antecedents                                 | Consequents          | Support | Confidence | Lift | Conviction |
|---------------------------------------------|----------------------|--------:|-----------:|-----:|-----------:|
| airline=American                            | sentiment=negative   | 0.2136 | 1.0 | 1.0 | ∞ |
| airline=United                              | sentiment=negative   | 0.2869 | 1.0 | 1.0 | ∞ |
| airline=United, reason=Lost Luggage         | sentiment=negative   | 0.0293 | 1.0 | 1.0 | ∞ |
| reason=Customer Service Issue               | sentiment=negative   | 0.3171 | 1.0 | 1.0 | ∞ |
| airline=American, reason=Late Flight        | sentiment=negative   | 0.0271 | 1.0 | 1.0 | ∞ |

Because the dataset labels sentiment explicitly, any rule combining an airline or negative reason invariably leads to `sentiment=negative`, hence lift=1.0 and conviction infinite. While the mathematical output lacks discriminative power (no positive/neutral consequents), the support values quantify the weight of each issue; for example, “Customer Service Issue” explains 31.7% of all tweets.

### 9.4 Word clouds
`visualizations/wordcloud_negative.png` (and additional positive/neutral clouds generated by `scripts/generate_additional_visuals.py`) visually emphasise lexicon such as “delay,” “service,” “time,” “flight,” and “customer,” reinforcing textual indicators of service shortfalls.
## 10. Machine Learning Pipeline (Phase 4)
`src/models/training.py` encapsulates model experimentation. The two main entry points are `train_pipeline` (train a specific model) and `select_and_train_best` (benchmark all candidates, persist the winner). The pipeline definition excerpt illustrates architecture and hyperparameters:

```python
# src/models/training.py (abridged)
def build_pipelines() -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000)),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
            )),
        ]),
        "linear_svc": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
        "complement_nb": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000)),
            ("clf", ComplementNB()),
        ]),
    }
```

Training procedure (`select_and_train_best`) performs stratified splitting for reproducible evaluation, iterates over pipelines, and persists the best result via `save_trained_model` and `save_metrics`. Iteration order ensures all models receive identical train/test folds within the same run.

## 11. Model Evaluation and Diagnostics
### 11.1 Benchmark results
Metrics extracted from `models/model_metrics.json` and `models/complement_nb/metrics.json`:

| Model              | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|--------------------|---------:|----------------:|-------------:|---------:|------------:|
| Linear SVC         | 0.7626   | 0.7018          | 0.6962       | 0.6986   | 0.7633 |
| Complement Naive Bayes | 0.7588 | 0.6879      | 0.7051       | 0.6943   | 0.7586 |

Logistic Regression metrics are not persisted, but logs indicate it was evaluated prior to SVC and Complement NB.

### 11.2 Confusion matrix (Linear SVC)
From `model_metrics.json`:
- True Negative predicted Negative: 1,561
- True Negative predicted Neutral: 212
- True Negative predicted Positive: 61
- True Neutral predicted Negative: 190
- True Neutral predicted Neutral: 357
- True Neutral predicted Positive: 70
- True Positive predicted Negative: 80
- True Positive predicted Neutral: 81
- True Positive predicted Positive: 311

Observations:
- The model struggles more with the neutral class (precision 0.549, recall 0.579), reflecting lexical overlap between neutral and negative statements.
- Positive recall (0.659) trails precision (0.704), implying the model is slightly conservative when labelling positive tweets.

### 11.3 Complement NB confusion matrix
- Negative: 1,539 correctly predicted, with 185 neutral and 110 positive misclassifications.
- Neutral: 326 correct vs 192 predicted as negative and 99 as positive.
- Positive: 353 correct vs 68 negative and 51 neutral mislabels.

Complement NB outperforms SVC on positive recall (0.748) but underperforms on neutral recall (0.528), vindicating SVC as the chosen production model.

### 11.4 Classification reports
Markdown exports `reports/model_performance_linear_svc.md` and `reports/model_performance_complement_nb.md` align with the JSON metrics and are available for academic appendices.

## 12. Model Artefacts and Versioning
- `models/sentiment_pipeline.joblib` (377 KB) contains the Linear SVC pipeline with TF-IDF vectoriser and SVC estimator.
- `models/model_metrics.json` stores predictive metrics and the confusion matrix used by the dashboard.
- `models/linear_svc/model.joblib` and `models/complement_nb/model.joblib` preserve individual runs executed via `scripts/train_svm_nb.py`, enabling side-by-side experimentation.
- `models/model_accuracy_summary.json` records the latest accuracy figures per dedicated model training run.
- `data/processed/confusion_matrix.csv`, `confusion_matrix_linear_svc.csv`, and `confusion_matrix_complement_nb.csv` provide CSV representations for integration into BI tools.
- Backups under `backups/linear_svc/models/` replicate core artefacts for rollback.

Version control is file-based: rerunning Phase 4 overwrites `models/sentiment_pipeline.joblib` and `model_metrics.json`. To retain historical snapshots, archive directories should be time-stamped or integrated with external versioning (e.g., DVC). The current repository demonstrates a single lineage culminating in the 2025-10-17 run.
## 13. Visualisation Suite (Phase 5)
Phase 5 plus the auxiliary script `scripts/generate_additional_visuals.py` output 13 PNGs and 1 HTML file. Each artefact is derived from a specific DataFrame transformation:

| Asset | Source Data | Transformation | Notes |
|-------|-------------|----------------|-------|
| `visualizations/sentiment_distribution.png` | `sentiment_distribution.csv` | Seaborn bar plot (count vs sentiment). | Saved at 150 DPI. Copied into `app/static/img/` for dashboard display.
| `visualizations/sentiment_distribution.html` | Same | Plotly pie chart (`create_plotly_sentiment_distribution`). | Embeds CDN Plotly 3.1.1; iframe-served in dashboard.
| `visualizations/airline_sentiment_heatmap.png` | `sentiment_by_airline.csv` | Heatmap showing absolute counts per airline/sentiment. | `sns.heatmap` with `coolwarm` palette.
| `visualizations/tweet_volume_timeline.png` | `timeline_daily.csv` | Line chart (date vs tweet count). | Highlights daily spikes.
| `visualizations/top_negative_reasons.png` | `top_negative_reasons.csv` | Horizontal bar chart of top 10 reasons. | Sorted descending, magma palette.
| `visualizations/sentiment_share_per_airline.png` | `sentiment_by_airline.csv` | Stacked bar chart of percentages. | Generated via auxiliary script; uses custom colour palette (#d63031, #fdcb6e, #00b894).
| `visualizations/negative_ratio_per_airline.png` | `sentiment_by_airline.csv` | Bar chart of negative ratio (negative/total*100). | Highlights risk ranking.
| `visualizations/negative_reason_heatmap.png` | `negative_reason_by_airline.csv` | Heatmap limited to top 8 reasons. | Aggregates counts via pivot_table.
| `visualizations/daily_sentiment_trend.png` | Aggregated daily sentiment counts from `scripts/generate_additional_visuals.build_daily_sentiment_counts` | Multi-line chart per sentiment. | Requires grouping by `tweet_created.dt.date`.
| `visualizations/wordcloud_negative.png` | Negative tweets subset | WordCloud with `STOPWORDS` plus custom stopwords `{http, https, co, amp}`. | Derived from `sentiment_patterns.generate_wordcloud`.
| `visualizations/wordcloud_positive.png` | Positive tweets subset | WordCloud. | Generated via auxiliary script.
| `visualizations/wordcloud_neutral.png` | Neutral tweets subset | WordCloud. | Generated via auxiliary script.
| `visualizations/model_confusion_matrix.png` | Confusion matrix array | Heatmap of SVC confusion matrix. | Created post-training via `charts.plot_confusion_matrix`.
| `visualizations/confusion_matrix_linear_svc.png` & `confusion_matrix_complement_nb.png` | Model-specific confusion matrices | Alternative confusion matrix heatmaps written by `scripts/train_svm_nb.py`. | Provide comparison across models.

These assets feed both offline documentation and the dashboard. `app/routes.ensure_static_assets` copies canonical PNGs/HTML into `app/static/` to decouple the Flask runtime from generation scripts.

## 14. Web Application Architecture

The Flask application resides in `app/` and still follows the factory pattern, but now orchestrates the full continuous-learning workflow.

- `app/__init__.py` loads environment variables, wires `SECRET_KEY`/admin credentials, and initialises Flask-Limiter (in-memory by default) so the public prediction endpoint is rate-limited. It also seeds Celery broker/result URLs from the environment to keep worker and web processes aligned.
- `app/routes.py` registers blueprint `bp` with expanded surface area: `/`, `/dashboard`, `/admin`, `/admin/review/<id>`, `/admin/retrain`, `/admin/retrain/status/<task_id>`, `/admin/model/<int:version>/activate`, `/api/live-metrics`, and `/logout`. The module owns prediction caching (`_predict_with_cache`), background job dispatch (`retrain_model_task.delay(...)`), and live metrics JSON for the dashboard.
- `ensure_static_assets` still syncs generated visualisations into `app/static/`, ensuring the UI references the most recent analytics output on startup.
- `load_model` now consults the `model_versions` table: it loads the active versioned pipeline if present, falls back to the baseline artifact otherwise, and clears the LRU prediction cache whenever a new model is activated.
- Prediction flow: `/` route instantiates `PredictionForm`, validates input, applies rate limiting, serves cached explanations when available, persists the submission via `database.insert_submission`, and renders `templates/index.html` with reasoning and top-feature highlights.
- Dashboard flow: `/dashboard` extends the original analytics cards with model version history, live accuracy/pending counts (refreshed via `/api/live-metrics`), and visual asset links.
- Admin flow: `/admin` now groups submissions by status through `database.fetch_all_submissions_grouped`, surfaces bulk review tables, and exposes the manual retrain button. The view polls `/admin/retrain/status/<task_id>` until the Celery worker responds, then offers activation via `/admin/model/<int:version>/activate`.
- `review_submission` updates rows, logs match/mismatch to `prediction_logs`, and the `/logout` endpoint still clears the session to revoke admin rights.

All templates extend `base.html`, sharing the same design system (hero blocks, cards, responsive tables) while the admin and dashboard templates incorporate the new messaging, modals, and badges required for the continuous learning UX.


# Twitter Airline Sentiment Analysis & Web Dashboard – Comprehensive Technical Documentation

_Last updated: 2025-10-17_

Authoring context: Automated synthesis based on repository state located at `C:\xampp\htdocs\SentimentAnalysis`.

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Project Context and Objectives](#2-project-context-and-objectives)
- [3. System Architecture Overview](#3-system-architecture-overview)
- [4. Data Sources and Governance](#4-data-sources-and-governance)
- [5. Dataset Profiling and Quality Metrics](#5-dataset-profiling-and-quality-metrics)
- [6. Data Processing Pipeline (Phases 1-3)](#6-data-processing-pipeline-phases-1-3)
- [7. Feature Engineering and NLP Normalisation](#7-feature-engineering-and-nlp-normalisation)
- [8. Exploratory Data Analysis Findings](#8-exploratory-data-analysis-findings)
- [9. Sentiment Pattern Mining and Association Rules](#9-sentiment-pattern-mining-and-association-rules)
- [10. Machine Learning Pipeline (Phase 4)](#10-machine-learning-pipeline-phase-4)
- [11. Model Evaluation and Diagnostics](#11-model-evaluation-and-diagnostics)
- [12. Model Artefacts and Versioning](#12-model-artefacts-and-versioning)
- [13. Visualisation Suite (Phase 5)](#13-visualisation-suite-phase-5)
- [14. Web Application Architecture](#14-web-application-architecture)
- [15. API and Route Documentation](#15-api-and-route-documentation)
- [16. Front-end Implementation](#16-front-end-implementation)
- [17. Database Schema and Persistence Layer](#17-database-schema-and-persistence-layer)
- [18. Infrastructure and Deployment Workflow](#18-infrastructure-and-deployment-workflow)
- [19. Technology Stack and Dependency Inventory](#19-technology-stack-and-dependency-inventory)
- [20. Project Structure and File Inventory](#20-project-structure-and-file-inventory)
- [21. Automated Testing and Quality Assurance](#21-automated-testing-and-quality-assurance)
- [22. Logging, Monitoring, and Observability](#22-logging-monitoring-and-observability)
- [23. Security Considerations](#23-security-considerations)
- [24. Performance Characteristics](#24-performance-characteristics)
- [25. Backup and Disaster Recovery](#25-backup-and-disaster-recovery)
- [26. Design Decisions and Rationale](#26-design-decisions-and-rationale)
- [27. Limitations and Technical Debt](#27-limitations-and-technical-debt)
- [28. Future Enhancements and Roadmap](#28-future-enhancements-and-roadmap)
- [29. Deliverables and Evidence of Completion](#29-deliverables-and-evidence-of-completion)
- [30. Licence, Attribution, and Ethical Notes](#30-licence-attribution-and-ethical-notes)
- [31. Glossary](#31-glossary)
- [32. Summary and Final Recommendations](#32-summary-and-final-recommendations)
- [33. Appendix: Data Dictionary](#33-appendix-data-dictionary)
- [34. Appendix: Phase Execution Timeline](#34-appendix-phase-execution-timeline)
- [35. Appendix: Deployment Runbook](#35-appendix-deployment-runbook)
- [36. Appendix: Sample Payloads and Data Artefacts](#36-appendix-sample-payloads-and-data-artefacts)
- [37. Appendix: Risk Register and Mitigation Plan](#37-appendix-risk-register-and-mitigation-plan)

---

## 15. API and Route Documentation

| Route | Methods | Description |
|-------|---------|-------------|
| `/` | GET, POST | Public prediction form. Applies Flask-Limiter rate limits and stores each submission in `submissions`. POST responses include explanation metadata. |
| `/dashboard` | GET | Analytics overview plus live model card. Fetches CSV-derived summaries and reads `/api/live-metrics` via front-end polling. |
| `/admin` | GET, POST | Admin login and review workspace. Pending/approved/rejected tables sourced from `database.fetch_all_submissions_grouped`. |
| `/admin/review/<int:id>` | POST | Approve or reject a submission, persisting true sentiment and comments. Logs match/mismatch to `prediction_logs`. |
| `/admin/retrain` | POST | Triggers Celery background retraining when ≥50 labelled approvals exist. Returns task id for polling. |
| `/admin/retrain/status/<task_id>` | GET | Reports progress of the retraining task (pending/complete/failed). |
| `/admin/model/<int:version>/activate` | POST | Activates a saved model version and resets the prediction cache. |
| `/api/live-metrics` | GET | Lightweight JSON endpoint exposing current accuracy and pending count; consumed by dashboard auto-refresh. |
| `/logout` | GET | Clears admin session state. |

## 16. Front-end Implementation

- Templates share `base.html` for layout, navigation, and CSS variables. `index.html` now renders explanation reasoning and influential features per prediction.
- `dashboard.html` includes the new model information card, live metrics script, and historical version table.
- `admin.html` renders grouped tabs, retraining modal, and asynchronous polling logic. JavaScript handles task polling, modal decisions, and badge styling.
- Static assets (`app/static/`) continue to be populated by `ensure_static_assets`, ensuring regenerated analytics artefacts are served without manual copying.

## 17. Database Schema and Persistence Layer

- `submissions` retains user predictions and admin corrections. Indexes support review_status queries.
- `model_versions` tracks each saved pipeline with metrics, training counts, activation flag, notes, and file paths.
- `prediction_logs` (new) records match/mismatch per approved submission and model version for future accuracy tracking.
- Helper functions in `src/data/database.py` now include grouped-fetch helpers, version management, match logging, and counts for dashboard metrics.

## 18. Infrastructure and Deployment Workflow

- Runtime services: Flask app, Celery worker, MySQL (XAMPP or Homebrew), and Redis/Memurai broker. Environment variables `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` must match the running Redis instance.
- Celery tasks live in `tasks/retraining.py`; the worker is launched with `python -m celery -A celery_app worker --loglevel=info` after activating the venv.
- Manual retraining is intentionally asynchronous; admins trigger jobs via `/admin/retrain` and the worker persists model artefacts under `models/version_<n>/`.
- `migrationsteps.md` documents the full macOS onboarding process (system packages → venv → SQL import → run).

## 19. Technology Stack and Dependency Inventory

| Package | Version | Role |
|---------|--------:|------|
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.4 | Numerical backend |
| scikit-learn | 1.7.2 | Machine learning algorithms |
| scipy | 1.16.2 | Scientific routines for scikit-learn |
| nltk | 3.9.2 | Stopword corpus and tokenisation |
| mlxtend | 0.23.4 | Association rule mining |
| wordcloud | 1.9.4 | Word cloud generation |
| seaborn | 0.13.2 | Statistical visualisations |
| matplotlib | 3.10.7 | Plotting |
| plotly | 6.3.1 | Interactive charting |
| kaleido | 1.1.0 | Static export for Plotly |
| Flask | 3.1.2 | Web framework |
| Flask-WTF | 1.2.2 | Form handling |
| mysql-connector-python | 9.4.0 | MySQL connectivity |
| celery | 5.5.3 | Background task queue powering manual retraining |
| redis | 6.4.0 | Broker/result backend for Celery; optional cache store |
| flask-limiter | 4.0.0 | HTTP rate limiting for public prediction route |
| limits | 5.6.0 | Flask-Limiter storage abstraction |
| rich | 14.2.0 | Console formatting dependency for Celery CLI |
| schedule | 1.2.2 | Optional scheduled retraining script (currently disabled) |

## 20. Project Structure and File Inventory

- `app/` – Flask application (routes, forms, static assets, templates).
- `src/` – Data loaders, preprocessing utilities, model training helpers, and pipeline scripts.
- `tasks/` – Celery task modules for retraining.
- `models/` – Persisted pipeline artefacts and metrics (versioned subdirectories).
- `data/processed/` – Analytical CSV outputs consumed by the dashboard.
- `tests/` – Browser automation (Puppeteer) and Python regression suite.
- `migrationsteps.md` – Operating-system-specific migration guide (macOS focus).

## 21. Automated Testing and Quality Assurance

- **Puppeteer E2E tests** (`tests/puppeteer/run-tests.js`) still cover the seven user journeys documented previously, exercising prediction, dashboard, admin login, and approval flows.
- **Pytest regression suite** (`tests/test_continuous_learning.py`) mocks the database connector to validate schema initialisation, submission workflow updates, retraining pipeline, and explanation helper. Execute with `python -m pytest tests/test_continuous_learning.py -q`; latest run (2025-10-17) reports four passes.
- CI Recommendation: run both suites before packaging releases to ensure the Celery/Flask contract stays intact.

## 35. Appendix: Deployment Runbook

1. **Obtain the project folder** (clone repo or unzip archive) into the desired workspace, e.g. `C:\xampp\htdocs\SentimentAnalysis` for Windows or `~/Projects/SentimentAnalysis` on macOS.
2. **Create and activate a Python virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate   # Windows PowerShell
   # or on macOS/Linux: source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Ensure infrastructure services are running**:
   - MySQL: start XAMPP MySQL (Windows) or `brew services start mysql` (macOS).
   - Redis or Memurai: start the service (`services.msc` on Windows / `brew services start redis` on macOS).
4. **Configure the application**:
   - Populate `.env` with secrets and connection strings (`DB_*`, `SECRET_KEY`, `ADMIN_*`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`).
   - Create the `airline_sentiment` database if not present and import the provided SQL dump when migrating existing data.
5. **Initialise helper tables and model metadata**:
   ```powershell
   python -m scripts.init_model_versions
   ```
6. **(Optional) Run automated tests** to verify the setup:
   ```powershell
   python -m pytest tests/test_continuous_learning.py -q
   npm test   # executes Puppeteer scenarios if Node dependencies installed
   ```
7. **Launch the web application (Terminal A)**:
   ```powershell
   .\venv\Scripts\Activate
   python run.py
   ```
   Access the UI at `http://127.0.0.1:5000/`.
8. **Launch the Celery worker (Terminal B)** to handle admin-triggered retraining:
   ```powershell
   .\venv\Scripts\Activate
   $env:CELERY_BROKER_URL = 'redis://127.0.0.1:6379/0'
   $env:CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'
   python -m celery -A celery_app worker --loglevel=info
   ```
   (Use `export` instead of `$env:` when on macOS/Linux shells.)
9. **Do not run** `scripts/scheduled_retrain.py` unless you explicitly want nightly automated retraining; manual retraining via the admin dashboard is the supported workflow.
10. **Shutdown** by pressing `Ctrl+C` in both terminals. Stop Redis/MySQL services if they were started manually.
11. For cross-machine migrations (e.g., macOS laptops), follow `migrationsteps.md` which expands these steps into a fully automated checklist.
