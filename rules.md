# RULES.md — Twitter Airline Sentiment Analysis & Web Dashboard (Full Autonomous Deployment)

## PROJECT OBJECTIVE

Perform comprehensive sentiment analysis on the Tweets.csv dataset containing customer feedback for US airlines, generate detailed analytical reports in English and Indonesian, build a functional web dashboard with sentiment prediction capabilities, set up complete database infrastructure, deploy the application to localhost, run automated tests, and deliver a fully operational system accessible via browser.

## CONTEXT

- **Business/Domain**: Social media sentiment analysis for US airline customer feedback on Twitter. Quantify sentiment, identify pain points, discover patterns in negative feedback, and create a user-facing system for real-time sentiment prediction.
- **User/Actor**: Academic project for lecturer and student presentation. End users include normal users (sentiment prediction), admins (review system), and stakeholders (dashboard viewers).
- **Key Constraints**: Dataset is Tweets.csv in root directory. XAMPP with Apache and MySQL already running. Must work on Windows. Python version detection required. Node.js already installed. All work must be fully autonomous from environment setup through deployment and testing.
- **Non-Goals**: Not a cloud deployment. Not production-grade security. Focus is on analysis completeness, academic report quality, and functional prototype demonstration.

## PHASES

### Phase 0: Pre-Flight System Check & Setup
Detect system state, verify prerequisites, install missing dependencies, create project structure, and prepare execution environment.

**Deliverables**: System capability report, installed dependencies, project directory structure, virtual environment, master orchestration script.

### Phase 1: Data Loading & Validation
Load and validate the Tweets.csv dataset, understand its structure, and prepare for analysis.

**Deliverables**: Validated dataset loaded into memory, initial data quality report, confirmed column structure.

### Phase 2: Exploratory Data Analysis (EDA)
Conduct comprehensive exploratory analysis including descriptive statistics, missing value analysis, distribution analysis, and initial pattern discovery.

**Deliverables**: EDA summary with statistics, data quality findings, initial insights on sentiment distribution and airline coverage.

### Phase 3: Sentiment Analysis & Pattern Discovery
Perform deep sentiment analysis across airlines, identify negative reason patterns, conduct association rule mining, and generate word clouds for negative tweets.

**Deliverables**: Sentiment metrics per airline, negative reason frequency analysis, association rules with confidence scores, word cloud visualizations.

### Phase 4: Predictive Modeling
Build and train machine learning model(s) for sentiment classification to power the web dashboard's prediction feature.

**Deliverables**: Trained sentiment classification model, model performance metrics, serialized model files for web integration.

### Phase 5: Visualization Creation
Create comprehensive data visualizations including charts for tweet volume per airline, sentiment distribution, temporal patterns, and negative reason breakdowns.

**Deliverables**: PNG/SVG chart files, embedded visualization code for dashboard, summary visualization report.

### Phase 6: Report Generation
Generate all required markdown reports: detailed English analysis results, Indonesian academic report (Laporan Project), Python notebook documentation, and presentation outline.

**Deliverables**: `analysis_results.md`, `laporan_project.md`, `python_notebook_documentation.md`, `presentation_outline.md`.

### Phase 7: Database Setup & Schema Creation
Connect to existing MySQL instance via XAMPP, create database and tables for storing user submissions and validation data.

**Deliverables**: `airline_sentiment` database created, `submissions` table schema deployed, connection validated.

### Phase 8: Web Dashboard Development
Build a functional three-page Flask website: (1) Home page with sentiment prediction input, (2) Dashboard with data visualizations, (3) Admin review page for validating predictions.

**Deliverables**: Complete Flask application with HTML/CSS/JS, working sentiment prediction API integration, interactive dashboard, admin review interface.

### Phase 9: Dependency Management & Server Deployment
Install all required Python and Node.js packages, configure Flask server, and deploy application to localhost.

**Deliverables**: All dependencies installed, Flask server running on available port, application accessible via browser.

### Phase 10: Automated Testing with Puppeteer
Install/verify Puppeteer, create comprehensive test suite, execute all tests, handle failures with retries and fixes, validate full system functionality.

**Deliverables**: Puppeteer test script, test execution report, screenshots of any failures, confirmation of system readiness.

### Phase 11: Final Integration & Launch
Validate all deliverables, ensure consistency across reports, perform final self-review, print access URLs, and keep server running for user access.

**Deliverables**: Complete project package, running server, printed URLs, execution summary report.

## DETAILED TASK BREAKDOWN

### Phase 0 Tasks

- **Task 0.1**: Detect current Python version using `python --version` or `python3 --version`. Store version info. Proceed with whatever version is found (no version enforcement). If Python not found, halt with clear error.
- **Task 0.2**: Verify Node.js installation using `node --version`. If not found, halt with error (you confirmed it's installed).
- **Task 0.3**: Check if Puppeteer is installed globally using `npm list -g puppeteer`. If not found, plan to install locally in project.
- **Task 0.4**: Verify XAMPP MySQL is running by attempting connection to `localhost:3306` with default credentials (root, no password). If connection fails, troubleshoot: check if MySQL service is running via `sc query mysql` or XAMPP control panel status. Attempt to start MySQL service if stopped. Retry connection. If repeated failures, document issue and attempt fixes (check port conflicts, restart XAMPP MySQL module). Do not halt; continue attempting fixes.
- **Task 0.5**: Create project directory structure:
```
  project_root/
  ├── data/ (for intermediate data files)
  ├── models/ (for trained models)
  ├── visualizations/ (for all chart images)
  ├── reports/ (for all markdown reports)
  ├── website/
  │   ├── static/
  │   │   ├── css/
  │   │   ├── js/
  │   │   └── images/
  │   ├── templates/
  │   └── app.py
  ├── tests/ (for Puppeteer test scripts)
  ├── logs/ (for execution logs)
  ├── requirements.txt
  └── run_everything.py
```
- **Task 0.6**: Create Python virtual environment in project root using `python -m venv venv`. Activate it using `venv\Scripts\activate` (Windows). All subsequent pip installs happen in this venv.
- **Task 0.7**: Create `run_everything.py` master orchestration script that:
  - Activates venv
  - Runs data analysis phases (1-6)
  - Sets up database (Phase 7)
  - Installs dependencies and starts Flask server (Phase 8-9)
  - Runs Puppeteer tests (Phase 10)
  - Prints final URLs and keeps server running
  - Logs all output to `logs/execution.log`
- **Task 0.8**: Create initial `requirements.txt` with base packages: `flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `wordcloud`, `mlxtend`, `nltk`, `mysql-connector-python`, `requests`. As execution proceeds, add any additional packages needed.

### Phase 1 Tasks

- **Task 1.1**: Load Tweets.csv from root directory using pandas. Handle file not found gracefully with clear error message and halt if missing.
- **Task 1.2**: Validate expected columns exist: `tweet_id`, `text`, `airline_sentiment`, `airline_sentiment_confidence`, `negativereason`, `negativereason_confidence`, `airline`, `retweet_count`, `created`, `name`, `location`, `timezone`. If column names differ slightly (case, underscores vs spaces), intelligently map them. Document any discrepancies.
- **Task 1.3**: Print dataset shape (rows, columns) and first 5 rows for verification.
- **Task 1.4**: Check data types and flag any unexpected types. Attempt automatic type conversion where appropriate.
- **Task 1.5**: Log successful data load to execution log.

### Phase 2 Tasks

- **Task 2.1**: Calculate descriptive statistics for numerical columns (`retweet_count`, `airline_sentiment_confidence`, `negativereason_confidence`) using `.describe()`.
- **Task 2.2**: Analyze missing values per column. Calculate count and percentage. Create missing value summary table.
- **Task 2.3**: Create sentiment distribution breakdown (count and percentage of positive/neutral/negative tweets). Create both raw counts and percentages.
- **Task 2.4**: Analyze tweet volume per airline. Create frequency table sorted by volume descending.
- **Task 2.5**: Parse `created` column to datetime. Extract date, hour, day of week. Analyze temporal patterns: tweets by date, tweets by hour, tweets by day of week.
- **Task 2.6**: Analyze user geography using `location` field. Extract top 20 locations by tweet count. Analyze `timezone` distribution.
- **Task 2.7**: Identify any data quality issues: duplicates, outliers in retweet_count, suspicious patterns.
- **Task 2.8**: Compile all EDA findings into structured summary with tables and key statistics. Save as intermediate file `data/eda_summary.json` for reference in report generation.

### Phase 3 Tasks

- **Task 3.1**: Calculate sentiment metrics per airline: for each airline, compute count and percentage of positive, neutral, negative tweets. Create summary DataFrame.
- **Task 3.2**: Filter to negative tweets only. Analyze `negativereason` column: create frequency distribution overall and per airline. Handle missing values in negativereason (many negative tweets may not have a specified reason).
- **Task 3.3**: Calculate average confidence scores: overall sentiment confidence, negative reason confidence (when available).
- **Task 3.4**: Perform association rule mining to discover airline-negativereason patterns. Use mlxtend's Apriori algorithm. Transform data into transaction format (each row = list of [airline, negativereason]). Set min_support=0.01, min_confidence=0.3. Generate rules with support, confidence, lift metrics. Save to `data/association_rules.csv`.
- **Task 3.5**: Generate word cloud from negative tweet text. Preprocessing: lowercase, remove URLs, remove mentions (@user), remove special characters, remove common stopwords (use NLTK stopwords + custom airline names). Use WordCloud library with max_words=200, background_color='white', width=1200, height=800. Save as `visualizations/negative_tweets_wordcloud.png`.
- **Task 3.6**: Extract top 10 most frequent words in negative tweets using CountVectorizer or similar. Do this overall and per airline. Store results.
- **Task 3.7**: Identify key patterns and correlations: which airlines have highest negative sentiment, which negative reasons are most common per airline, any surprising associations.
- **Task 3.8**: Compile all pattern findings into structured format for reporting.

### Phase 4 Tasks

- **Task 4.1**: Prepare training data. Features = `text` column, Target = `airline_sentiment` column. Remove any rows with missing text or sentiment.
- **Task 4.2**: Split data into train/test sets (80/20 split, stratified by sentiment to maintain class distribution). Use sklearn's train_test_split with random_state=42.
- **Task 4.3**: Text preprocessing pipeline: lowercase conversion, remove URLs (regex), remove mentions (@username), remove special characters, remove numbers, strip extra whitespace. Apply to all text.
- **Task 4.4**: Create TF-IDF vectorizer with parameters: max_features=5000, ngram_range=(1,2), min_df=5, max_df=0.8. Fit on training text, transform both train and test sets.
- **Task 4.5**: Train three classification models:
  - Logistic Regression (C=1.0, max_iter=1000, solver='liblinear')
  - Random Forest (n_estimators=100, max_depth=20, random_state=42)
  - Multinomial Naive Bayes (alpha=0.1)
- **Task 4.6**: Evaluate all models on test set. Calculate accuracy, precision (weighted), recall (weighted), F1-score (weighted), and confusion matrix. Create classification report for each model.
- **Task 4.7**: Select best performing model based on F1-score. If F1 scores are within 2% of each other, prefer Logistic Regression for interpretability.
- **Task 4.8**: Save best model using joblib to `models/sentiment_model.pkl`. Save vectorizer to `models/vectorizer.pkl`. Save model metadata (model type, parameters, performance metrics) to `models/model_info.json`.
- **Task 4.9**: Document model architecture, hyperparameters, training process, and performance metrics for inclusion in reports.

### Phase 5 Tasks

- **Task 5.1**: Create bar chart: tweet volume per airline. Use matplotlib or seaborn. X-axis=airline names, Y-axis=tweet count. Sort bars by count descending. Add value labels on bars. Title: "Tweet Volume by Airline". Save as `visualizations/tweets_per_airline.png` (300 DPI).
- **Task 5.2**: Create stacked bar chart: sentiment distribution per airline. X-axis=airlines, Y-axis=count, stacked bars colored by sentiment (green=positive, yellow=neutral, red=negative). Include legend. Title: "Sentiment Distribution by Airline". Save as `visualizations/sentiment_by_airline.png`.
- **Task 5.3**: Create pie chart: overall sentiment distribution. Show percentages and counts. Colors: green, yellow, red. Title: "Overall Sentiment Distribution". Save as `visualizations/overall_sentiment.png`.
- **Task 5.4**: Create bar chart: top 10 negative reasons with counts. Sort descending. Add value labels. Title: "Top 10 Reasons for Negative Sentiment". Save as `visualizations/top_negative_reasons.png`.
- **Task 5.5**: Create line chart: tweet volume over time (daily). X-axis=date, Y-axis=tweet count. Add trend line if applicable. Title: "Tweet Volume Over Time". Save as `visualizations/temporal_pattern.png`.
- **Task 5.6**: Create heatmap: airline (rows) vs negative reason (columns) showing count of tweets. Use seaborn heatmap with annotation. Title: "Airline vs Negative Reason Heatmap". Save as `visualizations/airline_negativereason_heatmap.png`.
- **Task 5.7**: Create interactive Plotly versions of key charts for dashboard embedding: sentiment by airline (stacked bar), temporal pattern (line chart). Save as standalone HTML files: `visualizations/sentiment_by_airline_interactive.html`, `visualizations/temporal_pattern_interactive.html`.
- **Task 5.8**: Create summary visualization document listing all charts with brief descriptions. Save as `visualizations/visualization_summary.md`.

### Phase 6 Tasks

- **Task 6.1**: Write `reports/analysis_results.md` in English. Structure:
  - **Executive Summary** (200 words): High-level overview of project, key findings, main insights.
  - **1. Introduction**: Project background, objectives, dataset description.
  - **2. Dataset Overview**: Source, size, columns, date range, airlines covered. Include summary statistics table.
  - **3. Exploratory Data Analysis**: 
    - Sentiment distribution (overall and per airline) with table
    - Tweet volume analysis with chart reference
    - Temporal patterns with chart reference
    - Geographic distribution insights
    - Data quality findings
  - **4. Sentiment Analysis Results**:
    - Detailed per-airline sentiment breakdown with table
    - Comparison across airlines
    - Statistical significance of differences (if applicable)
  - **5. Negative Sentiment Deep Dive**:
    - Negative reason frequency analysis with table
    - Word cloud insights with image embed
    - Top negative words/phrases
  - **6. Association Rule Mining**:
    - Methodology explanation
    - Key rules discovered (airline-negativereason associations)
    - Interpretation of support, confidence, lift metrics
  - **7. Predictive Modeling**:
    - Model selection process
    - Training methodology
    - Performance metrics table (accuracy, precision, recall, F1)
    - Confusion matrix visualization
    - Feature importance (if available from model)
  - **8. Visualizations**: Embed all charts with captions
  - **9. Key Insights & Recommendations**:
    - Top 5 actionable insights for airlines
    - Recommendations for customer service improvement
    - Suggestions for further analysis
  - **10. Limitations & Future Work**:
    - Data limitations
    - Model limitations
    - Suggested improvements
  - **References**: Any sources, libraries used
  - Target length: 3000-4500 words. Use markdown tables, headers, embedded images, bullet points. Ensure professional academic tone.
- **Task 6.2**: Write `reports/laporan_project.md` in Indonesian. Follow standard academic structure:
  - **Judul**: "Analisis Sentimen Tweet Maskapai Penerbangan AS Menggunakan Machine Learning dan Asosiasi Data Mining"
  - **Abstrak** (150-200 kata): Ringkasan proyek, metodologi, temuan utama, kesimpulan.
  - **1. Pendahuluan**:
    - 1.1 Latar Belakang: Pentingnya analisis sentimen media sosial untuk industri penerbangan
    - 1.2 Rumusan Masalah: Pertanyaan penelitian yang dijawab
    - 1.3 Tujuan Penelitian: Tujuan analisis ini
    - 1.4 Manfaat Penelitian: Untuk akademis dan praktis
  - **2. Tinjauan Pustaka**:
    - 2.1 Analisis Sentimen: Definisi dan aplikasi
    - 2.2 Machine Learning untuk Klasifikasi Teks: Overview teknik
    - 2.3 Association Rule Mining: Penjelasan Apriori algorithm
    - 2.4 Penelitian Terkait: Studi sebelumnya tentang analisis sentimen media sosial
  - **3. Metodologi**:
    - 3.1 Sumber Data: Deskripsi dataset Tweets.csv
    - 3.2 Teknik Analisis: EDA, sentiment analysis, association rules, predictive modeling
    - 3.3 Tools dan Library: Python, pandas, scikit-learn, dll.
    - 3.4 Tahapan Penelitian: Phase-by-phase breakdown
  - **4. Hasil dan Pembahasan**: (Synchronized content from English report, translated and adapted)
    - 4.1 Analisis Eksploratif Data
    - 4.2 Hasil Analisis Sentimen
    - 4.3 Analisis Alasan Sentimen Negatif
    - 4.4 Temuan Association Rule Mining
    - 4.5 Hasil Pemodelan Prediktif
    - 4.6 Visualisasi Data
  - **5. Kesimpulan dan Saran**:
    - 5.1 Kesimpulan: Ringkasan temuan utama
    - 5.2 Saran: Rekomendasi untuk maskapai dan penelitian lanjutan
  - **Daftar Pustaka**: Format APA/IEEE untuk semua referensi
  - Target length: 3000-4500 words. Use formal Bahasa Indonesia. Ensure academic rigor and proper citations.
- **Task 6.3**: Write `reports/python_notebook_documentation.md` documenting the entire workflow as if it were a Jupyter notebook. Structure:
  - **Notebook Title**: "Twitter Airline Sentiment Analysis - Complete Workflow"
  - **Section 1: Setup and Imports**: Markdown explanation + code block showing all imports
  - **Section 2: Data Loading**: Markdown explanation + code block for loading Tweets.csv + sample output
  - **Section 3: Exploratory Data Analysis**: 
    - Subsections for each EDA task
    - Markdown explanation of what's being analyzed + code block + sample output/table
  - **Section 4: Data Preprocessing**: Text cleaning code + explanations
  - **Section 5: Sentiment Analysis**: Code for sentiment calculations + tables
  - **Section 6: Negative Reason Analysis**: Code + visualizations
  - **Section 7: Association Rule Mining**: Code + explanation of rules
  - **Section 8: Word Cloud Generation**: Code + image embed
  - **Section 9: Predictive Modeling**: 
    - Train/test split code
    - Model training code for each model
    - Evaluation code with results
  - **Section 10: Visualizations**: Code for each chart + image embeds
  - **Section 11: Conclusions**: Summary of findings
  - Use proper markdown formatting with code blocks in Python syntax highlighting. Include comments in code. Show realistic output examples (sample dataframes, metrics). Target length: 2000-3000 words of explanation + code.
- **Task 6.4**: Write `reports/presentation_outline.md` for academic presentation. Structure:
  - **Slide 1: Title Slide**
    - Title: "Twitter Airline Sentiment Analysis: Uncovering Customer Feedback Patterns"
    - Authors, Institution, Date
    - Speaking points: Brief introduction of self and project
  - **Slide 2: Problem Statement**
    - Bullet points: Why analyze airline sentiment, business impact, research questions
    - Visual: Icon or image of social media + airlines
    - Speaking points: Context setting, importance of social listening
  - **Slide 3: Dataset Description**
    - Bullet points: Source, size, timeframe, airlines covered, key variables
    - Visual: Sample tweet examples, data structure diagram
    - Speaking points: Dataset characteristics, data collection methodology
  - **Slide 4: Methodology Overview**
    - Bullet points: Analysis phases (EDA, Sentiment Analysis, Association Rules, Predictive Modeling)
    - Visual: Flowchart of methodology
    - Speaking points: High-level explanation of each phase
  - **Slide 5-6: EDA Key Findings**
    - Slide 5: Sentiment distribution chart, tweet volume by airline
    - Slide 6: Temporal patterns, geographic insights
    - Speaking points: Walk through key statistics, highlight interesting patterns
  - **Slide 7-8: Sentiment Analysis Results**
    - Slide 7: Sentiment by airline comparison table/chart
    - Slide 8: Negative reason breakdown
    - Speaking points: Which airlines have most negative sentiment, why customers are unhappy
  - **Slide 9: Word Cloud Insights**
    - Visual: Negative tweets word cloud (large, center of slide)
    - Speaking points: Most common complaint themes, language patterns
  - **Slide 10: Association Rules**
    - Table: Top 5 airline-negativereason rules with support/confidence/lift
    - Speaking points: Explain association rule concept, interpret top rules
  - **Slide 11: Predictive Model**
    - Bullet points: Model type, training process, performance metrics
    - Visual: Confusion matrix or performance comparison chart
    - Speaking points: How model works, accuracy achieved, practical application
  - **Slide 12: Web Dashboard Demo**
    - Visual: Screenshots of home page, dashboard, admin interface
    - Speaking points: Live demo (if presenting in person), explain features
  - **Slide 13: Key Insights**
    - Bullet points: Top 5 actionable insights for airlines
    - Speaking points: Practical implications, recommendations
  - **Slide 14: Limitations & Future Work**
    - Bullet points: Data limitations, model constraints, suggested improvements
    - Speaking points: Honest assessment, areas for expansion
  - **Slide 15: Conclusion & Q&A**
    - Bullet points: Summary of project value, thank you
    - Speaking points: Wrap up, invite questions
  - For each slide, provide 3-5 bullet points of content and 2-3 sentences of speaking notes. Include notes on which visuals to display.

### Phase 7 Tasks

- **Task 7.1**: Import MySQL connector library. If `mysql-connector-python` import fails, try `pymysql` as alternative. If both fail, attempt to install using pip and retry.
- **Task 7.2**: Establish connection to MySQL via XAMPP: host='localhost', port=3306, user='root', password='' (empty). If connection fails, troubleshoot:
  - Check if MySQL service is running
  - Verify port 3306 is not blocked
  - Try alternative connection parameters
  - Retry up to 3 times with 5-second delays
  - If all retries fail, log detailed error and halt database setup (but continue with other phases where possible)
- **Task 7.3**: Create database `airline_sentiment` if not exists using SQL: `CREATE DATABASE IF NOT EXISTS airline_sentiment;`. Execute and verify.
- **Task 7.4**: Switch to `airline_sentiment` database using `USE airline_sentiment;`.
- **Task 7.5**: Create `submissions` table with schema:
```sql
  CREATE TABLE IF NOT EXISTS submissions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tweet_text TEXT NOT NULL,
    predicted_sentiment VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_correct BOOLEAN NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_sentiment (predicted_sentiment)
  );
```
  Execute and verify table creation.
- **Task 7.6**: Test insert operation: insert a dummy submission record and verify it's retrievable. Delete dummy record after test.
- **Task 7.7**: Save database connection parameters to config file `website/config.py` for Flask app to use:
```python
  DB_CONFIG = {
      'host': 'localhost',
      'port': 3306,
      'user': 'root',
      'password': '',
      'database': 'airline_sentiment'
  }
```
- **Task 7.8**: Log successful database setup to execution log.

### Phase 8 Tasks

- **Task 8.1**: Create Flask app structure in `website/` directory.
- **Task 8.2**: Create `website/app.py` with the following routes and functionality:
  - **Import statements**: Flask, render_template, request, jsonify, datetime, joblib, mysql.connector (or pymysql), config
  - **App initialization**: `app = Flask(__name__)`
  - **Model loading on startup**: Load `models/sentiment_model.pkl` and `models/vectorizer.pkl` using joblib. Store in global variables. Handle file not found gracefully.
  - **Database connection helper**: Function to get MySQL connection using DB_CONFIG from config.py. Include error handling.
  - **Route `/`** (GET): Render `index.html` (home page)
  - **Route `/dashboard`** (GET): Render `dashboard.html`
  - **Route `/admin`** (GET): Render `admin.html`
  - **Route `/predict`** (POST):
    - Accept JSON with field `tweet_text`
    - Validate input (not empty, length < 500 chars)
    - Preprocess text (same preprocessing as training: lowercase, remove URLs, etc.)
    - Vectorize using loaded vectorizer
    - Predict sentiment using loaded model
    - Get prediction probability (confidence score)
    - Insert submission into database (tweet_text, predicted_sentiment, confidence, timestamp)
    - Return JSON: `{sentiment: "positive/neutral/negative", confidence: 0.XX, message: "success"}`
    - Error handling: return error JSON if prediction fails
  - **Route `/submissions`** (GET):
    - Query database for all submissions ordered by timestamp DESC
    - Return JSON array: `[{id, tweet_text, predicted_sentiment, confidence, timestamp, is_correct}, ...]`
  - **Route `/validate`** (POST):
    - Accept JSON with fields `submission_id` and `is_correct` (boolean)
    - Update database: `UPDATE submissions SET is_correct = ? WHERE id = ?`
    - Return JSON: `{message: "success"}` or error
  - **Error handlers**: 404, 500 error pages
  - **Run configuration**: `if __name__ == '__main__': app.run(debug=True, port=5000, host='0.0.0.0')`
- **Task 8.3**: Create `website/templates/base.html` with:
  - HTML5 boilerplate
  - Bootstrap 5 CSS CDN or Tailwind CSS CDN for styling
  - Navigation bar with links to Home, Dashboard, Admin
  - Block content for child templates
  - Footer with project info
- **Task 8.4**: Create `website/templates/index.html` (extends base.html):
  - Hero section with project title and description
  - Form with:
    - Textarea for tweet input (placeholder: "Enter a tweet about an airline...")
    - Submit button (styled attractively)
    - Loading spinner (hidden by default)
  - Results display area (hidden by default):
    - Shows predicted sentiment (with color coding: green/yellow/red)
    - Shows confidence score as percentage
    - Option to submit another tweet
  - Link to Dashboard
- **Task 8.5**: Create `website/templates/dashboard.html` (extends base.html):
  - Grid layout (2-3 columns) for visualizations
  - Summary statistics cards at top:
    - Total tweets analyzed
    - Overall sentiment breakdown (%)
    - Most common negative reason
  - Embedded visualizations:
    - Tweet volume by airline (static image)
    - Sentiment by airline (interactive Plotly chart from HTML file)
    - Overall sentiment pie chart (static image)
    - Top negative reasons (static image)
    - Temporal pattern (interactive Plotly chart)
    - Word cloud (static image)
    - Heatmap (static image)
  - Responsive design (stacks on mobile)
- **Task 8.6**: Create `website/templates/admin.html` (extends base.html):
  - Simple authentication placeholder: password input field (client-side check for password="admin123", not secure but functional)
  - After "authentication", show submissions table:
    - Columns: ID, Tweet Text (truncated to 100 chars with "..."), Predicted Sentiment, Confidence, Timestamp, Validation Status, Actions
    - Each row has two buttons: "Mark Correct" (green), "Mark Incorrect" (red)
    - Validation status shows: "Pending" (grey), "Correct" (green checkmark), "Incorrect" (red X)
  - Search/filter functionality: filter by sentiment, validation status
  - Pagination (if >50 submissions)
  - Real-time updates via AJAX
- **Task 8.7**: Create `website/static/css/style.css`:
  - Custom styles for enhanced UI
  - Color scheme: professional blue/green palette
  - Responsive design utilities
  - Animation for loading states
  - Button hover effects
- **Task 8.8**: Create `website/static/js/main.js`:
  - Form submission handler for home page:
    - Prevent default form submit
    - Get tweet text from textarea
    - Validate input client-side
    - Show loading spinner
    - Fetch POST to `/predict` with JSON body
    - On success: hide form, show results with sentiment and confidence
    - On error: show error message
    - Reset form button functionality
  - Smooth scroll to results
  - Error handling and user feedback
- **Task 8.9**: Create `website/static/js/admin.js`:
  - On page load, fetch submissions from `/submissions`
  - Render table dynamically with fetched data
  - Attach click handlers to "Mark Correct" and "Mark Incorrect" buttons:
    - Fetch POST to `/validate` with submission_id and is_correct
    - On success: update row UI to show validation status
    - On error: show error message
  - Implement search/filter functionality (client-side filtering)
  - Handle pagination if needed
- **Task 8.10**: Copy all visualization images from `visualizations/` to `website/static/images/`.
- **Task 8.11**: Copy interactive Plotly HTML files to `website/static/` or embed them directly in dashboard.html using iframe or Plotly JS.
- **Task 8.12**: Create `website/README_WEBSITE.md` with instructions:
  - Prerequisites: Python, pip, XAMPP with MySQL
  - Installation: `pip install -r requirements.txt`
  - Database setup: Ensure MySQL is running, database and table are created (refer to Phase 7)
  - Running: `python website/app.py` or `python -m flask run` from website directory
  - Access: Open browser to `http://localhost:5000`
  - Testing: How to test prediction, view dashboard, use admin panel
  - Troubleshooting: Common issues and solutions

### Phase 9 Tasks

- **Task 9.1**: Ensure virtual environment is activated (if not, activate it).
- **Task 9.2**: Install all Python dependencies from `requirements.txt` using `pip install -r requirements.txt`. If any package fails to install:
  - Log the failure
  - Search for alternative package (e.g., if `mysql-connector-python` fails, try `pymysql`)
  - Update requirements.txt with working alternative
  - Retry installation
  - If critical package cannot be installed after alternatives, log error but continue where possible
- **Task 9.3**: Verify all required packages are importable by attempting imports in a test script. Log any import errors.
- **Task 9.4**: Check if port 5000 is available. If occupied, try ports 5001, 5002, etc. until finding available port. Update Flask app
run configuration with selected port.
- **Task 9.5**: Start Flask server in background process:
  - On Windows, use `start /B python website/app.py > logs/flask.log 2>&1`
  - Alternative: Use `pythonw website/app.py` to run without console window
  - Store process ID for potential cleanup
  - Wait 5 seconds for server to initialize
- **Task 9.6**: Verify Flask server is running by sending GET request to `http://localhost:{port}/`. If response is not 200, troubleshoot:
  - Check logs/flask.log for errors
  - Verify model files are loaded correctly
  - Check database connection
  - Retry server start up to 3 times
  - If repeated failures, log detailed error but continue to testing phase to capture failure details
- **Task 9.7**: Log server startup success with port number and URL.

### Phase 10 Tasks

- **Task 10.1**: Check if Puppeteer is installed globally using `npm list -g puppeteer`. If found, note version. If not found, install locally in project: `cd tests && npm init -y && npm install puppeteer`. Use `--no-sandbox` flag for compatibility.
- **Task 10.2**: Create `tests/test_suite.js` Puppeteer test script with the following test cases:
  - **Setup**: Import puppeteer, set base URL from environment or default to `http://localhost:5000`
  - **Test 1: Home Page Load**
    - Navigate to home page
    - Wait for page load
    - Verify page title contains "Sentiment" or "Airline"
    - Verify textarea element exists
    - Verify submit button exists
    - Take screenshot: `screenshots/01_home_page.png`
  - **Test 2: Submit Test Tweet - Positive**
    - Navigate to home page
    - Type test tweet: "I love flying with this airline! Great service and comfortable seats."
    - Click submit button
    - Wait for results to appear (max 10 seconds)
    - Verify result element contains sentiment ("positive", "neutral", or "negative")
    - Verify confidence score is displayed (number between 0-100)
    - Take screenshot: `screenshots/02_prediction_positive.png`
  - **Test 3: Submit Test Tweet - Negative**
    - Navigate to home page
    - Type test tweet: "Terrible customer service! My flight was delayed for 5 hours and no one helped."
    - Click submit button
    - Wait for results
    - Verify prediction appears
    - Take screenshot: `screenshots/03_prediction_negative.png`
  - **Test 4: Dashboard Page Load**
    - Navigate to `/dashboard`
    - Wait for page load and charts to render (wait for images or canvas elements)
    - Verify at least 4 visualization elements exist (images or charts)
    - Scroll through page to ensure all content loads
    - Take screenshot: `screenshots/04_dashboard.png`
  - **Test 5: Dashboard Charts Render**
    - On dashboard, verify specific chart images load successfully (check for broken images)
    - If interactive charts exist, verify they're rendered (check for canvas or SVG elements)
    - Take screenshot of each major chart section
  - **Test 6: Admin Page Load**
    - Navigate to `/admin`
    - Verify password input or submissions table exists
    - If password gate exists, enter "admin123" and submit
    - Verify access granted and submissions table appears
    - Take screenshot: `screenshots/05_admin_page.png`
  - **Test 7: Admin Submissions Table**
    - On admin page (after auth), verify submissions table has columns
    - Verify at least the test submissions from Test 2-3 appear in table
    - Take screenshot: `screenshots/06_admin_submissions.png`
  - **Test 8: Admin Validation Actions**
    - Click "Mark Correct" button on first submission
    - Wait for UI update
    - Verify validation status changes to "Correct" or shows checkmark
    - Click "Mark Incorrect" button on second submission
    - Verify validation status changes to "Incorrect" or shows X
    - Take screenshot: `screenshots/07_admin_validation.png`
  - **Test 9: Error Handling - Empty Tweet**
    - Navigate to home page
    - Click submit without entering text
    - Verify error message appears or button is disabled
    - Take screenshot: `screenshots/08_error_empty.png`
  - **Test 10: Database Persistence**
    - Navigate to admin page
    - Count number of submissions
    - Navigate to home, submit new tweet
    - Return to admin page
    - Verify submission count increased by 1
    - Take screenshot: `screenshots/09_persistence_check.png`
  - **Teardown**: Close browser, generate test report
- **Task 10.3**: Configure Puppeteer to use Chrome instead of Chromium:
  - Set `executablePath` to Chrome installation path (common Windows paths: `C:/Program Files/Google/Chrome/Application/chrome.exe` or `C:/Program Files (x86)/Google/Chrome/Application/chrome.exe`)
  - If Chrome not found at default paths, search in Program Files directories
  - If Chrome still not found, fall back to Puppeteer's bundled Chromium but log warning
- **Task 10.4**: Add retry logic to test script: if any test fails, retry up to 3 times with increasing wait times (2s, 5s, 10s). Log each retry attempt.
- **Task 10.5**: Add intelligent fix attempts: if element not found, try:
  - Waiting longer (increase timeout)
  - Scrolling to element
  - Checking for alternative selectors
  - Taking screenshot before declaring failure
- **Task 10.6**: Run test suite: `node tests/test_suite.js`. Capture all console output to `logs/puppeteer_tests.log`.
- **Task 10.7**: Parse test results:
  - Count passed tests
  - Count failed tests
  - List specific failures with error messages
  - Compile screenshots of failures
- **Task 10.8**: If any tests fail after retries:
  - Analyze failure screenshots
  - Check Flask server logs for errors
  - Verify database is accessible
  - Attempt automated fixes where possible (e.g., restart server, clear database, retry)
  - Document unfixable failures clearly
- **Task 10.9**: Generate test report: `tests/test_report.md` with:
  - Test execution timestamp
  - Total tests run
  - Passed/Failed counts
  - Detailed results for each test with screenshot references
  - Any errors or warnings encountered
  - Summary: "All tests passed" or "X tests failed, see details below"
- **Task 10.10**: If all critical tests pass (Tests 1-4, 6 at minimum), mark testing phase as successful. If critical tests fail, mark as failed but continue to final phase.

### Phase 11 Tasks

- **Task 11.1**: Validate all required deliverables exist:
  - [ ] `reports/analysis_results.md` (3000+ words)
  - [ ] `reports/laporan_project.md` (3000+ words, Indonesian)
  - [ ] `reports/python_notebook_documentation.md` (2000+ words with code)
  - [ ] `reports/presentation_outline.md` (15+ slides outlined)
  - [ ] `data/association_rules.csv`
  - [ ] `models/sentiment_model.pkl`
  - [ ] `models/vectorizer.pkl`
  - [ ] `visualizations/negative_tweets_wordcloud.png`
  - [ ] `visualizations/tweets_per_airline.png`
  - [ ] `visualizations/sentiment_by_airline.png`
  - [ ] `visualizations/overall_sentiment.png`
  - [ ] `visualizations/top_negative_reasons.png`
  - [ ] `visualizations/temporal_pattern.png`
  - [ ] `visualizations/airline_negativereason_heatmap.png`
  - [ ] `website/app.py` and all HTML/CSS/JS files
  - [ ] `tests/test_report.md`
  - [ ] MySQL database `airline_sentiment` with `submissions` table
- **Task 11.2**: Quick content validation:
  - Open each markdown report and verify it's not empty and has proper structure (headers present)
  - Verify English and Indonesian reports are approximately same length (within 20%)
  - Check that visualization files are valid images (file size > 10KB)
  - Verify model files can be loaded with joblib
- **Task 11.3**: Cross-reference consistency:
  - Ensure sentiment statistics in reports match actual data analysis
  - Verify dashboard visualizations match those referenced in reports
  - Check that model performance metrics are consistent across all reports
- **Task 11.4**: Create `PROJECT_SUMMARY.md` in root directory with:
  - Project title and date
  - Executive summary (3-4 sentences)
  - Deliverables checklist with file paths
  - Key findings (top 5 bullet points)
  - How to access the website (URL and credentials if needed)
  - How to run tests
  - Known issues or limitations
  - Contact/attribution information
- **Task 11.5**: Verify Flask server is still running. If not, restart it.
- **Task 11.6**: Determine final access port (5000 or alternative if changed).
- **Task 11.7**: Open default browser automatically with final URL: use Python's `webbrowser.open(f'http://localhost:{port}/')` or Windows command `start http://localhost:{port}/`.
- **Task 11.8**: Print final access information to console in clear, formatted output:
```
  ========================================
  🎉 DEPLOYMENT COMPLETE 🎉
  ========================================
  
  Your Twitter Airline Sentiment Analysis application is now running!
  
  📊 Access URLs:
     Home Page (Prediction): http://localhost:{port}/
     Dashboard (Analytics):  http://localhost:{port}/dashboard
     Admin Panel:            http://localhost:{port}/admin
  
  🔐 Admin Credentials:
     Password: admin123
  
  📁 Project Deliverables:
     • Analysis Report (EN):      reports/analysis_results.md
     • Laporan Project (ID):      reports/laporan_project.md
     • Notebook Documentation:    reports/python_notebook_documentation.md
     • Presentation Outline:      reports/presentation_outline.md
     • Association Rules:         data/association_rules.csv
     • ML Model:                  models/sentiment_model.pkl
     • Visualizations:            visualizations/ (7 charts + word cloud)
     • Test Report:               tests/test_report.md
  
  ✅ Test Results: {X/10} tests passed
  
  🗄️ Database: airline_sentiment (MySQL via XAMPP)
     Table: submissions ({X} entries)
  
  🛑 To Stop Server:
     Press Ctrl+C in this terminal or close this window
  
  📝 For detailed instructions, see:
     PROJECT_SUMMARY.md
     website/README_WEBSITE.md
  
  ========================================
  Server will continue running until manually stopped.
  Browser should open automatically. If not, copy the URL above.
  ========================================
```
- **Task 11.9**: Keep Flask server running in foreground (if using pythonw, it's already in background). Script should not exit. If running as orchestration script, use `input("Press Enter to stop server...")` to keep process alive until user manually stops it.
- **Task 11.10**: Log completion to `logs/execution.log` with timestamp and summary statistics.
- **Task 11.11**: If any phase had critical failures, print WARNING section listing what failed and suggesting fixes.

## EXECUTION GUIDELINES

- Work autonomously through all 12 phases (0-11) sequentially without waiting for user confirmation between phases.
- At the start of each phase, print clear phase header: `========== PHASE X: [PHASE NAME] ==========`
- Create internal markdown checklists using `- [ ]` syntax for task tracking within complex phases.
- Handle errors pragmatically:
  - Package installation fails → search for alternative packages intelligently, update requirements.txt
  - Database connection fails → retry with exponential backoff, attempt troubleshooting steps, document issue
  - Visualization generation fails → log error, skip that visualization, continue with others
  - Model training fails → try simpler model or reduce feature count, document limitation
  - Test fails → retry with increased timeouts, attempt automated fixes, take screenshots, document failure
- Do NOT halt execution unless:
  - Tweets.csv is completely missing or unreadable after retry attempts
  - Python environment is fundamentally broken (cannot import pandas/numpy)
  - All package installation alternatives exhausted for critical dependency
- Track progress by printing completion messages: `✓ Phase X Complete` after each phase
- For long-running tasks (model training, multiple visualizations), print incremental progress: `Training model... Step 1/3 complete`
- Prioritize completing as many deliverables as possible. If one component fails, document it and continue with others.
- All reports must be thorough and professional quality. Use proper markdown formatting, tables, embedded images.
- Indonesian report must use formal academic Bahasa Indonesia with proper grammar and terminology.
- Website must be functional, not just presentational. All APIs must work, all buttons must have actions.
- Testing must be comprehensive. Each test should verify actual functionality, not just page load.

## OUTPUT & DELIVERABLES

### Reports (reports/)
- **analysis_results.md** — Comprehensive English analysis report (3000-4500 words) with executive summary, methodology, findings, visualizations, insights, recommendations.
- **laporan_project.md** — Indonesian academic report (3000-4500 words) following standard structure: Judul, Abstrak, Pendahuluan, Tinjauan Pustaka, Metodologi, Hasil dan Pembahasan, Kesimpulan, Daftar Pustaka.
- **python_notebook_documentation.md** — Complete workflow documentation (2000-3000 words) in Jupyter notebook style with markdown explanations, code blocks, outputs.
- **presentation_outline.md** — Slide-by-slide presentation outline (15-20 slides) with bullet points and speaking notes for academic presentation.

### Data & Models (data/, models/)
- **association_rules.csv** — Association rules between airlines and negative reasons with support, confidence, lift metrics.
- **sentiment_model.pkl** — Trained sentiment classification model (Logistic Regression, Random Forest, or Naive Bayes).
- **vectorizer.pkl** — TF-IDF vectorizer fitted on training data.
- **model_info.json** — Model metadata including type, hyperparameters, performance metrics.
- **eda_summary.json** — Intermediate EDA findings for report generation.

### Visualizations (visualizations/)
- **negative_tweets_wordcloud.png** — Word cloud of most common words in negative tweets.
- **tweets_per_airline.png** — Bar chart of tweet volume by airline.
- **sentiment_by_airline.png** — Stacked bar chart of sentiment distribution per airline.
- **overall_sentiment.png** — Pie chart of overall sentiment distribution.
- **top_negative_reasons.png** — Bar chart of top 10 negative reasons.
- **temporal_pattern.png** — Line chart of tweet volume over time.
- **airline_negativereason_heatmap.png** — Heatmap of airline vs negative reason correlations.
- **sentiment_by_airline_interactive.html** — Interactive Plotly chart for dashboard.
- **temporal_pattern_interactive.html** — Interactive Plotly chart for dashboard.
- **visualization_summary.md** — Summary document listing all visualizations with descriptions.

### Website (website/)
- **app.py** — Flask application with routes for home, dashboard, admin, prediction API, submissions API, validation API.
- **config.py** — Database configuration parameters.
- **templates/base.html** — Base template with navigation and layout.
- **templates/index.html** — Home page with tweet input form and prediction display.
- **templates/dashboard.html** — Analytics dashboard with embedded visualizations.
- **templates/admin.html** — Admin panel with submissions table and validation interface.
- **static/css/style.css** — Custom styles for enhanced UI.
- **static/js/main.js** — JavaScript for home page form submission and prediction display.
- **static/js/admin.js** — JavaScript for admin panel functionality (fetch submissions, validation actions).
- **static/images/** — All visualization images copied for dashboard display.
- **README_WEBSITE.md** — Instructions for running and using the web application.

### Tests (tests/)
- **test_suite.js** — Puppeteer test script with 10 comprehensive tests.
- **test_report.md** — Test execution report with pass/fail results and screenshots.
- **screenshots/** — Screenshots from each test for verification and debugging.

### Logs (logs/)
- **execution.log** — Complete log of all phases, tasks, outputs, errors.
- **flask.log** — Flask server output and errors.
- **puppeteer_tests.log** — Puppeteer test execution output.

### Database (MySQL via XAMPP)
- **Database**: `airline_sentiment`
- **Table**: `submissions` with schema:
  - id (INT, PRIMARY KEY, AUTO_INCREMENT)
  - tweet_text (TEXT, NOT NULL)
  - predicted_sentiment (VARCHAR(20), NOT NULL)
  - confidence (FLOAT, NOT NULL)
  - timestamp (DATETIME, DEFAULT CURRENT_TIMESTAMP)
  - is_correct (BOOLEAN, NULL)
  - Indexes on timestamp and predicted_sentiment

### Root Directory Files
- **Tweets.csv** — Original dataset (provided by user)
- **requirements.txt** — All Python dependencies with versions
- **run_everything.py** — Master orchestration script
- **PROJECT_SUMMARY.md** — Index of all deliverables with descriptions and access instructions

## QUALITY STANDARDS (Self-Review Checklist)

Before concluding, verify:

### Data & Analysis
- [ ] Tweets.csv successfully loaded (verify row count matches expected ~14,000+ tweets)
- [ ] All 12 phases (0-11) attempted (mark which completed successfully)
- [ ] EDA completed with sentiment distribution, volume analysis, temporal patterns
- [ ] Sentiment analysis calculated per airline with confidence scores
- [ ] Negative reason analysis completed with frequency distributions
- [ ] Association rules generated with minimum 10 meaningful rules
- [ ] Word cloud created from negative tweets (file size > 50KB)
- [ ] All 7 required visualizations generated (check file sizes > 10KB each)

### Machine Learning
- [ ] Sentiment classification model trained (Logistic Regression, Random Forest, or Naive Bayes)
- [ ] Model performance metrics calculated (accuracy > 70% expected)
- [ ] Model and vectorizer serialized to .pkl files
- [ ] Model successfully loads in Flask app

### Reports & Documentation
- [ ] analysis_results.md exists and is 3000+ words
- [ ] analysis_results.md contains all required sections (Executive Summary through Limitations)
- [ ] Embedded visualizations in analysis_results.md render correctly
- [ ] laporan_project.md exists and is 3000+ words
- [ ] laporan_project.md uses proper formal Indonesian language
- [ ] English and Indonesian reports are content-synchronized (same findings, translated)
- [ ] python_notebook_documentation.md exists with code blocks and explanations
- [ ] presentation_outline.md has 15+ slides with bullet points and speaking notes
- [ ] All reports use proper markdown formatting (headers, tables, lists, image embeds)

### Database
- [ ] MySQL connection successful via XAMPP
- [ ] Database `airline_sentiment` created
- [ ] Table `submissions` created with correct schema
- [ ] Test insert/select operations work
- [ ] Flask app successfully connects to database

### Website
- [ ] Flask app.py exists with all 6 required routes (/, /dashboard, /admin, /predict, /submissions, /validate)
- [ ] All 3 HTML templates exist (index.html, dashboard.html, admin.html)
- [ ] Templates extend base.html correctly
- [ ] CSS and JavaScript files exist and are linked correctly
- [ ] Model and vectorizer load successfully on app startup
- [ ] /predict route accepts tweet text and returns prediction JSON
- [ ] Prediction preprocessing matches training preprocessing
- [ ] Dashboard displays all visualizations (7 static images + 2 interactive charts)
- [ ] Admin page shows submissions table
- [ ] Validation buttons functional (mark correct/incorrect)

### Deployment & Testing
- [ ] requirements.txt created with all dependencies
- [ ] Flask server starts successfully on port 5000 (or alternative)
- [ ] Server accessible via browser (GET / returns 200)
- [ ] Puppeteer installed (globally or locally)
- [ ] test_suite.js created with 10 tests
- [ ] Chrome executable path configured correctly (or Chromium fallback)
- [ ] Tests executed (at minimum 7/10 should pass for MVP)
- [ ] Test report generated with results
- [ ] Critical tests passed: Home page load, Prediction works, Dashboard loads, Admin loads
- [ ] Screenshots captured for all tests
- [ ] Browser opens automatically to application URL

### Final Deliverables
- [ ] PROJECT_SUMMARY.md created with deliverables index
- [ ] README_WEBSITE.md created with setup instructions
- [ ] All file paths in summary are correct and files exist
- [ ] Access URLs printed to console clearly
- [ ] Admin credentials provided (password: admin123)
- [ ] Server remains running after script completes
- [ ] Instructions provided for stopping server

### Error Handling & Documentation
- [ ] Any failed phases are documented in console output
- [ ] Any missing deliverables are noted with explanation
- [ ] Workarounds for failed components are documented
- [ ] Alternative packages used are noted in requirements.txt
- [ ] Known issues listed in PROJECT_SUMMARY.md

## ASSUMPTIONS & UNKNOWNS

### Data Assumptions
- **Assumed**: Tweets.csv exists in root directory with columns: tweet_id, text, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, airline, retweet_count, created, name, location, timezone
- **Assumed**: Dataset contains approximately 10,000-15,000 tweets (standard Twitter US Airline Sentiment dataset size)
- **Assumed**: Sentiment values are: "positive", "neutral", "negative" (lowercase or title case)
- **Assumed**: Airlines include: United, American, Southwest, Delta, US Airways, Virgin America
- **Handled**: If column names differ (capitalization, underscores), agent maps intelligently
- **Handled**: Missing values in negativereason and negativereason_confidence (expected for non-negative tweets)

### Environment Assumptions
- **Assumed**: Windows 10 or 11 operating system
- **Assumed**: XAMPP installed with MySQL running on localhost:3306
- **Assumed**: MySQL credentials: user=root, password='' (empty/default)
- **Assumed**: Python 3.7+ installed and in PATH
- **Assumed**: pip available for package installation
- **Assumed**: Node.js installed and in PATH
- **Assumed**: Google Chrome installed at default location (C:/Program Files/Google/Chrome/Application/chrome.exe)
- **Assumed**: Internet connection available for package downloads
- **Handled**: Python version detected automatically, code adapts to version
- **Handled**: If port 5000 occupied, auto-select next available port
- **Handled**: If Chrome not found, fallback to Puppeteer's Chromium

### Package & Dependency Assumptions
- **Assumed**: Can install via pip: flask, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, wordcloud, mlxtend, nltk, mysql-connector-python
- **Assumed**: Can install via npm: puppeteer (if not already installed globally)
- **Handled**: If mysql-connector-python fails, try pymysql as alternative
- **Handled**: If any visualization library fails, try alternative (matplotlib → seaborn)
- **Handled**: If wordcloud fails, skip word cloud generation and document it
- **Handled**: Package version conflicts resolved by pip automatically

### Website & Deployment Assumptions
- **Assumed**: Localhost deployment sufficient (no cloud hosting required)
- **Assumed**: Flask development server acceptable (not production WSGI server)
- **Assumed**: Simple password authentication sufficient for admin panel (password=admin123)
- **Assumed**: In-memory or MySQL database acceptable for submissions storage
- **Assumed**: Browser will be available to open URLs automatically
- **Handled**: Flask runs in background, does not block terminal
- **Handled**: Server keeps running until manually stopped by user

### Testing Assumptions
- **Assumed**: Puppeteer can control Chrome browser programmatically
- **Assumed**: Network latency for API calls < 5 seconds
- **Assumed**: Test submissions will persist in database for admin review
- **Handled**: If tests fail, retry with longer timeouts
- **Handled**: If element selectors change, try alternative selectors
- **Handled**: Screenshots captured for all tests regardless of pass/fail

### Report & Documentation Assumptions
- **Assumed**: Indonesian academic report follows standard thesis structure (Pendahuluan, Tinjauan Pustaka, etc.)
- **Assumed**: Presentation is for ~15-20 minute academic talk
- **Assumed**: Audience includes lecturer and fellow students (not industry professionals)
- **Assumed**: Markdown rendering available for viewing reports (GitHub, VS Code, etc.)
- **Handled**: Reports are self-contained (all images embedded or referenced locally)

### Known Limitations
- **Security**: Admin authentication is basic (client-side password check), not production-secure
- **Scalability**: SQLite/MySQL setup not optimized for high traffic
- **Model**: Sentiment model trained on provided dataset only, may not generalize to other contexts
- **Testing**: Automated tests do not cover all edge cases or UI states
- **Deployment**: Local deployment only, not containerized or cloud-ready
- **Language**: Indonesian translation may have minor grammatical variations (not professional translation)

### Unknown/Handled Gracefully
- **If Tweets.csv has different structure**: Agent attempts column mapping and documents discrepancies
- **If MySQL repeatedly fails**: Agent logs detailed errors, attempts all troubleshooting steps, documents failure, continues with other deliverables
- **If model training fails**: Agent tries simpler models, reduces features, documents performance limitations
- **If certain visualizations fail**: Agent skips failed chart, documents in reports, continues with successful ones
- **If Puppeteer tests fail**: Agent retries with fixes, documents failures, marks system as "partially functional"
- **If browser doesn't open**: Agent prints URLs clearly for manual access

## CRITICAL SUCCESS CRITERIA

The project is considered **SUCCESSFULLY COMPLETE** if:
1. ✅ All 4 markdown reports generated (analysis_results.md, laporan_project.md, python_notebook_documentation.md, presentation_outline.md)
2. ✅ At least 5 of 7 visualizations generated successfully
3. ✅ Sentiment classification model trained with accuracy > 65%
4. ✅ Association rules generated (minimum 5 rules)
5. ✅ MySQL database and table created
6. ✅ Flask website deployed and accessible via browser
7. ✅ Prediction API functional (can submit tweet and get sentiment prediction)
8. ✅ At least 6 of 10 Puppeteer tests pass
9. ✅ Server running and URLs printed to console
10. ✅ PROJECT_SUMMARY.md created

The project is considered **PARTIALLY COMPLETE** if:
- 3 of 4 reports generated, OR
- Model trained but accuracy < 65%, OR
- Website deployed but some features non-functional, OR
- Only 4-5 Puppeteer tests pass

If fewer than 5 critical criteria met, project is **INCOMPLETE** but agent must:
- Complete as much as possible
- Document all failures clearly
- Provide troubleshooting suggestions
- Generate partial deliverables report