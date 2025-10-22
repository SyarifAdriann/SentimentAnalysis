# Migration Guide (macOS)

This walkthrough assumes the machine only has a stock Python 3 installation (no Homebrew, pip packages, MySQL, Redis, etc.). Follow the steps exactly in order; each command is intended to be executed from the project root unless explicitly stated otherwise.

---

## 1. Install System Dependencies

1. **Install Homebrew** (package manager) if it is not already installed:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   After installation, add Homebrew to your shell:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

2. **Install Git** (needed for some tooling even if you won't commit):
   ```bash
   brew install git
   ```

3. **Install MySQL Community Server**:
   ```bash
   brew install mysql
   ```
   Start MySQL and enable it at login:
   ```bash
   brew services start mysql
   ```

4. **Install Redis** (Celery broker / result backend):
   ```bash
   brew install redis
   brew services start redis
   ```

5. **Python virtual environment tooling** ships with Python 3, so no extra install is required.

---

## 2. Prepare the Project

1. **Unzip the project** to the desired location, e.g. `~/Projects/SentimentAnalysis`.

2. **Move into the project directory**:
   ```bash
   cd ~/Projects/SentimentAnalysis
   ```

3. **Create a virtual environment** (named `venv` to match the repo layout):
   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment** (do this for every new terminal session working on the project):
   ```bash
   source venv/bin/activate
   ```

5. **Upgrade pip and install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 3. Configure Environment

1. **Ensure `.env` exists**. Copy from `.env.example` if necessary:
   ```bash
   cp .env.example .env
   ```

2. **Set credentials inside `.env`**, for example:
   ```text
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_mysql_password
   DB_NAME=airline_sentiment
   SECRET_KEY=somesecretkey
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=supervisor
   CELERY_BROKER_URL=redis://127.0.0.1:6379/0
   CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0
   ```

3. **Set (or reset) the MySQL root password** if prompted:
   ```bash
   mysqladmin -u root password 'your_mysql_password'
   ```

4. **Create the database** (if it does not exist):
   ```bash
   mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS airline_sentiment CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
   ```

5. **Import the SQL dump** that ships with the project (replace filename as needed):
   ```bash
   mysql -u root -p airline_sentiment < database_backup.sql
   ```

---

## 4. Initialize Application Metadata

With the virtual environment active (`source venv/bin/activate`):

1. **Run database initialization / migration script**:
   ```bash
   python -m scripts.init_model_versions
   ```

2. **Run automated tests** (optional but recommended):
   ```bash
   python -m pytest tests/test_continuous_learning.py -q
   ```

---

## 5. Launch the Application

Open two terminals (or tabs), both inside `~/Projects/SentimentAnalysis`.

### Terminal A – Flask Web App
```bash
source venv/bin/activate
python run.py
```
Serves `http://127.0.0.1:5000/`. Leave it running.

### Terminal B – Celery Worker
```bash
source venv/bin/activate
export CELERY_BROKER_URL='redis://127.0.0.1:6379/0'
export CELERY_RESULT_BACKEND='redis://127.0.0.1:6379/0'
python -m celery -A celery_app worker --loglevel=info
```
Keep this running to process background retraining jobs.

> **Note**: Do **not** launch `scripts/scheduled_retrain.py` unless you explicitly want automated nightly retraining. Manual retraining from the admin dashboard works without it.

---

## 6. Use the Application

1. Visit `http://127.0.0.1:5000/` to submit tweets and see predictions.
2. Log in to `http://127.0.0.1:5000/admin` with the credentials from `.env` to review and label submissions.
3. Approve submissions with the correct true sentiment so they become retraining data.
4. When ≥50 approved entries have `true_sentiment`, press **Retrain Model** on the admin dashboard; the Celery worker handles it in the background and shows the comparison modal when done.

---

## 7. Troubleshooting

- **ModuleNotFoundError**: Ensure `source venv/bin/activate` ran in the current shell.
- **MySQL errors**: Confirm `brew services list | grep mysql` shows `started`, credentials in `.env` are correct, and the database exists.
- **Redis/Celery errors**: Confirm `brew services list | grep redis` shows `started`, and environment variables are exported before launching Celery.
- **Rate-limit warning**: Safe to ignore in development; configure `LIMITER_STORAGE_URI=redis://127.0.0.1:6379/1` in `.env` for production.

---

## 8. Shutdown

- Stop Flask (`Ctrl+C` in Terminal A).
- Stop Celery (`Ctrl+C` in Terminal B).
- Optionally stop background services:
  ```bash
  brew services stop redis
  brew services stop mysql
  ```

You now have a fully operating clone ready to develop and test on macOS.
