"""Database utilities for MySQL setup and interactions."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mysql.connector
from mysql.connector import Error

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "airline_sentiment"),
        )


def get_connection(
    config: Optional[DatabaseConfig] = None,
    *,
    include_db: bool = True,
):
    if config is None:
        config = DatabaseConfig.from_env()

    connection_params = {
        "host": config.host,
        "port": config.port,
        "user": config.user,
        "password": config.password,
    }

    if include_db:
        connection_params["database"] = config.database

    return mysql.connector.connect(**connection_params)


def create_model_versions_table(cursor) -> None:
    """Create table to track model versions and metrics."""

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            version_number INT NOT NULL,
            accuracy DECIMAL(6,4) NOT NULL,
            precision_macro DECIMAL(6,4) DEFAULT NULL,
            recall_macro DECIMAL(6,4) DEFAULT NULL,
            f1_macro DECIMAL(6,4) DEFAULT NULL,
            training_samples INT NOT NULL,
            model_path VARCHAR(255) NOT NULL,
            metrics_path VARCHAR(255) DEFAULT NULL,
            is_active BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE KEY unique_version (version_number)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )


def create_prediction_logs_table(cursor) -> None:
    """Create table for tracking per-submission model performance."""

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            submission_id INT NOT NULL,
            model_version INT NOT NULL,
            was_correct BOOLEAN NOT NULL,
            logged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_submission_model (submission_id, model_version),
            CONSTRAINT fk_prediction_logs_submission
                FOREIGN KEY (submission_id) REFERENCES submissions(id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )


def ensure_additional_indexes(cursor) -> None:
    """Create performance indexes, ignoring duplicates."""

    statements = (
        (
            "ALTER TABLE submissions ADD INDEX idx_review_status_true_sentiment (review_status, true_sentiment)",
        ),
        (
            "ALTER TABLE submissions ADD INDEX idx_updated_at (updated_at)",
        ),
        (
            "ALTER TABLE model_versions ADD INDEX idx_is_active (is_active)",
        ),
    )

    for (statement,) in statements:
        try:
            cursor.execute(statement)
        except Error as exc:  # Duplicate index -> ignore
            if getattr(exc, "errno", None) == 1061:
                continue
            LOGGER.debug("Skipping index statement due to error: %s", exc)


def initialize_database() -> None:
    """Initialise database and ensure required tables exist."""

    config = DatabaseConfig.from_env()
    LOGGER.info("Connecting to MySQL at %s:%s", config.host, config.port)

    try:
        with get_connection(config, include_db=False) as conn:
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.close()
    except Error as exc:
        LOGGER.error("Failed to create database: %s", exc)
        raise

    submissions_sql = (
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tweet_text TEXT NOT NULL,
            predicted_sentiment VARCHAR(20) NOT NULL,
            prediction_confidence DECIMAL(5,4) DEFAULT NULL,
            assigned_airline VARCHAR(50) DEFAULT NULL,
            true_sentiment VARCHAR(20) DEFAULT NULL,
            review_status ENUM('pending','approved','rejected') DEFAULT 'pending',
            admin_comment TEXT DEFAULT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_review_status (review_status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;
        """
    )

    with get_connection(config, include_db=True) as conn:
        cursor = conn.cursor()
        cursor.execute(submissions_sql)
        create_model_versions_table(cursor)
        create_prediction_logs_table(cursor)
        ensure_additional_indexes(cursor)
        conn.commit()
        cursor.close()

    LOGGER.info("Database '%s' initialised with required tables.", config.database)


def insert_submission(
    *,
    tweet_text: str,
    predicted_sentiment: str,
    prediction_confidence: Optional[float],
    assigned_airline: Optional[str] = None,
    true_sentiment: Optional[str] = None,
) -> int:
    config = DatabaseConfig.from_env()
    sql = (
        """
        INSERT INTO submissions (
            tweet_text,
            predicted_sentiment,
            prediction_confidence,
            assigned_airline,
            true_sentiment
        ) VALUES (%s, %s, %s, %s, %s)
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (
                tweet_text,
                predicted_sentiment,
                prediction_confidence,
                assigned_airline,
                true_sentiment,
            ),
        )
        conn.commit()
        inserted_id = cursor.lastrowid
        cursor.close()

    return inserted_id


def fetch_pending_submissions() -> list[Dict[str, Any]]:
    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT id, tweet_text, predicted_sentiment, prediction_confidence,
               assigned_airline, true_sentiment, review_status,
               admin_comment, created_at, updated_at
        FROM submissions
        WHERE review_status = 'pending'
        ORDER BY created_at DESC
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

    return rows


def fetch_recent_submissions(limit: int = 5) -> list[Dict[str, Any]]:
    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT id, tweet_text, predicted_sentiment, prediction_confidence,
               assigned_airline, true_sentiment, review_status,
               admin_comment, created_at, updated_at
        FROM submissions
        ORDER BY created_at DESC
        LIMIT %s
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        cursor.close()

    return rows


def get_submission_by_id(submission_id: int) -> Optional[Dict[str, Any]]:
    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT *
        FROM submissions
        WHERE id = %s
        LIMIT 1
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (submission_id,))
        row = cursor.fetchone()
        cursor.close()

    return row



def update_submission_status(
    *,
    submission_id: int,
    status: str,
    true_sentiment: Optional[str] = None,
    admin_comment: Optional[str] = None,
) -> None:
    config = DatabaseConfig.from_env()
    sql = (
        """
        UPDATE submissions
        SET review_status = %s,
            true_sentiment = %s,
            admin_comment = %s
        WHERE id = %s
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (status, true_sentiment, admin_comment, submission_id))
        conn.commit()
        cursor.close()


def fetch_all_submissions_grouped(limit: int = 1000) -> Dict[str, list[Dict[str, Any]]]:
    """Fetch submissions grouped by review status."""

    config = DatabaseConfig.from_env()
    results: Dict[str, list[Dict[str, Any]]] = {"pending": [], "approved": [], "rejected": []}

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        for status in results.keys():
            order_by = "created_at" if status == "pending" else "updated_at"
            query = f"""
                SELECT * FROM submissions
                WHERE review_status = %s
                ORDER BY {order_by} DESC
                LIMIT %s
            """
            cursor.execute(query, (status, limit))
            rows = cursor.fetchall()
            results[status] = rows
        cursor.close()

    return results


def count_approved_with_true_sentiment() -> int:
    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT COUNT(*)
        FROM submissions
        WHERE review_status = 'approved'
          AND true_sentiment IS NOT NULL
          AND true_sentiment != ''
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        (count,) = cursor.fetchone()
        cursor.close()

    return int(count)


def count_pending_submissions() -> int:
    config = DatabaseConfig.from_env()
    query = "SELECT COUNT(*) FROM submissions WHERE review_status = 'pending'"

    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        (count,) = cursor.fetchone()
        cursor.close()

    return int(count)


def fetch_training_data() -> list[Dict[str, Any]]:
    """Fetch approved submissions that have ground truth labels."""

    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT tweet_text, true_sentiment AS sentiment
        FROM submissions
        WHERE review_status = 'approved'
          AND true_sentiment IS NOT NULL
          AND true_sentiment != ''
          AND tweet_text IS NOT NULL
        ORDER BY updated_at ASC
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

    return rows


def save_model_version(
    *,
    version_num: int,
    metrics: Dict[str, Any],
    model_path: str,
    training_samples: int,
    notes: Optional[str] = None,
    is_active: bool = False,
    metrics_path: Optional[str] = None,
) -> None:
    config = DatabaseConfig.from_env()
    sql = (
        """
        INSERT INTO model_versions (
            version_number,
            accuracy,
            precision_macro,
            recall_macro,
            f1_macro,
            training_samples,
            model_path,
            metrics_path,
            is_active,
            notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    )

    macro_metrics = metrics.get("macro avg", {}) if isinstance(metrics, dict) else {}

    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (
                version_num,
                metrics.get("accuracy", 0),
                macro_metrics.get("precision", 0),
                macro_metrics.get("recall", 0),
                macro_metrics.get("f1-score", 0),
                training_samples,
                model_path,
                metrics_path,
                is_active,
                notes,
            ),
        )
        conn.commit()
        cursor.close()


def get_latest_model_version() -> Optional[Dict[str, Any]]:
    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT *
        FROM model_versions
        WHERE is_active = TRUE
        ORDER BY version_number DESC
        LIMIT 1
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()

    return result


def set_active_model(version_number: int) -> None:
    config = DatabaseConfig.from_env()
    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE model_versions SET is_active = FALSE")
        cursor.execute(
            """
            UPDATE model_versions
            SET is_active = TRUE
            WHERE version_number = %s
            """,
            (version_number,),
        )
        conn.commit()
        cursor.close()


def get_all_model_versions() -> list[Dict[str, Any]]:
    config = DatabaseConfig.from_env()
    query = (
        """
        SELECT *
        FROM model_versions
        ORDER BY version_number DESC
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

    return rows


def log_prediction_performance(
    *,
    submission_id: int,
    model_version: int,
    was_correct: bool,
) -> None:
    """Persist evaluation outcome for a reviewed prediction."""

    config = DatabaseConfig.from_env()
    sql = (
        """
        INSERT INTO prediction_logs (submission_id, model_version, was_correct)
        VALUES (%s, %s, %s)
        """
    )

    with get_connection(config) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (submission_id, model_version, was_correct))
        conn.commit()
        cursor.close()

