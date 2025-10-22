"""Celery application configuration for background tasks."""

from __future__ import annotations

import os

from celery import Celery

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL)

celery = Celery(
    "sentiment_tasks",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

# Discover tasks within the project when running worker
celery.autodiscover_tasks(["tasks"])
