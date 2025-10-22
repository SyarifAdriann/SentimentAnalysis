"""Flask application factory."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask
from werkzeug.security import generate_password_hash

from app.extensions import limiter

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
    app.config.setdefault(
        "CELERY_BROKER_URL",
        os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    )
    app.config.setdefault(
        "CELERY_RESULT_BACKEND",
        os.getenv("CELERY_RESULT_BACKEND", app.config["CELERY_BROKER_URL"]),
    )

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password_hash = os.getenv("ADMIN_PASSWORD_HASH")
    admin_password = os.getenv("ADMIN_PASSWORD", "supervisor")
    app.config["ADMIN_USERNAME"] = admin_username
    app.config["ADMIN_PASSWORD_HASH"] = (
        admin_password_hash
        if admin_password_hash
        else generate_password_hash(admin_password)
    )

    limiter.init_app(app)

    from . import routes  # noqa: WPS433

    routes.init_app(app)
    return app
