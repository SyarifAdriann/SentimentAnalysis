"""Extension instances for the Flask app."""

from __future__ import annotations

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Instantiate without app; initialise in factory.
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
