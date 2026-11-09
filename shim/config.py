"""Configuration management for Bike4Mind OpenAI Shim."""
import os
import sys
import logging

logger = logging.getLogger("bike4mind_shim")

def get_api_key(required: bool = True) -> str:
    """Retrieve B4M_API_KEY from environment. Fail fast if required and missing."""
    key = os.getenv("B4M_API_KEY", "").strip()
    if required and not key:
        err_msg = (
            "CRITICAL: B4M_API_KEY environment variable is missing or empty. "
            "Please export B4M_API_KEY=<your_bike4mind_api_key> before running the service."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    return key

def get_base_url() -> str:
    """Retrieve upstream Bike4Mind base URL."""
    url = os.getenv("B4M_BASE_URL", "https://app.bike4mind.com").strip()
    return url.rstrip("/")

def get_port() -> int:
    """Retrieve port from environment (default 8000)."""
    raw_port = os.getenv("PORT", "8000").strip()
    try:
        return int(raw_port)
    except ValueError:
        logger.warning("Invalid PORT environment variable '%s', defaulting to 8000", raw_port)
        return 8000

def get_host() -> str:
    """Retrieve host to bind to (default 127.0.0.1)."""
    return os.getenv("HOST", "127.0.0.1").strip()
