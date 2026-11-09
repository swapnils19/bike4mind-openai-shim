"""Pytest configuration and shared fixtures."""
import os
import pytest
from starlette.testclient import TestClient
from shim.main import app

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set default test environment variables."""
    monkeypatch.setenv("B4M_API_KEY", "b4m_live_test_dummy_key_abcdef")
    monkeypatch.setenv("B4M_BASE_URL", "https://app.bike4mind.com")
    monkeypatch.setenv("PORT", "8000")
    monkeypatch.setenv("HOST", "127.0.0.1")

@pytest.fixture
def client():
    """Synchronous FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client

