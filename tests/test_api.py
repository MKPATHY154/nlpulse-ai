"""Smoke tests for the public FastAPI contract."""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_without_persistence_returns_analysis() -> None:
    response = client.post(
        "/analyze",
        json={"text": "Please help me fix this login error.", "persist": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["label"] == "request"
    assert "summary" in payload


def test_invalid_empty_text_is_rejected() -> None:
    response = client.post("/analyze", json={"text": "   ", "persist": False})
    assert response.status_code == 422
