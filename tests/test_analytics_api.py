"""Tests for document filters, analytics, deletion, and CSV export."""

import csv
import io

import pytest
from fastapi.testclient import TestClient

from api import main
from src.database import Database
from src.search_engine import SearchEngine


@pytest.fixture()
def client(tmp_path, monkeypatch):
    database = Database(tmp_path / "analytics.db")
    monkeypatch.setattr(main, "DATABASE", database)
    monkeypatch.setattr(main, "SEARCH", SearchEngine())
    return TestClient(main.app)


def _create(client: TestClient, text: str, name: str) -> int:
    response = client.post("/analyze", json={"name": name, "text": text})
    assert response.status_code == 200
    return response.json()["document_id"]


def test_filters_analytics_and_delete(client: TestClient) -> None:
    negative_id = _create(client, "Urgent: the broken outage is terrible.", "incident.txt")
    positive_id = _create(client, "Thanks, the helpful fix is excellent.", "praise.txt")

    response = client.get("/documents", params={"sentiment": "negative", "urgency": "critical"})
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [negative_id]

    overview = client.get("/analytics/overview").json()
    assert overview["document_count"] == 2
    assert overview["sentiment_distribution"]["negative"] == 1
    assert overview["sentiment_distribution"]["positive"] == 1
    assert len(overview["recent_documents"]) == 2

    deleted = client.delete(f"/documents/{negative_id}")
    assert deleted.status_code == 200
    assert client.get(f"/documents/{negative_id}").status_code == 404
    assert client.delete(f"/documents/{negative_id}").status_code == 404


def test_csv_export_is_quoted_and_filters_rows(client: TestClient) -> None:
    _create(client, "Please review, now", "name,with,commas.txt")
    _create(client, "A neutral note", "neutral.txt")

    response = client.get("/documents/export.csv", params={"urgency": "medium"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 1
    assert rows[0]["name"] == "name,with,commas.txt"
    assert rows[0]["urgency"] == "medium"
