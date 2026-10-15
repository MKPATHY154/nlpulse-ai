"""Gmail configuration tests that do not contact Google."""

from pathlib import Path

from src.gmail_connector import GmailConnector, GmailConfigurationError


def test_status_reports_unconfigured_without_credentials(tmp_path: Path) -> None:
    status = GmailConnector(environ={}).status()
    assert status["configured"] is False
    assert status["connected"] is False
    assert "GMAIL_CLIENT_ID" in status["missing_configuration"]


def test_operations_raise_clear_configuration_error() -> None:
    connector = GmailConnector(environ={})
    try:
        connector.authorization_url()
    except GmailConfigurationError as exc:
        assert "GMAIL_CLIENT_ID" in str(exc)
    else:
        raise AssertionError("expected missing Gmail configuration error")


def test_parse_message_extracts_plain_text_and_headers() -> None:
    import base64

    encoded = base64.urlsafe_b64encode(b"Hello from Gmail").decode()
    message = {"id": "abc", "threadId": "thread", "snippet": "Hello", "payload": {"headers": [{"name": "Subject", "value": "Welcome"}, {"name": "From", "value": "a@example.com"}], "mimeType": "text/plain", "body": {"data": encoded}}}
    parsed = GmailConnector.parse_message(message)
    assert parsed["subject"] == "Welcome"
    assert parsed["sender"] == "a@example.com"
    assert parsed["body"] == "Hello from Gmail"
