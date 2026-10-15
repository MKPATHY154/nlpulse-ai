"""Optional, read-only Gmail integration.

Google client libraries are imported only when a Gmail operation is used so
the rest of NLPulse remains usable without Gmail being configured.
"""

from __future__ import annotations

import base64
import os
import pickle
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


class GmailConfigurationError(RuntimeError):
    """Raised when Gmail cannot be used with the current configuration."""


class GmailConnector:
    """Small wrapper around Google's Gmail API with read-only permissions."""

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self.environ = environ if environ is not None else os.environ

    def _settings(self) -> dict[str, str]:
        required = ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REDIRECT_URI", "GMAIL_TOKEN_PATH")
        missing = [name for name in required if not self.environ.get(name)]
        if missing:
            raise GmailConfigurationError(
                "Missing Gmail configuration: " + ", ".join(missing) + ". Set these environment variables before connecting."
            )
        return {name: self.environ[name] for name in required}

    @staticmethod
    def _google_modules() -> tuple[Any, Any, Any, Any]:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import Flow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise GmailConfigurationError(
                "Gmail support requires google-api-python-client, google-auth-oauthlib, and google-auth-httplib2."
            ) from exc
        return Request, Credentials, Flow, build

    def status(self) -> dict[str, Any]:
        settings_missing = [name for name in ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REDIRECT_URI", "GMAIL_TOKEN_PATH") if not self.environ.get(name)]
        token_path = self.environ.get("GMAIL_TOKEN_PATH")
        return {
            "configured": not settings_missing,
            "connected": bool(token_path and Path(token_path).exists()),
            "token_path": token_path,
            "missing_configuration": settings_missing,
            "scope": GMAIL_READONLY_SCOPE,
        }

    def _client_config(self, settings: dict[str, str]) -> dict[str, Any]:
        return {"web": {"client_id": settings["GMAIL_CLIENT_ID"], "client_secret": settings["GMAIL_CLIENT_SECRET"], "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [settings["GMAIL_REDIRECT_URI"]]}}

    def authorization_url(self) -> str:
        settings = self._settings()
        _, _, Flow, _ = self._google_modules()
        flow = Flow.from_client_config(self._client_config(settings), scopes=[GMAIL_READONLY_SCOPE], redirect_uri=settings["GMAIL_REDIRECT_URI"])
        url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
        return url

    def exchange_callback(self, callback_url: str | None = None, code: str | None = None) -> dict[str, Any]:
        settings = self._settings()
        _, _, Flow, _ = self._google_modules()
        flow = Flow.from_client_config(self._client_config(settings), scopes=[GMAIL_READONLY_SCOPE], redirect_uri=settings["GMAIL_REDIRECT_URI"])
        if callback_url:
            flow.fetch_token(authorization_response=callback_url)
        elif code:
            flow.fetch_token(code=code)
        else:
            raise GmailConfigurationError("Gmail callback requires an authorization code.")
        token_path = Path(settings["GMAIL_TOKEN_PATH"])
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with token_path.open("wb") as handle:
            pickle.dump(flow.credentials, handle)
        return {"connected": True, "token_path": str(token_path)}

    def disconnect(self) -> dict[str, Any]:
        settings = self._settings()
        path = Path(settings["GMAIL_TOKEN_PATH"])
        existed = path.exists()
        path.unlink(missing_ok=True)
        return {"disconnected": True, "removed": existed}

    def _service(self) -> Any:
        settings = self._settings()
        Request, Credentials, _, build = self._google_modules()
        path = Path(settings["GMAIL_TOKEN_PATH"])
        if not path.exists():
            raise GmailConfigurationError("Gmail is not connected. Visit the Gmail authorization URL first.")
        try:
            with path.open("rb") as handle:
                credentials = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise GmailConfigurationError("The Gmail token file could not be read; disconnect and reconnect Gmail.") from exc
        if getattr(credentials, "expired", False) and getattr(credentials, "refresh_token", None):
            credentials.refresh(Request())
            with path.open("wb") as handle:
                pickle.dump(credentials, handle)
        if not getattr(credentials, "valid", True):
            raise GmailConfigurationError("The Gmail credentials are invalid; disconnect and reconnect Gmail.")
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    @staticmethod
    def _decode(data: str | None) -> str:
        if not data:
            return ""
        try:
            return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")
        except (ValueError, UnicodeDecodeError):
            return ""

    @classmethod
    def parse_message(cls, message: dict[str, Any]) -> dict[str, Any]:
        payload = message.get("payload") or {}
        headers = {str(item.get("name", "")).lower(): item.get("value", "") for item in payload.get("headers", [])}
        plain_parts: list[str] = []
        html_parts: list[str] = []

        def walk(part: dict[str, Any]) -> None:
            mime = part.get("mimeType", "")
            body = cls._decode((part.get("body") or {}).get("data"))
            if body and mime == "text/plain":
                plain_parts.append(body)
            elif body and mime == "text/html":
                html_parts.append(body)
            for child in part.get("parts", []) or []:
                walk(child)

        walk(payload)
        body = "\n".join(plain_parts).strip() or "\n".join(html_parts).strip() or str(message.get("snippet", ""))
        date = headers.get("date", "")
        try:
            date = parsedate_to_datetime(date).isoformat() if date else ""
        except (TypeError, ValueError):
            pass
        return {"id": message.get("id"), "thread_id": message.get("threadId"), "subject": headers.get("subject", ""), "sender": headers.get("from", ""), "to": headers.get("to", ""), "date": date, "snippet": message.get("snippet", ""), "body": body}

    def sync(self, max_results: int = 50) -> list[dict[str, Any]]:
        service = self._service()
        response = service.users().messages().list(userId="me", maxResults=max_results).execute()
        messages = []
        for item in response.get("messages", []):
            raw = service.users().messages().get(userId="me", id=item["id"], format="full").execute()
            messages.append(self.parse_message(raw))
        return messages


__all__ = ["GMAIL_READONLY_SCOPE", "GmailConfigurationError", "GmailConnector"]
