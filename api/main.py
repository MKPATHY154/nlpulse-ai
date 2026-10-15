"""NLPulse AI REST API for document analysis, persistence, and search."""

from __future__ import annotations

import shutil
import tempfile
import os
import csv
import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.database import Database
from src.ingestion import ExtractionError, extract_text
from src.pipeline import analyze_text
from src.gmail_connector import GmailConfigurationError, GmailConnector
from src.search_engine import SearchEngine


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = Path("/tmp") if os.environ.get("VERCEL") else ROOT / "data"
DATABASE = Database(DATA_DIR / "nlpulse.db")
SEARCH = SearchEngine()
SEARCH.rebuild_from_database(DATABASE)

app = FastAPI(
    title="NLPulse AI",
    description="Intelligent document and email understanding platform",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    name: str = Field(default="pasted-text", max_length=255)
    persist: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)


def _persist(name: str, text: str, result: dict[str, Any], source: str = "api") -> dict[str, Any]:
    document_id = DATABASE.create_document(
        name=name,
        content=text,
        source=source,
        category=result.get("classification", {}).get("label"),
        sentiment=result.get("sentiment", {}).get("label"),
        urgency=result.get("urgency", {}).get("level"),
    )
    DATABASE.save_analysis(document_id, result)
    for entity in result["entities"]:
        DATABASE.save_entity(document_id, entity["text"], entity["type"])
    SEARCH.add({"id": document_id, "name": name, "content": text, "source": source})
    return {"document_id": document_id, "analysis": result}


def _gmail_error(exc: GmailConfigurationError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/integrations/gmail/status")
def gmail_status() -> dict[str, Any]:
    return GmailConnector().status()


@app.get("/integrations/gmail/start")
def gmail_start() -> dict[str, str]:
    try:
        return {"authorization_url": GmailConnector().authorization_url()}
    except GmailConfigurationError as exc:
        raise _gmail_error(exc) from exc


@app.get("/integrations/gmail/callback")
def gmail_callback(code: str | None = None, error: str | None = None, state: str | None = None) -> dict[str, Any]:
    if error:
        raise HTTPException(status_code=400, detail=f"Gmail authorization failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Gmail callback did not include an authorization code")
    try:
        # Passing the code is sufficient for Google and avoids trusting unrelated query fields.
        return GmailConnector().exchange_callback(code=code)
    except GmailConfigurationError as exc:
        raise _gmail_error(exc) from exc


@app.post("/integrations/gmail/sync")
def gmail_sync(max_results: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    try:
        messages = GmailConnector().sync(max_results=max_results)
        imported = []
        for message in messages:
            text = "\n".join(filter(None, [message.get("subject", ""), message.get("sender", ""), message.get("body", "")]))
            result = analyze_text(text)
            imported.append({"message": message, **_persist(message.get("subject") or message.get("id") or "gmail-message", result["text"], result, source="gmail")})
        return {"synced": len(imported), "messages": imported}
    except GmailConfigurationError as exc:
        raise _gmail_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/integrations/gmail/disconnect")
def gmail_disconnect() -> dict[str, Any]:
    try:
        return GmailConnector().disconnect()
    except GmailConfigurationError as exc:
        raise _gmail_error(exc) from exc


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "documents": len(DATABASE.list_documents())}


@app.post("/analyze")
@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        result = analyze_text(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _persist(request.name, result["text"], result) if request.persist else result


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract, analyze, and persist a TXT, CSV, PDF, or DOCX upload."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".csv", ".pdf", ".docx"}:
        raise HTTPException(status_code=415, detail="Supported files: TXT, CSV, PDF, DOCX")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temporary_path = Path(handle.name)
            shutil.copyfileobj(file.file, handle)
        text = extract_text(temporary_path)
        result = analyze_text(text)
        return _persist(file.filename or "uploaded-document", result["text"], result, source="upload")
    except (ExtractionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@app.get("/documents")
def documents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    query: str | None = Query(default=None, max_length=500),
    search: str | None = Query(default=None, max_length=500),
    category: str | None = Query(default=None, max_length=100),
    sentiment: str | None = Query(default=None, max_length=50),
    urgency: str | None = Query(default=None, max_length=50),
    created_after: str | None = Query(default=None, max_length=40),
    created_before: str | None = Query(default=None, max_length=40),
    date_from: str | None = Query(default=None, max_length=40),
    date_to: str | None = Query(default=None, max_length=40),
) -> list[dict[str, Any]]:
    return DATABASE.list_documents(
        limit=limit,
        offset=offset,
        query=query or search,
        category=category,
        sentiment=sentiment,
        urgency=urgency,
        created_after=created_after or date_from,
        created_before=created_before or date_to,
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: int) -> dict[str, Any]:
    if not DATABASE.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    SEARCH.rebuild_from_database(DATABASE)
    return {"deleted": True, "document_id": document_id}


@app.get("/analytics/overview")
def analytics_overview(recent_limit: int = Query(default=10, ge=0, le=100)) -> dict[str, Any]:
    return DATABASE.analytics_overview(recent_limit=recent_limit)


def _csv_value(value: Any) -> str:
    """Prevent spreadsheet formula execution while retaining valid CSV data."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


@app.get("/documents/export.csv")
def export_documents_csv(
    query: str | None = Query(default=None, max_length=500),
    search: str | None = Query(default=None, max_length=500),
    category: str | None = Query(default=None, max_length=100),
    sentiment: str | None = Query(default=None, max_length=50),
    urgency: str | None = Query(default=None, max_length=50),
    created_after: str | None = Query(default=None, max_length=40),
    created_before: str | None = Query(default=None, max_length=40),
    date_from: str | None = Query(default=None, max_length=40),
    date_to: str | None = Query(default=None, max_length=40),
) -> Response:
    documents = DATABASE.list_documents(
        limit=None,
        query=query or search,
        category=category,
        sentiment=sentiment,
        urgency=urgency,
        created_after=created_after or date_from,
        created_before=created_before or date_to,
    )
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    fields = ("id", "name", "content", "source", "created_at", "category", "sentiment", "urgency")
    writer.writerow(fields)
    for item in documents:
        writer.writerow([_csv_value(item.get(field)) for field in fields])
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=documents.csv"},
    )


@app.get("/documents/{document_id}")
def document(document_id: int) -> dict[str, Any]:
    item = DATABASE.get_document(document_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    item["analyses"] = DATABASE.list_analyses(document_id)
    item["entities"] = DATABASE.list_entities(document_id)
    return item


@app.post("/search")
@app.post("/api/search")
def search(request: SearchRequest) -> list[dict[str, Any]]:
    return SEARCH.search(request.query, limit=request.limit)


@app.post("/models/train")
@app.post("/api/models/train")
def train_model(algorithm: str = Query(default="logistic")) -> dict[str, Any]:
    """Train the optional learned classifier using its configured dataset."""
    from src.ml_classifier import TextClassifier

    try:
        model = TextClassifier(algorithm=algorithm).train()
        return {"status": "trained", "algorithm": model.algorithm, "classes": sorted(model.model.classes_)}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/docs", include_in_schema=False)
def api_docs() -> RedirectResponse:
    """Support the browser-friendly API documentation path used by the UI."""
    return RedirectResponse(url="/docs")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def browser_app() -> HTMLResponse:
    """Serve the browser interface when Vercel routes the root to Python."""
    page = ROOT / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>NLPulse AI</h1><p>API is running. Open /docs.</p>")
