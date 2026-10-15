"""NLPulse AI REST API for document analysis, persistence, and search."""

from __future__ import annotations

import shutil
import tempfile
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.database import Database
from src.ingestion import ExtractionError, extract_text
from src.pipeline import analyze_text
from src.search_engine import SearchEngine


ROOT = Path(__file__).resolve().parents[1]
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
    document_id = DATABASE.create_document(name=name, content=text, source=source)
    DATABASE.save_analysis(document_id, result)
    for entity in result["entities"]:
        DATABASE.save_entity(document_id, entity["text"], entity["type"])
    SEARCH.add({"id": document_id, "name": name, "content": text, "source": source})
    return {"document_id": document_id, "analysis": result}


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
def documents(limit: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0)) -> list[dict[str, Any]]:
    return DATABASE.list_documents(limit=limit, offset=offset)


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
