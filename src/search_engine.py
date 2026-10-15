"""In-memory TF-IDF cosine search over persisted or supplied documents."""

from __future__ import annotations

from typing import Any, Iterable


class SearchEngine:
    def __init__(self, documents: Iterable[dict[str, Any]] | None = None) -> None:
        self.documents: list[dict[str, Any]] = []
        self.vectorizer = None
        self.matrix = None
        if documents is not None: self.rebuild(documents)

    def rebuild(self, documents: Iterable[dict[str, Any]]) -> "SearchEngine":
        self.documents = [dict(document) for document in documents]
        if not self.documents:
            self.vectorizer = None
            self.matrix = None
            return self
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError("Search requires scikit-learn. Install it to build a search index.") from exc
        texts = [str(document.get("content", "")) for document in self.documents]
        self.vectorizer = TfidfVectorizer() if texts else None
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None
        return self

    build_index = rebuild

    def rebuild_from_database(self, database: Any) -> "SearchEngine":
        """Rebuild the index from an object exposing ``list_documents``."""
        return self.rebuild(database.list_documents())

    def add(self, document: dict[str, Any]) -> "SearchEngine":
        return self.rebuild([*self.documents, document])

    def search(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        if not query.strip() or self.matrix is None: return []
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        ranked = sorted(((float(score), index) for index, score in enumerate(scores) if score >= threshold), reverse=True)
        return [{**self.documents[index], "score": round(score, 6)} for score, index in ranked[:max(0, limit)]]

    query = search
