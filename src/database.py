"""Small SQLite persistence layer for documents and their NLP analyses."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


class Database:
    """SQLite-backed repository. Connections are short-lived and thread-safe."""

    def __init__(self, path: str | Path = "nlpulse.db") -> None:
        self.path = str(path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
            self._memory_connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                    content TEXT NOT NULL, source TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    category TEXT, sentiment TEXT, urgency TEXT
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER NOT NULL,
                    result TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER NOT NULL,
                    text TEXT NOT NULL, type TEXT NOT NULL, metadata TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(name);
                CREATE INDEX IF NOT EXISTS idx_entities_document ON entities(document_id);
            """)
            # Keep databases created by earlier releases compatible with the
            # analytics fields without requiring a destructive migration.
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            for column in ("category", "sentiment", "urgency"):
                if column not in columns:
                    db.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")
            db.executescript("""
                CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
                CREATE INDEX IF NOT EXISTS idx_documents_sentiment ON documents(sentiment);
                CREATE INDEX IF NOT EXISTS idx_documents_urgency ON documents(urgency);
                CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
            """)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def create_document(
        self,
        name: str,
        content: str,
        source: str | None = None,
        category: str | None = None,
        sentiment: str | None = None,
        urgency: str | None = None,
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO documents(name, content, source, category, sentiment, urgency) VALUES (?, ?, ?, ?, ?, ?)",
                (name, content, source, category, sentiment, urgency),
            )
            return int(cursor.lastrowid)

    save_document = create_document

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            return self._row(db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone())

    def list_documents(
        self,
        limit: int | None = None,
        offset: int = 0,
        query: str | None = None,
        category: str | None = None,
        sentiment: str | None = None,
        urgency: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[dict[str, Any]]:
        # The latest analysis is used as a fallback for documents written by
        # older versions before the denormalized filter columns existed.
        query_sql = """
            SELECT d.id, d.name, d.content, d.source, d.created_at,
                COALESCE(d.category, CASE WHEN json_valid(a.result)
                THEN json_extract(a.result, '$.classification.label') END) AS category,
                COALESCE(d.sentiment, CASE WHEN json_valid(a.result)
                THEN json_extract(a.result, '$.sentiment.label') END) AS sentiment,
                COALESCE(d.urgency, CASE WHEN json_valid(a.result)
                THEN json_extract(a.result, '$.urgency.level') END) AS urgency
            FROM documents d
            LEFT JOIN analyses a ON a.id = (SELECT MAX(id) FROM analyses WHERE document_id = d.id)
            WHERE 1 = 1
        """
        params: list[Any] = []
        if query:
            query_sql += " AND (d.name LIKE ? OR d.content LIKE ?)"
            term = f"%{query}%"
            params.extend((term, term))
        for column, value in (("category", category), ("sentiment", sentiment), ("urgency", urgency)):
            if value:
                json_path = {"category": "classification.label", "sentiment": "sentiment.label", "urgency": "urgency.level"}[column]
                query_sql += f" AND COALESCE(d.{column}, CASE WHEN json_valid(a.result) THEN json_extract(a.result, '$.{json_path}') END) = ?"
                params.append(value)
        if created_after:
            query_sql += " AND d.created_at >= ?"
            params.append(created_after)
        if created_before:
            query_sql += " AND d.created_at <= ?"
            params.append(created_before)
        query_sql += " ORDER BY d.created_at DESC, d.id DESC"
        if limit is not None:
            query_sql += " LIMIT ? OFFSET ?"
            params.extend((max(0, limit), max(0, offset)))
        with self._connect() as db:
            return [dict(row) for row in db.execute(query_sql, tuple(params))]

    def search_documents(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.list_documents(limit=limit, query=query)

    def delete_document(self, document_id: int) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def analytics_overview(self, recent_limit: int = 10) -> dict[str, Any]:
        documents = self.list_documents(limit=None)

        def distribution(field: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for document in documents:
                value = document.get(field) or "unknown"
                counts[str(value)] = counts.get(str(value), 0) + 1
            return dict(sorted(counts.items()))

        return {
            "document_count": len(documents),
            "category_distribution": distribution("category"),
            "sentiment_distribution": distribution("sentiment"),
            "urgency_distribution": distribution("urgency"),
            "recent_documents": documents[:max(0, recent_limit)],
        }

    def save_analysis(self, document_id: int, result: dict[str, Any] | str) -> int:
        payload = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        with self._connect() as db:
            cursor = db.execute("INSERT INTO analyses(document_id, result) VALUES (?, ?)", (document_id, payload))
            return int(cursor.lastrowid)

    def list_analyses(self, document_id: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM analyses WHERE document_id = ? ORDER BY id DESC" if document_id else "SELECT * FROM analyses ORDER BY id DESC", (document_id,) if document_id else ()).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            try: item["result"] = json.loads(item["result"])
            except (TypeError, json.JSONDecodeError): pass
            output.append(item)
        return output

    def get_analysis(self, analysis_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = self._row(db.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone())
        if row:
            try: row["result"] = json.loads(row["result"])
            except (TypeError, json.JSONDecodeError): pass
        return row

    def save_entity(self, document_id: int, text: str, entity_type: str, metadata: dict[str, Any] | None = None) -> int:
        with self._connect() as db:
            cursor = db.execute("INSERT INTO entities(document_id, text, type, metadata) VALUES (?, ?, ?, ?)", (document_id, text, entity_type, json.dumps(metadata) if metadata else None))
            return int(cursor.lastrowid)

    def list_entities(self, document_id: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM entities WHERE document_id = ? ORDER BY id" if document_id else "SELECT * FROM entities ORDER BY id", (document_id,) if document_id else ()).fetchall()
        return [dict(row) for row in rows]
