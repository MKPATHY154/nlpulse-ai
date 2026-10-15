"""CPU-friendly learned text classifier and labeled-CSV utilities."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Sequence

DEFAULT_TRAINING_DATA: tuple[tuple[str, str], ...] = (
    ("please help me fix this problem", "request"),
    ("I need assistance with my account", "request"),
    ("the application is broken and shows an error", "complaint"),
    ("this service is terrible", "complaint"),
    ("what time does support open", "question"),
    ("how can I reset my password", "question"),
    ("I love the new dashboard", "feedback"),
    ("please improve the search experience", "feedback"),
)


def load_labeled_csv(
    path: str | Path, text_column: str = "text", label_column: str = "label"
) -> tuple[list[str], list[str]]:
    """Read and validate a UTF-8 CSV containing text and label columns."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or text_column not in reader.fieldnames or label_column not in reader.fieldnames:
                raise ValueError(f"CSV must contain '{text_column}' and '{label_column}' columns")
            texts, labels = [], []
            for line_number, row in enumerate(reader, start=2):
                text = (row.get(text_column) or "").strip()
                label = (row.get(label_column) or "").strip()
                if not text or not label:
                    raise ValueError(f"CSV row {line_number} must have non-empty text and label")
                texts.append(text)
                labels.append(label)
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded") from exc
    if len(texts) < 2:
        raise ValueError("CSV must contain at least two labeled rows")
    if len(set(labels)) < 2:
        raise ValueError("Training requires at least two distinct labels")
    return texts, labels


def _sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
        return TfidfVectorizer, LogisticRegression, LinearSVC, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
    except ImportError as exc:
        raise RuntimeError("ML classification requires scikit-learn. Install it to train or load models.") from exc


class TextClassifier:
    """TF-IDF classifier preserving the original NLPulse public API."""

    def __init__(self, algorithm: str = "logistic", training_data: Sequence[tuple[str, str]] | None = None) -> None:
        self.algorithm = algorithm.lower()
        if self.algorithm not in {"logistic", "logisticregression", "svm", "linearsvc"}:
            raise ValueError("algorithm must be 'logistic' or 'linearsvc'")
        self.training_data = list(training_data or DEFAULT_TRAINING_DATA)
        self.vectorizer = None
        self.model = None

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def train(self, texts: Sequence[str] | None = None, labels: Sequence[str] | None = None) -> "TextClassifier":
        TfidfVectorizer, LogisticRegression, LinearSVC, *_ = _sklearn()
        if texts is None or labels is None:
            texts, labels = zip(*self.training_data)
        texts, labels = list(texts), list(labels)
        if len(texts) != len(labels) or len(texts) < 2:
            raise ValueError("Training requires at least two text/label pairs")
        if len(set(labels)) < 2:
            raise ValueError("Training requires at least two distinct labels")
        if any(not str(text).strip() or not str(label).strip() for text, label in zip(texts, labels)):
            raise ValueError("Training text and labels must be non-empty")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        features = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=500, random_state=42) if self.algorithm.startswith("logistic") else LinearSVC(random_state=42)
        self.model.fit(features, labels)
        return self

    @classmethod
    def train_from_csv(cls, path: str | Path, algorithm: str = "logistic", text_column: str = "text", label_column: str = "label") -> "TextClassifier":
        texts, labels = load_labeled_csv(path, text_column, label_column)
        return cls(algorithm=algorithm).train(texts, labels)

    fit = train

    def predict(self, texts: str | Sequence[str]) -> list[str] | str:
        if not self.is_fitted:
            self.train()
        values = [texts] if isinstance(texts, str) else list(texts)
        predictions = self.model.predict(self.vectorizer.transform(values)).tolist()
        return predictions[0] if isinstance(texts, str) else predictions

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> dict[str, object]:
        _, _, _, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score = _sklearn()
        labels = list(labels)
        if len(texts) != len(labels) or not labels:
            raise ValueError("Evaluation requires equally sized, non-empty text and label sequences")
        predictions = self.predict(texts)
        class_labels = sorted(set(labels) | set(predictions))
        report = classification_report(labels, predictions, labels=class_labels, output_dict=True, zero_division=0)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_precision": float(precision_score(labels, predictions, labels=class_labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(labels, predictions, labels=class_labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, labels=class_labels, average="macro", zero_division=0)),
            "labels": class_labels,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
            "report": report,
        }

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise ValueError("Train the classifier before saving")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TextClassifier":
        try:
            with Path(path).open("rb") as handle:
                model = pickle.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Model artifact not found: {path}") from exc
        if not isinstance(model, cls):
            raise ValueError("File does not contain a TextClassifier")
        return model
