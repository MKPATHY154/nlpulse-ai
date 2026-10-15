"""CPU-friendly learned text classifier built on scikit-learn."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

DEFAULT_TRAINING_DATA: tuple[tuple[str, str], ...] = (
    ("please help me fix this problem", "request"), ("I need assistance with my account", "request"),
    ("the application is broken and shows an error", "complaint"), ("this service is terrible", "complaint"),
    ("what time does support open", "question"), ("how can I reset my password", "question"),
    ("I love the new dashboard", "feedback"), ("please improve the search experience", "feedback"),
)


def _sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score, classification_report
        return TfidfVectorizer, LogisticRegression, LinearSVC, accuracy_score, classification_report
    except ImportError as exc:
        raise RuntimeError("ML classification requires scikit-learn. Install it to train or load models.") from exc


class TextClassifier:
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
        if len(texts) != len(labels) or len(texts) < 2:
            raise ValueError("Training requires at least two text/label pairs")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        features = self.vectorizer.fit_transform(texts)
        self.model = (LogisticRegression(max_iter=500, random_state=42) if self.algorithm.startswith("logistic") else LinearSVC(random_state=42))
        self.model.fit(features, labels)
        return self

    fit = train

    def predict(self, texts: str | Sequence[str]) -> list[str] | str:
        if not self.is_fitted: self.train()
        values = [texts] if isinstance(texts, str) else list(texts)
        predictions = self.model.predict(self.vectorizer.transform(values)).tolist()
        return predictions[0] if isinstance(texts, str) else predictions

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> dict[str, object]:
        _, _, _, accuracy_score, classification_report = _sklearn()
        predictions = self.predict(texts)
        result = {"accuracy": float(accuracy_score(labels, predictions)), "report": classification_report(labels, predictions, output_dict=True, zero_division=0)}
        return result

    def save(self, path: str | Path) -> None:
        if not self.is_fitted: raise ValueError("Train the classifier before saving")
        with Path(path).open("wb") as handle: pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TextClassifier":
        with Path(path).open("rb") as handle: model = pickle.load(handle)
        if not isinstance(model, cls): raise ValueError("File does not contain a TextClassifier")
        return model
