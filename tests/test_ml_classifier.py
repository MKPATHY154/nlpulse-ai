"""Behavior tests for labeled CSV training and model artifacts."""

import csv

import pytest

from evaluation.evaluate import evaluate_file
from src.ml_classifier import TextClassifier, load_labeled_csv


def test_csv_training_evaluates_all_requested_metrics_and_round_trips(tmp_path):
    csv_path = tmp_path / "training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for label, examples in {"a": ["alpha one", "alpha two", "alpha three", "alpha four"], "b": ["beta one", "beta two", "beta three", "beta four"]}.items():
            for text in examples:
                writer.writerow([text, label])
    result = evaluate_file(csv_path, algorithm="linearsvc")
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "confusion_matrix", "report"} <= result.keys()
    model = TextClassifier.train_from_csv(csv_path, algorithm="logistic")
    artifact = tmp_path / "models" / "model.pkl"
    model.save(artifact)
    assert TextClassifier.load(artifact).predict("alpha example") == "a"


def test_invalid_csv_has_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("message,category\nhello,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text.*label"):
        load_labeled_csv(path)
