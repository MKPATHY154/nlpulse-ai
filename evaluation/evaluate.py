"""Train and evaluate the learned NLPulse classifier from Python or CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml_classifier import TextClassifier, load_labeled_csv


def _split(texts: list[str], labels: list[str], test_size: float = 0.25, random_state: int = 42) -> tuple[list[str], list[str], list[str], list[str]]:
    from sklearn.model_selection import train_test_split

    counts = {label: labels.count(label) for label in set(labels)}
    test_count = max(1, round(len(labels) * test_size))
    can_stratify = min(counts.values()) >= 2 and test_count >= len(counts) and len(labels) - test_count >= len(counts)
    try:
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels if can_stratify else None)
    except ValueError:
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state)


def evaluate_file(path: str | Path, text_column: str = "text", label_column: str = "label", algorithm: str = "logistic", test_size: float = 0.25) -> dict[str, object]:
    texts, labels = load_labeled_csv(path, text_column, label_column)
    train_texts, test_texts, train_labels, test_labels = _split(texts, labels, test_size)
    classifier = TextClassifier(algorithm=algorithm).train(train_texts, train_labels)
    result = classifier.evaluate(test_texts, test_labels)
    result.update({"algorithm": classifier.algorithm, "train_size": len(train_texts), "test_size": len(test_texts), "csv_file": str(path)})
    return result


def compare_algorithms(path: str | Path, text_column: str = "text", label_column: str = "label", test_size: float = 0.25) -> dict[str, dict[str, object]]:
    """Evaluate Logistic Regression and LinearSVC on the same holdout split."""
    texts, labels = load_labeled_csv(path, text_column, label_column)
    train_texts, test_texts, train_labels, test_labels = _split(texts, labels, test_size)
    results = {}
    for algorithm in ("logistic", "linearsvc"):
        classifier = TextClassifier(algorithm=algorithm).train(train_texts, train_labels)
        result = classifier.evaluate(test_texts, test_labels)
        result.update({"algorithm": classifier.algorithm, "train_size": len(train_texts), "test_size": len(test_texts), "csv_file": str(path)})
        results[algorithm] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an NLPulse text classifier")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--algorithm", choices=("logistic", "linearsvc", "both"), default="logistic")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    result = compare_algorithms(args.csv_file, args.text_column, args.label_column, args.test_size) if args.algorithm == "both" else evaluate_file(args.csv_file, args.text_column, args.label_column, args.algorithm, args.test_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
