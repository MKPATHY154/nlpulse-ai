"""Evaluate the learned NLPulse classifier from Python or the command line."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.ml_classifier import TextClassifier


def evaluate_file(path: str | Path, text_column: str = "text", label_column: str = "label", algorithm: str = "logistic") -> dict[str, object]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or text_column not in rows[0] or label_column not in rows[0]:
        raise ValueError(f"CSV must contain '{text_column}' and '{label_column}' columns")
    classifier = TextClassifier(algorithm=algorithm).train()
    return classifier.evaluate([row[text_column] for row in rows], [row[label_column] for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an NLPulse text classifier")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--algorithm", choices=("logistic", "linearsvc"), default="logistic")
    args = parser.parse_args()
    print(json.dumps(evaluate_file(args.csv_file, args.text_column, args.label_column, args.algorithm), indent=2))


if __name__ == "__main__":
    main()
