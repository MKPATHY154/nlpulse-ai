"""Train and persist an NLPulse model from a labeled CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python scripts/train_model.py ...`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml_classifier import TextClassifier, load_labeled_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an NLPulse text classifier from labeled CSV data")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/text_classifier.pkl"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--algorithm", choices=("logistic", "linearsvc"), default="logistic")
    args = parser.parse_args()
    texts, labels = load_labeled_csv(args.csv_file, args.text_column, args.label_column)
    model = TextClassifier(algorithm=args.algorithm).train(texts, labels)
    model.save(args.output)
    print(json.dumps({"status": "trained", "algorithm": model.algorithm, "rows": len(texts), "classes": sorted(model.model.classes_), "artifact": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
