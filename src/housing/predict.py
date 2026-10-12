"""Command-line price prediction.

    python -m housing.predict examples/sample_house.json
    python -m housing.predict houses.csv --format csv

Input is a JSON object / list of objects, or a CSV with the 79 Ames column names.
Output is JSON on stdout. Invalid input exits with status 2 and a readable message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .inference import load_model, predict_batch


def _read_records(path: Path, fmt: str) -> list[dict]:
    if fmt == "csv" or (fmt == "auto" and path.suffix.lower() == ".csv"):
        frame = pd.read_csv(path)
        return [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in frame.to_dict("records")
        ]
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else [payload]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict Ames house prices with a 90% interval.")
    ap.add_argument("input", type=Path, help="JSON (object or list) or CSV file")
    ap.add_argument("--model", type=Path, default=None, help="path to a joblib model bundle")
    ap.add_argument("--format", choices=["auto", "json", "csv"], default="auto")
    args = ap.parse_args(argv)
    try:
        records = _read_records(args.input, args.format)
        preds = predict_batch(records, load_model(args.model))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps([p.model_dump() for p in preds], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
