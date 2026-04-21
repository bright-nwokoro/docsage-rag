"""Evaluation harness scaffold.

Loads tests/eval/golden.yaml, runs each item against the live backend, and
scores answer presence + citation precision/recall. Currently reports zero
because the golden set is empty — populate golden.yaml to use.
"""
from pathlib import Path

import yaml


def main() -> None:
    path = Path(__file__).parent / "golden.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    items = data.get("items", [])
    print(f"loaded {len(items)} golden-set items from {path}")
    if not items:
        print("golden set is empty — add items to evaluate")
        return
    # TODO (future): run each question against POST /query, score citations.
    print("eval harness is a scaffold; full implementation pending roadmap task")


if __name__ == "__main__":
    main()
