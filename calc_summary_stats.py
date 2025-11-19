import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate OmniDocBench summary statistics."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "OmniDocBench",
        help="Path to the OmniDocBench dataset directory (default: ./OmniDocBench).",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=None,
        help="Optional direct path to OmniDocBench annotations JSON file.",
    )
    return parser.parse_args()


def resolve_annotations_path(dataset_dir: Path, annotations_path: Path | None) -> Path:
    if annotations_path:
        return annotations_path
    return dataset_dir / "annotations" / "OmniDocBench.json"


def load_annotations(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected annotations JSON to contain a list of documents.")
    return data


def compute_counts(annotations: list) -> Tuple[int, Counter, int]:
    counts = Counter()
    missing = 0
    for entry in annotations:
        data_source = (
            entry.get("page_info", {})
            .get("page_attribute", {})
            .get("data_source")
        )
        if data_source is None:
            missing += 1
        else:
            counts[data_source] += 1
    return len(annotations), counts, missing


def print_summary(total: int, counts: Dict[str, int], missing: int) -> None:
    print(f"Total documents: {total}")
    print("Documents per data_source:")
    for source, count in counts.most_common():
        percentage = (count / total) * 100 if total else 0
        print(f"  {source}: {count} ({percentage:.2f}%)")
    if missing:
        print(f"Documents missing data_source: {missing}")


def main():
    args = parse_args()
    annotations_path = resolve_annotations_path(args.dataset_dir, args.annotations_path)
    annotations = load_annotations(annotations_path)
    total, counts, missing = compute_counts(annotations)
    print_summary(total, counts, missing)


if __name__ == "__main__":
    main()

