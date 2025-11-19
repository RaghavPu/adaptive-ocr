import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize normalized edit distance by compression level and data_source."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "OmniDocBench",
        help="Path to OmniDocBench dataset directory.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Root directory containing per-compression evaluation results.",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=None,
        help="Optional override path to OmniDocBench.json annotations.",
    )
    parser.add_argument(
        "--results-filename",
        type=str,
        default="individual_results.json",
        help="Filename to look for inside each compression-level directory.",
    )
    return parser.parse_args()


def resolve_annotations_path(dataset_dir: Path, override: Path | None) -> Path:
    if override:
        return override
    return dataset_dir / "annotations" / "OmniDocBench.json"


def load_annotations(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected annotations JSON to be a list.")

    doc_to_source: Dict[str, str] = {}
    for entry in data:
        image_path = entry.get("page_info", {}).get("image_path")
        if not image_path:
            continue
        doc_id = Path(image_path).stem
        data_source = (
            entry.get("page_info", {})
            .get("page_attribute", {})
            .get("data_source")
        )
        if data_source:
            doc_to_source[doc_id] = data_source
    return doc_to_source


def find_results_files(root: Path, filename: str) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    results_map: Dict[str, Path] = {}

    root_file = root / filename
    if root_file.exists():
        label = root.name or "results"
        results_map[label] = root_file

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        candidate = subdir / filename
        if candidate.exists():
            results_map[subdir.name] = candidate

    if not results_map:
        raise FileNotFoundError(
            f"No '{filename}' files found beneath {root}. "
            "Ensure each compression level directory contains evaluation outputs."
        )

    return results_map


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of result entries in {path}")
    return data


def summarize(
    doc_to_source: Dict[str, str],
    level_to_results: Dict[str, Path],
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, int]]:
    summary: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(int)

    for level, result_path in level_to_results.items():
        entries = load_results(result_path)
        for entry in entries:
            doc_id = entry.get("document_id")
            if not doc_id:
                doc_path = entry.get("document_path")
                if doc_path:
                    doc_id = Path(doc_path).stem
            if not doc_id:
                stats["missing_doc_id"] += 1
                continue

            data_source = doc_to_source.get(doc_id)
            if not data_source:
                stats["missing_data_source"] += 1
                continue

            ned = entry.get("normalized_edit_distance")
            if isinstance(ned, (int, float)):
                summary[level][data_source].append(float(ned))
            else:
                stats["missing_ned"] += 1

    return summary, stats


def format_table(summary: Dict[str, Dict[str, List[float]]]) -> str:
    if not summary:
        return "No data available."

    data_sources = sorted(
        {source for level_data in summary.values() for source in level_data}
    )

    def sort_key(level: str):
        try:
            return (0, float(level))
        except ValueError:
            return (1, level)

    levels = sorted(summary.keys(), key=sort_key)

    header = ["Compression Level"] + data_sources
    rows = [header]

    for level in levels:
        row = [level]
        for source in data_sources:
            values = summary[level].get(source)
            if values:
                row.append(f"{mean(values):.3f}")
            else:
                row.append("-")
        rows.append(row)

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for row in rows:
        parts = [
            row[i].ljust(col_widths[i]) if i == 0 else row[i].rjust(col_widths[i])
            for i in range(len(row))
        ]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def main():
    args = parse_args()
    annotations_path = resolve_annotations_path(args.dataset_dir, args.annotations_file)
    doc_to_source = load_annotations(annotations_path)
    level_to_results = find_results_files(args.results_root, args.results_filename)
    summary, stats = summarize(doc_to_source, level_to_results)
    print(format_table(summary))
    if stats:
        print("\nSkipped entries summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

