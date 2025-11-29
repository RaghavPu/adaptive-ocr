#!/usr/bin/env python3
"""
Label each document with its optimal model size using elbow method.

Uses the geometric distance method: finds the point with maximum perpendicular 
distance from the line connecting the first and last points in the 
performance curve.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def find_elbow_by_distance(model_order: List[str], accuracies: Dict[str, float]) -> Tuple[str, int]:
    """
    Find the elbow point using perpendicular distance from first-to-last line.
    
    This is the classic geometric elbow method.
    
    Args:
        model_order: List of model names in size order
        accuracies: Dict mapping model name to accuracy
        
    Returns:
        Tuple of (optimal_model_name, elbow_index)
    """
    # Convert to arrays
    x = np.arange(len(model_order), dtype=float)
    y = np.array([accuracies[m] for m in model_order], dtype=float)
    
    # Handle edge cases
    if len(x) < 3:
        return model_order[0], 0
    
    # Check if all accuracies are the same (flat line)
    if np.allclose(y, y[0]):
        # No elbow, just return smallest model
        return model_order[0], 0
    
    # Check if slope is negative (performance decreases with larger models)
    if y[-1] <= y[0]:
        # If final model is worse than or equal to first model, use tiny
        return model_order[0], 0
    
    # First and last point define the line
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Calculate perpendicular distance from each point to the line p1-p2
    max_distance = 0
    elbow_idx = 0
    
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    
    if line_length == 0:
        # First and last points are the same, no elbow
        return model_order[0], 0
    
    for i in range(1, len(x) - 1):  # Don't check first and last
        point = np.array([x[i], y[i]])
        
        # Vector from p1 to point
        point_vec = point - p1
        
        # Perpendicular distance formula
        distance = np.abs(np.cross(line_vec, point_vec)) / line_length
        
        if distance > max_distance:
            max_distance = distance
            elbow_idx = i
    
    return model_order[elbow_idx], elbow_idx


def find_elbow_by_derivative(model_order: List[str], accuracies: Dict[str, float]) -> Tuple[str, int]:
    """
    Find elbow by detecting where the rate of improvement drops significantly.
    
    Args:
        model_order: List of model names in size order
        accuracies: Dict mapping model name to accuracy
        
    Returns:
        Tuple of (optimal_model_name, elbow_index)
    """
    # Calculate improvements between consecutive models
    improvements = []
    for i in range(len(model_order) - 1):
        curr_acc = accuracies[model_order[i]]
        next_acc = accuracies[model_order[i+1]]
        improvement = next_acc - curr_acc
        improvements.append(improvement)
    
    if not improvements:
        return model_order[0], 0
    
    # Check if all improvements are similar (no clear elbow)
    if all(abs(imp - improvements[0]) < 0.001 for imp in improvements):
        return model_order[0], 0
    
    # Find where the rate of improvement drops significantly
    # Look for the first point where improvement is less than half the previous
    for i in range(1, len(improvements)):
        if improvements[i] < improvements[i-1] * 0.5:
            # Return the model at index i (before the drop)
            return model_order[i], i
    
    # If no significant drop, look for maximum improvement
    max_improvement_idx = improvements.index(max(improvements))
    return model_order[max_improvement_idx + 1], max_improvement_idx + 1


def load_all_results(results_root: Path, model_order: List[str]) -> Dict[str, Dict[str, Dict]]:
    """
    Load individual results from all model directories.
    
    Args:
        results_root: Root directory containing model subdirectories
        model_order: List of model names to load
        
    Returns:
        Dict mapping model_name -> document_id -> result_dict
    """
    results_by_model = {}
    
    for model in model_order:
        model_dir = results_root / model
        results_file = model_dir / "individual_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results_list = json.load(f)
        
        # Convert list to dict keyed by document_id
        results_dict = {r['document_id']: r for r in results_list}
        results_by_model[model] = results_dict
        print(f"Loaded {len(results_dict)} documents from {model}")
    
    return results_by_model


def label_documents(results_by_model: Dict[str, Dict[str, Dict]], 
                    model_order: List[str],
                    method: str = 'distance') -> Dict[str, Dict]:
    """
    Label each document with its optimal model using elbow method.
    
    Args:
        results_by_model: Dict mapping model -> doc_id -> results
        model_order: List of model names in size order
        method: 'distance' or 'derivative'
        
    Returns:
        Dict mapping document_id to labeling info
    """
    # Get all document IDs (from first model)
    doc_ids = list(results_by_model[model_order[0]].keys())
    
    labeled_docs = {}
    
    for doc_id in doc_ids:
        # Get accuracies for this document across all models
        accuracies = {}
        for model in model_order:
            if doc_id in results_by_model[model]:
                accuracies[model] = results_by_model[model][doc_id]['character_accuracy']
            else:
                # Document missing in this model's results
                print(f"Warning: {doc_id} missing from {model} results")
                continue
        
        # Skip if we don't have results for all models
        if len(accuracies) != len(model_order):
            print(f"Skipping {doc_id}: incomplete results")
            continue
        
        # Find elbow
        if method == 'distance':
            optimal_model, elbow_idx = find_elbow_by_distance(model_order, accuracies)
        elif method == 'derivative':
            optimal_model, elbow_idx = find_elbow_by_derivative(model_order, accuracies)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        labeled_docs[doc_id] = {
            'optimal_model': optimal_model,
            'elbow_index': elbow_idx,
            'accuracies': accuracies,
            'best_accuracy': max(accuracies.values()),
            'worst_accuracy': min(accuracies.values()),
            'improvement_range': max(accuracies.values()) - min(accuracies.values())
        }
    
    return labeled_docs


def print_statistics(labeled_docs: Dict[str, Dict], model_order: List[str]):
    """Print summary statistics about the labeling."""
    print("\n" + "="*60)
    print("LABELING STATISTICS")
    print("="*60)
    
    # Count optimal model distribution
    optimal_counts = Counter(doc['optimal_model'] for doc in labeled_docs.values())
    
    print(f"\nTotal documents labeled: {len(labeled_docs)}")
    print("\nOptimal model distribution:")
    for model in model_order:
        count = optimal_counts[model]
        percentage = (count / len(labeled_docs)) * 100
        print(f"  {model:8s}: {count:4d} documents ({percentage:5.1f}%)")
    
    # Average improvement range by optimal model
    print("\nAverage accuracy improvement range by optimal model:")
    for model in model_order:
        docs_with_model = [d for d in labeled_docs.values() if d['optimal_model'] == model]
        if docs_with_model:
            avg_range = np.mean([d['improvement_range'] for d in docs_with_model])
            avg_best = np.mean([d['best_accuracy'] for d in docs_with_model])
            print(f"  {model:8s}: improvement_range={avg_range:.4f}, best_acc={avg_best:.4f}")


def save_results(labeled_docs: Dict[str, Dict], output_file: Path):
    """Save labeled results to JSON file."""
    # Convert to list for easier viewing
    output_list = [
        {
            'document_id': doc_id,
            **info
        }
        for doc_id, info in labeled_docs.items()
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Label documents with optimal model size using elbow method"
    )
    parser.add_argument(
        '--results-root',
        type=Path,
        default=Path(__file__).parent / 'results',
        help='Root directory containing model result folders'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'optimal_model_labels.json',
        help='Output JSON file for labeled results'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['distance', 'derivative'],
        default='distance',
        help='Elbow detection method (default: distance)'
    )
    parser.add_argument(
        '--model-order',
        type=str,
        nargs='+',
        default=['tiny', 'small', 'base', 'large', 'gundam'],
        help='Model names in size order'
    )
    
    args = parser.parse_args()
    
    print("Loading results from all models...")
    results_by_model = load_all_results(args.results_root, args.model_order)
    
    print(f"\nApplying {args.method} elbow method...")
    labeled_docs = label_documents(results_by_model, args.model_order, args.method)
    
    print_statistics(labeled_docs, args.model_order)
    
    save_results(labeled_docs, args.output)


if __name__ == '__main__':
    main()

