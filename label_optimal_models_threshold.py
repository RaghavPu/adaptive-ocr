#!/usr/bin/env python3
"""
Label documents with optimal model using simple threshold approach.

Strategy: Pick the SMALLEST model that achieves within X% of the best accuracy.
This is more practical than geometric elbow method.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def find_optimal_by_threshold(model_order: List[str], 
                               accuracies: Dict[str, float],
                               threshold_pct: float = 0.04) -> Tuple[str, int, Dict]:
    """
    Find optimal model using threshold approach.
    
    Pick the SMALLEST model that achieves within threshold_pct of best accuracy.
    
    Args:
        model_order: List of model names in size order
        accuracies: Dict mapping model name to accuracy
        threshold_pct: Percentage threshold (e.g., 0.04 = 4%)
        
    Returns:
        Tuple of (optimal_model_name, index, details)
    """
    # Get accuracies in order
    accs = [accuracies[m] for m in model_order]
    
    # Find best accuracy
    best_acc = max(accs)
    
    # Calculate threshold
    threshold = best_acc - threshold_pct
    
    # Find smallest model meeting threshold
    for i, (model, acc) in enumerate(zip(model_order, accs)):
        if acc >= threshold:
            return model, i, {
                'best_accuracy': best_acc,
                'threshold': threshold,
                'chosen_accuracy': acc,
                'margin_from_best': best_acc - acc
            }
    
    # Fallback to first model (shouldn't happen)
    return model_order[0], 0, {
        'best_accuracy': best_acc,
        'threshold': threshold,
        'chosen_accuracy': accs[0],
        'margin_from_best': best_acc - accs[0]
    }


def load_all_results(results_root: Path, model_order: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Load individual results from all model directories."""
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
                    threshold_pct: float = 0.04) -> Dict[str, Dict]:
    """Label each document with optimal model using threshold method."""
    
    # Get all document IDs
    doc_ids = list(results_by_model[model_order[0]].keys())
    
    labeled_docs = {}
    
    for doc_id in doc_ids:
        # Get accuracies for this document across all models
        accuracies = {}
        for model in model_order:
            if doc_id in results_by_model[model]:
                accuracies[model] = results_by_model[model][doc_id]['character_accuracy']
            else:
                print(f"Warning: {doc_id} missing from {model} results")
                continue
        
        # Skip if we don't have results for all models
        if len(accuracies) != len(model_order):
            print(f"Skipping {doc_id}: incomplete results")
            continue
        
        # Find optimal model using threshold
        optimal_model, opt_idx, details = find_optimal_by_threshold(
            model_order, accuracies, threshold_pct
        )
        
        labeled_docs[doc_id] = {
            'optimal_model': optimal_model,
            'optimal_index': opt_idx,
            'accuracies': accuracies,
            'best_accuracy': details['best_accuracy'],
            'chosen_accuracy': details['chosen_accuracy'],
            'margin_from_best': details['margin_from_best'],
            'threshold_used': threshold_pct,
            'worst_accuracy': min(accuracies.values()),
            'improvement_range': max(accuracies.values()) - min(accuracies.values())
        }
    
    return labeled_docs


def print_statistics(labeled_docs: Dict[str, Dict], model_order: List[str], threshold_pct: float):
    """Print summary statistics about the labeling."""
    print("\n" + "="*60)
    print(f"LABELING STATISTICS (Threshold: {threshold_pct*100:.0f}%)")
    print("="*60)
    
    # Count optimal model distribution
    optimal_counts = Counter(doc['optimal_model'] for doc in labeled_docs.values())
    
    print(f"\nTotal documents labeled: {len(labeled_docs)}")
    print(f"\nOptimal model distribution:")
    for model in model_order:
        count = optimal_counts[model]
        percentage = (count / len(labeled_docs)) * 100
        print(f"  {model:8s}: {count:4d} documents ({percentage:5.1f}%)")
    
    # Statistics by optimal model
    print(f"\nStatistics by optimal model:")
    for model in model_order:
        docs_with_model = [d for d in labeled_docs.values() if d['optimal_model'] == model]
        if docs_with_model:
            avg_margin = np.mean([d['margin_from_best'] for d in docs_with_model])
            avg_best = np.mean([d['best_accuracy'] for d in docs_with_model])
            avg_chosen = np.mean([d['chosen_accuracy'] for d in docs_with_model])
            
            print(f"\n  {model.upper()}:")
            print(f"    Avg best accuracy:    {avg_best:.4f}")
            print(f"    Avg chosen accuracy:  {avg_chosen:.4f}")
            print(f"    Avg margin from best: {avg_margin:.4f} ({avg_margin*100:.1f}%)")
    
    # How many are at exactly the best?
    at_best = sum(1 for d in labeled_docs.values() if d['margin_from_best'] < 0.001)
    print(f"\nDocuments at best model: {at_best} ({100*at_best/len(labeled_docs):.1f}%)")
    
    # How many saved compute?
    saved_compute = {}
    for model_idx, model in enumerate(model_order):
        count = sum(1 for d in labeled_docs.values() 
                   if d['optimal_index'] <= model_idx)
        saved_compute[model] = count
    
    print(f"\nCumulative compute savings:")
    for model in model_order:
        count = saved_compute[model]
        pct = 100 * count / len(labeled_docs)
        print(f"  Up to {model:8s}: {count:4d} docs ({pct:5.1f}%)")


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


def compare_with_old_labels(new_labels: Dict, old_labels_file: Path):
    """Compare new labels with old geometric elbow labels."""
    if not old_labels_file.exists():
        return
    
    with open(old_labels_file) as f:
        old_data = json.load(f)
    old_labels = {d['document_id']: d['optimal_model'] for d in old_data}
    
    print("\n" + "="*60)
    print("COMPARISON WITH GEOMETRIC ELBOW METHOD")
    print("="*60)
    
    changed = 0
    moved_smaller = 0
    moved_larger = 0
    
    model_order = ['tiny', 'small', 'base', 'large', 'gundam']
    
    for doc_id, new_data in new_labels.items():
        new_model = new_data['optimal_model']
        old_model = old_labels.get(doc_id)
        
        if old_model and old_model != new_model:
            changed += 1
            old_idx = model_order.index(old_model)
            new_idx = model_order.index(new_model)
            
            if new_idx < old_idx:
                moved_smaller += 1
            else:
                moved_larger += 1
    
    print(f"\nTotal documents: {len(new_labels)}")
    print(f"Changed labels: {changed} ({100*changed/len(new_labels):.1f}%)")
    print(f"  Moved to smaller model: {moved_smaller}")
    print(f"  Moved to larger model: {moved_larger}")
    print(f"Unchanged: {len(new_labels) - changed} ({100*(len(new_labels)-changed)/len(new_labels):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Label documents with optimal model using threshold method"
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
        default=Path(__file__).parent / 'optimal_model_labels_threshold.json',
        help='Output JSON file for labeled results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.04,
        help='Accuracy threshold as decimal (default: 0.04 = 4%%)'
    )
    parser.add_argument(
        '--model-order',
        type=str,
        nargs='+',
        default=['tiny', 'small', 'base', 'large', 'gundam'],
        help='Model names in size order'
    )
    parser.add_argument(
        '--compare-with',
        type=Path,
        default=Path(__file__).parent / 'optimal_model_labels.json',
        help='Compare with existing labels file'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from all models...")
    results_by_model = load_all_results(args.results_root, args.model_order)
    
    print(f"\nApplying threshold method (threshold: {args.threshold*100:.0f}%)...")
    print(f"Strategy: Pick SMALLEST model within {args.threshold*100:.0f}% of best")
    labeled_docs = label_documents(results_by_model, args.model_order, args.threshold)
    
    print_statistics(labeled_docs, args.model_order, args.threshold)
    
    if args.compare_with:
        compare_with_old_labels(labeled_docs, args.compare_with)
    
    save_results(labeled_docs, args.output)


if __name__ == '__main__':
    main()

