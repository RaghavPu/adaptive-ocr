#!/usr/bin/env python3
"""
Plot Pareto frontier comparing adaptive-OCR against single-resolution approaches.

Shows the trade-off between tokens (cost) and normalized edit distance (quality).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_evaluation_data(eval_results_path):
    """Load evaluation results from JSON file."""
    with open(eval_results_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_avg_tokens_per_image(total_tokens, num_images):
    """Calculate average tokens per image."""
    return total_tokens / num_images if num_images > 0 else 0


def create_pareto_plot(eval_results_path, output_path='pareto_plot.png', exclude_gundam=True):
    """
    Create Pareto frontier plot showing adaptive-OCR vs single-resolution methods.
    
    Args:
        eval_results_path: Path to evaluation_results.json
        output_path: Path to save the plot
        exclude_gundam: Whether to exclude gundam from single-resolution methods
    """
    # Load data
    data = load_evaluation_data(eval_results_path)
    
    token_stats = data['token_savings']
    accuracy_stats = data['accuracy']
    
    num_images = accuracy_stats['total_evaluated']
    
    # Extract data for each method
    methods = []
    
    # Single-resolution methods
    model_order = ['tiny', 'small', 'base', 'large', 'gundam']
    for model in model_order:
        if exclude_gundam and model == 'gundam':
            continue
            
        total_tokens = token_stats['per_model_tokens'][model]
        avg_tokens = calculate_avg_tokens_per_image(total_tokens, num_images)
        ned = accuracy_stats['per_model_accuracies'][model]
        
        methods.append({
            'name': model.capitalize(),
            'tokens': avg_tokens,
            'ned': ned,
            'type': 'single-resolution'
        })
    
    # Adaptive-OCR method
    adaptive_tokens = token_stats['avg_tokens_per_image']
    adaptive_ned = accuracy_stats['avg_predicted_accuracy']
    
    methods.append({
        'name': 'Adaptive-OCR',
        'tokens': adaptive_tokens,
        'ned': adaptive_ned,
        'type': 'adaptive'
    })
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Separate single-resolution and adaptive methods
    single_res = [m for m in methods if m['type'] == 'single-resolution']
    adaptive = [m for m in methods if m['type'] == 'adaptive']
    
    # Plot single-resolution methods
    for method in single_res:
        ax.scatter(method['tokens'], method['ned'], 
                  s=200, alpha=0.7, edgecolors='black', linewidth=2,
                  label=method['name'], zorder=3)
        ax.annotate(method['name'], 
                   (method['tokens'], method['ned']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Plot adaptive method with different style
    for method in adaptive:
        ax.scatter(method['tokens'], method['ned'],
                  s=300, alpha=0.9, edgecolors='red', linewidth=3,
                  marker='*', color='gold', zorder=4,
                  label=method['name'])
        ax.annotate(method['name'],
                   (method['tokens'], method['ned']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold', color='red')
    
    # Draw Pareto frontier (convex hull of single-resolution methods)
    if len(single_res) > 1:
        # Sort by tokens
        sorted_single = sorted(single_res, key=lambda x: x['tokens'])
        
        # Find Pareto optimal points (lower tokens and lower NED is better)
        pareto_points = []
        for i, point in enumerate(sorted_single):
            is_pareto = True
            for j, other in enumerate(sorted_single):
                if i != j:
                    # Check if other point dominates this one
                    if (other['tokens'] <= point['tokens'] and 
                        other['ned'] <= point['ned'] and
                        (other['tokens'] < point['tokens'] or other['ned'] < point['ned'])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_points.append(point)
        
        # Sort Pareto points by tokens for plotting
        pareto_points = sorted(pareto_points, key=lambda x: x['tokens'])
        
        if len(pareto_points) > 1:
            pareto_tokens = [p['tokens'] for p in pareto_points]
            pareto_ned = [p['ned'] for p in pareto_points]
            ax.plot(pareto_tokens, pareto_ned, 
                   '--', color='gray', alpha=0.5, linewidth=2,
                   label='Pareto Frontier (single-resolution)', zorder=1)
    
    # Highlight adaptive-OCR's advantage
    if adaptive:
        adaptive_point = adaptive[0]
        # Draw arrow or highlight to show it's better
        ax.axhline(y=adaptive_point['ned'], color='green', 
                  linestyle=':', alpha=0.3, linewidth=1, zorder=0)
        ax.axvline(x=adaptive_point['tokens'], color='green',
                  linestyle=':', alpha=0.3, linewidth=1, zorder=0)
    
    # Labels and formatting
    ax.set_xlabel('Average Vision Tokens per Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Edit Distance (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Adaptive-OCR vs Single-Resolution Methods\n(Pareto Optimality Analysis)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Invert y-axis since lower NED is better
    ax.invert_yaxis()
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add text annotation explaining Pareto optimality
    if adaptive:
        adaptive_point = adaptive[0]
        # Find closest single-resolution method
        closest = min(single_res, 
                     key=lambda x: ((x['tokens'] - adaptive_point['tokens'])**2 + 
                                    (x['ned'] - adaptive_point['ned'])**2)**0.5)
        
        # Calculate improvements
        token_improvement = ((closest['tokens'] - adaptive_point['tokens']) / closest['tokens']) * 100
        ned_improvement = ((adaptive_point['ned'] - closest['ned']) / closest['ned']) * 100
        
        textstr = f'Adaptive-OCR achieves:\n'
        textstr += f'• {token_improvement:.1f}% fewer tokens than {closest["name"]}\n'
        if ned_improvement < 0:
            textstr += f'• {abs(ned_improvement):.1f}% higher NED (trade-off)'
        else:
            textstr += f'• {ned_improvement:.1f}% lower NED (better)'
        
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Plot Pareto frontier comparing adaptive-OCR vs single-resolution methods'
    )
    parser.add_argument('--eval_results', type=str,
                       default='simple_cnn_output/evaluation_results.json',
                       help='Path to evaluation_results.json')
    parser.add_argument('--output', type=str,
                       default='figs/pareto_plot.png',
                       help='Output path for the plot')
    parser.add_argument('--include_gundam', action='store_true',
                       help='Include gundam in single-resolution methods')
    
    args = parser.parse_args()
    
    eval_path = Path(args.eval_results)
    if not eval_path.exists():
        print(f"Error: Evaluation results file not found: {eval_path}")
        return
    
    create_pareto_plot(
        eval_path,
        output_path=args.output,
        exclude_gundam=not args.include_gundam
    )


if __name__ == '__main__':
    main()

