#!/usr/bin/env python3
"""Batch evaluation script for all DeepSeek-OCR model variants."""

import subprocess
import sys
from pathlib import Path

# Model configurations based on DeepSeek-OCR native resolutions
MODEL_CONFIGS = {
    'tiny': {
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'image_size': 512,
        'base_size': 512,
        'crop_mode': False,
    },
    'small': {
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'image_size': 640,
        'base_size': 640,
        'crop_mode': False,
    },
    'base': {
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'image_size': 1024,
        'base_size': 1024,
        'crop_mode': False,
    },
    'large': {
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'image_size': 1280,
        'base_size': 1280,
        'crop_mode': False,
    },
    'gundam': {
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'image_size': 1024,  # Max size for gundam mode (mixes 640x640 and 1024x1024)
        'base_size': 1024,
        'crop_mode': True,  # Enable dynamic tiling
    },
    'gundam-M': {
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'image_size': 1280,  # Max size for gundam-M (adaptive 512px to 1280px)
        'base_size': 1280,
        'crop_mode': True,  # Enable dynamic tiling with adaptive resolution
    },
}

def run_evaluation(model_name, output_dir, dataset_path, image_size, base_size, max_docs=None, crop_mode=False):
    """Run evaluation for a specific model configuration."""
    # Get the path to evaluate.py in the same directory
    evaluate_script = Path(__file__).parent / 'evaluate.py'
    
    cmd = [
        sys.executable, str(evaluate_script),
        '--dataset_path', dataset_path,
        '--model_name', model_name,
        '--output_dir', output_dir,
        '--image_size', str(image_size),
        '--base_size', str(base_size),
    ]
    
    if crop_mode:
        cmd.append('--crop_mode')
    
    if max_docs is not None:
        cmd.extend(['--max_docs', str(max_docs)])
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {output_dir}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Crop mode: {crop_mode}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, check=False, cwd=Path(__file__).parent.parent)
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch evaluate all model variants')
    parser.add_argument('--dataset_path', type=str, default='OmniDocBench',
                       help='Path to OmniDocBench dataset')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS.keys()) + ['all'],
                       default=['all'], help='Models to evaluate')
    parser.add_argument('--max_docs', type=int,
                       help='Optional limit on number of documents to process per model')
    # Remove the global --crop_mode flag since it's now model-specific
    # parser.add_argument('--crop_mode', action='store_true',
    #                    help='Enable crop mode when preprocessing images')
    
    args = parser.parse_args()
    
    models_to_eval = MODEL_CONFIGS.keys() if 'all' in args.models else args.models
    
    results = {}
    for model_key in models_to_eval:
        if model_key not in MODEL_CONFIGS:
            print(f"Warning: Unknown model {model_key}, skipping")
            continue
            
        config = MODEL_CONFIGS[model_key]
        output_dir = f"results/{model_key}"
        
        success = run_evaluation(
            config['model_name'],
            output_dir,
            args.dataset_path,
            config['image_size'],
            config['base_size'],
            max_docs=args.max_docs,
            crop_mode=config.get('crop_mode', False)  # Use model-specific crop_mode
        )
        
        results[model_key] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for model, status in results.items():
        print(f"{model:15s}: {status}")
    print("="*60)

if __name__ == '__main__':
    main()