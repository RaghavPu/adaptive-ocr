#!/usr/bin/env python3
"""Main evaluation script for DeepSeek-OCR on OmniDocBench."""

import argparse
import sys
from pathlib import Path

from pipeline.document_processor import DocumentProcessor


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DeepSeek-OCR on OmniDocBench documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single document
  python evaluate.py --doc_path /path/to/document.pdf --gt_path /path/to/ground_truth.json
  
  # Process entire OmniDocBench dataset
  python evaluate.py --dataset_path /path/to/omni_doc_bench --output_dir results
  
  # Use custom prompt
  python evaluate.py --doc_path doc.pdf --gt_path gt.json --prompt "<image>\\n<|grounding|>Convert the document to markdown."
        """
    )
    
    parser.add_argument(
        '--doc_path',
        type=str,
        help='Path to single document file (PDF or image)'
    )
    
    parser.add_argument(
        '--gt_path',
        type=str,
        help='Path to ground truth file (JSON or text)'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to OmniDocBench dataset directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='deepseek-ai/DeepSeek-OCR',
        help='DeepSeek-OCR model name or path (default: deepseek-ai/DeepSeek-OCR)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='<image>\nFree OCR.',
        help='OCR prompt to use (default: "<image>\\nFree OCR.")'
    )
    
    parser.add_argument(
        '--no_save_individual',
        action='store_true',
        help='Do not save individual document results (only aggregated metrics)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.doc_path and not args.dataset_path:
        parser.error("Either --doc_path or --dataset_path must be provided")
    
    if args.doc_path and not Path(args.doc_path).exists():
        parser.error(f"Document not found: {args.doc_path}")
    
    if args.gt_path and not Path(args.gt_path).exists():
        parser.error(f"Ground truth file not found: {args.gt_path}")
    
    if args.dataset_path and not Path(args.dataset_path).exists():
        parser.error(f"Dataset directory not found: {args.dataset_path}")
    
    # Initialize processor
    try:
        processor = DocumentProcessor(model_name=args.model_name, device=args.device)
    except Exception as e:
        print(f"Error initializing processor: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process documents
    try:
        if args.dataset_path:
            # Process entire dataset
            print(f"Processing dataset: {args.dataset_path}")
            metrics = processor.process_dataset(
                args.dataset_path,
                args.output_dir,
                prompt=args.prompt,
                save_individual_results=not args.no_save_individual
            )
            
            print(f"\nResults saved to: {args.output_dir}")
            
        elif args.doc_path:
            # Process single document
            print(f"Processing document: {args.doc_path}")
            metrics = processor.process_single_document(
                args.doc_path,
                args.gt_path,
                prompt=args.prompt
            )
            
            # Print results
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Document: {metrics.get('document_path', 'N/A')}")
            
            if metrics.get('ground_truth') is not None:
                print("\n--- Metrics ---")
                print(f"Character Accuracy: {metrics.get('character_accuracy', 'N/A'):.4f}" 
                      if isinstance(metrics.get('character_accuracy'), float) 
                      else f"Character Accuracy: {metrics.get('character_accuracy', 'N/A')}")
                print(f"Word Accuracy: {metrics.get('word_accuracy', 'N/A'):.4f}"
                      if isinstance(metrics.get('word_accuracy'), float)
                      else f"Word Accuracy: {metrics.get('word_accuracy', 'N/A')}")
                print(f"Edit Distance: {metrics.get('edit_distance', 'N/A')}")
                print(f"Normalized Edit Distance: {metrics.get('normalized_edit_distance', 'N/A'):.4f}"
                      if isinstance(metrics.get('normalized_edit_distance'), float)
                      else f"Normalized Edit Distance: {metrics.get('normalized_edit_distance', 'N/A')}")
            else:
                print("\nNo ground truth provided - only OCR output available")
            
            print("\n--- OCR Output (first 500 chars) ---")
            ocr_output = metrics.get('ocr_output', '')
            print(ocr_output[:500] + ('...' if len(ocr_output) > 500 else ''))
            print("="*60)
            
            # Save result if output directory specified
            if args.output_dir:
                import json
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{Path(args.doc_path).stem}_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"\nResult saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

