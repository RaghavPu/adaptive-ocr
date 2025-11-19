"""Main document processor orchestrator."""

import json
import random
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional

from .ocr_inference import DeepSeekOCRInference
from .metrics import OCRMetrics
from .omni_doc_bench_loader import OmniDocBenchLoader


class DocumentProcessor:
    """Main orchestrator for processing documents and calculating metrics."""
    
    def __init__(
        self, 
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda'
    ):
        """Initialize document processor.
        
        Args:
            model_name: DeepSeek-OCR model name or path
            device: Device to run inference on
        """
        self.ocr_model = DeepSeekOCRInference(model_name, device)
        self.metrics = OCRMetrics()
        self.loader = OmniDocBenchLoader()
    
    def process_single_document(
        self, 
        doc_path: str, 
        gt_path: Optional[str] = None,
        gt_text: Optional[str] = None,  # Add this parameter
        prompt: str = "<image>\nFree OCR.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> Dict:
        """Process a single document and return metrics.
        
        Args:
            doc_path: Path to document file
            gt_path: Optional path to ground truth file
            gt_text: Optional ground truth text (takes precedence over gt_path)
            prompt: OCR prompt to use
            base_size: Base image size for preprocessing
            image_size: Target image size for preprocessing
            crop_mode: Whether to use crop mode
            
        Returns:
            Dictionary containing OCR output and metrics
        """
        print(f"Processing document: {doc_path}")
        
        # Get OCR output
        try:
            ocr_output = self.ocr_model.process_document(
                doc_path, 
                prompt=prompt,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode
            )
        except Exception as e:
            print(f"Error during OCR inference: {e}")
            ocr_output = ""
        
        result = {
            'document_path': str(doc_path),
            'ocr_output': ocr_output,
        }
        
        # Get ground truth - prefer gt_text over gt_path
        ground_truth = None
        if gt_text is not None:
            ground_truth = gt_text
        elif gt_path:
            try:
                ground_truth = self.loader.load_ground_truth(gt_path)
            except Exception as e:
                print(f"Error loading ground truth from file: {e}")
                ground_truth = None
        
        # Calculate metrics if we have ground truth
        if ground_truth is not None:
            try:
                metrics = self.metrics.calculate_all_metrics(ocr_output, ground_truth)
                result.update(metrics)
                result['ground_truth'] = ground_truth
                result['ground_truth_path'] = gt_path if gt_path else 'embedded'
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                result['error'] = str(e)
        else:
            result['ground_truth'] = None
            result['ground_truth_path'] = None
        
        return result
    
    def process_dataset(
        self, 
        dataset_path: str, 
        output_dir: str = 'results',
        prompt: str = "<image>\nFree OCR.",
        save_individual_results: bool = True,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        max_docs: Optional[int] = None
    ) -> Dict:
        """Process entire OmniDocBench dataset.
        
        Args:
            dataset_path: Path to OmniDocBench dataset directory
            output_dir: Directory to save results
            prompt: OCR prompt to use
            save_individual_results: Whether to save individual document results
            base_size: Base image size for preprocessing
            image_size: Target image size for preprocessing
            crop_mode: Whether to use crop mode
            max_docs: Optional limit on number of documents to process
            
        Returns:
            Dictionary containing aggregated metrics
        """
        print(f"Loading dataset from: {dataset_path}")
        documents = self.loader.load_dataset(dataset_path)
        
        if not documents:
            raise ValueError(f"No documents found in dataset: {dataset_path}")
        
        original_count = len(documents)
        if max_docs is not None:
            documents = random.sample(documents, min(max_docs, len(documents)))
            print(f"Limiting dataset to {len(documents)} documents (requested max_docs={max_docs})")
        
        print(f"Found {len(documents)} documents")
        
        results = []
        errors = []
        
        # Process each document
        for doc_info in tqdm(documents, desc="Processing documents"):
            try:
                # For OmniDocBench, ground truth is embedded in doc_info['gt_text']
                # Pass it directly instead of a file path
                gt_text = doc_info.get('gt_text')
                
                metrics = self.process_single_document(
                    doc_info['doc_path'],
                    gt_path=None,  # No file path, we'll pass text directly
                    gt_text=gt_text,  # Pass embedded text
                    prompt=prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode
                )
                
                # If we have embedded ground truth text, use it
                if gt_text is not None:
                    metrics['ground_truth'] = gt_text
                    # Recalculate metrics with the correct ground truth
                    metrics.update(self.metrics.calculate_all_metrics(
                        metrics.get('ocr_output', ''),
                        gt_text
                    ))
                
                metrics['document_id'] = doc_info['id']
                results.append(metrics)
            except Exception as e:
                error_info = {
                    'document_id': doc_info.get('id', 'unknown'),
                    'document_path': doc_info.get('doc_path', 'unknown'),
                    'error': str(e)
                }
                errors.append(error_info)
                print(f"Error processing {doc_info.get('id', 'unknown')}: {e}")
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(results)
        aggregated['total_documents'] = len(documents)
        aggregated['processed_documents'] = len(results)
        aggregated['errors'] = len(errors)
        aggregated['requested_max_docs'] = max_docs
        aggregated['original_dataset_size'] = original_count
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_individual_results:
            with open(output_path / 'individual_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'aggregated_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
        if errors:
            with open(output_path / 'errors.json', 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_summary(aggregated)
        
        return aggregated
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all documents.
        
        Args:
            results: List of individual document results
            
        Returns:
            Dictionary containing aggregated metrics
        """
        metrics_with_gt = [r for r in results if 'ground_truth' in r and r['ground_truth'] is not None]
        
        if not metrics_with_gt:
            return {
                'message': 'No ground truth available for any documents',
                'total_documents': len(results),
                'documents_with_ground_truth': 0
            }
        
        # Calculate averages
        num_docs = len(metrics_with_gt)
        
        aggregated = {
            'total_documents': len(results),
            'documents_with_ground_truth': num_docs,
            'documents_without_ground_truth': len(results) - num_docs,
            'avg_edit_distance': sum(m.get('edit_distance', 0) for m in metrics_with_gt) / num_docs,
            'avg_normalized_edit_distance': sum(m.get('normalized_edit_distance', 0) for m in metrics_with_gt) / num_docs,
            'avg_character_accuracy': sum(m.get('character_accuracy', 0) for m in metrics_with_gt) / num_docs,
            'avg_word_accuracy': sum(m.get('word_accuracy', 0) for m in metrics_with_gt) / num_docs,
            'avg_sentence_accuracy': sum(m.get('sentence_accuracy', 0) for m in metrics_with_gt) / num_docs,
            'avg_predicted_length': sum(m.get('predicted_length', 0) for m in metrics_with_gt) / num_docs,
            'avg_ground_truth_length': sum(m.get('ground_truth_length', 0) for m in metrics_with_gt) / num_docs,
        }
        
        # Calculate min/max for key metrics
        edit_distances = [m.get('edit_distance', 0) for m in metrics_with_gt]
        char_accuracies = [m.get('character_accuracy', 0) for m in metrics_with_gt]
        word_accuracies = [m.get('word_accuracy', 0) for m in metrics_with_gt]
        
        aggregated.update({
            'min_edit_distance': min(edit_distances) if edit_distances else 0,
            'max_edit_distance': max(edit_distances) if edit_distances else 0,
            'min_character_accuracy': min(char_accuracies) if char_accuracies else 0,
            'max_character_accuracy': max(char_accuracies) if char_accuracies else 0,
            'min_word_accuracy': min(word_accuracies) if word_accuracies else 0,
            'max_word_accuracy': max(word_accuracies) if word_accuracies else 0,
        })
        
        return aggregated
    
    def _print_summary(self, aggregated: Dict):
        """Print summary of aggregated metrics.
        
        Args:
            aggregated: Dictionary containing aggregated metrics
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total documents processed: {aggregated.get('total_documents', 0)}")
        print(f"Documents with ground truth: {aggregated.get('documents_with_ground_truth', 0)}")
        
        if aggregated.get('documents_with_ground_truth', 0) > 0:
            print("\n--- Accuracy Metrics ---")
            print(f"Average Character Accuracy: {aggregated.get('avg_character_accuracy', 0):.4f}")
            print(f"Average Word Accuracy: {aggregated.get('avg_word_accuracy', 0):.4f}")
            print(f"Average Sentence Accuracy: {aggregated.get('avg_sentence_accuracy', 0):.4f}")
            
            print("\n--- Edit Distance Metrics ---")
            print(f"Average Edit Distance: {aggregated.get('avg_edit_distance', 0):.2f}")
            print(f"Average Normalized Edit Distance: {aggregated.get('avg_normalized_edit_distance', 0):.4f}")
            print(f"Edit Distance Range: [{aggregated.get('min_edit_distance', 0)}, {aggregated.get('max_edit_distance', 0)}]")
            
            print("\n--- Length Statistics ---")
            print(f"Average Predicted Length: {aggregated.get('avg_predicted_length', 0):.1f} chars")
            print(f"Average Ground Truth Length: {aggregated.get('avg_ground_truth_length', 0):.1f} chars")
        
        if aggregated.get('errors', 0) > 0:
            print(f"\nErrors encountered: {aggregated.get('errors', 0)}")
        
        print("="*60 + "\n")

