"""Metrics calculation for OCR evaluation."""

import Levenshtein
from typing import Dict, Tuple


class OCRMetrics:
    """Calculate various metrics for OCR evaluation."""
    
    @staticmethod
    def edit_distance(predicted: str, ground_truth: str) -> int:
        """Calculate Levenshtein edit distance between predicted and ground truth.
        
        Args:
            predicted: The OCR output text
            ground_truth: The ground truth text
            
        Returns:
            Edit distance (number of character operations needed)
        """
        return Levenshtein.distance(predicted, ground_truth)
    
    @staticmethod
    def normalized_edit_distance(predicted: str, ground_truth: str) -> float:
        """Calculate normalized edit distance (0-1 scale).
        
        Args:
            predicted: The OCR output text
            ground_truth: The ground truth text
            
        Returns:
            Normalized edit distance (0 = identical, 1 = completely different)
        """
        if len(ground_truth) == 0 and len(predicted) == 0:
            return 0.0
        if len(ground_truth) == 0:
            return 1.0
        
        edit_dist = OCRMetrics.edit_distance(predicted, ground_truth)
        max_len = max(len(predicted), len(ground_truth))
        return edit_dist / max_len if max_len > 0 else 0.0
    
    @staticmethod
    def character_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy.
        
        Args:
            predicted: The OCR output text
            ground_truth: The ground truth text
            
        Returns:
            Character accuracy (0-1, where 1 is perfect match)
        """
        if len(ground_truth) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        
        edit_dist = OCRMetrics.edit_distance(predicted, ground_truth)
        max_len = max(len(predicted), len(ground_truth))
        return 1.0 - (edit_dist / max_len) if max_len > 0 else 1.0
    
    @staticmethod
    def word_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate word-level accuracy.
        
        Args:
            predicted: The OCR output text
            ground_truth: The ground truth text
            
        Returns:
            Word accuracy (0-1, where 1 is perfect match)
        """
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Calculate word-level edit distance
        correct = sum(1 for p, g in zip(pred_words, gt_words) if p == g)
        return correct / len(gt_words)
    
    @staticmethod
    def sentence_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate sentence-level accuracy (exact match).
        
        Args:
            predicted: The OCR output text
            ground_truth: The ground truth text
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0
    
    @staticmethod
    def calculate_all_metrics(predicted: str, ground_truth: str) -> Dict:
        """Calculate all metrics at once.
        
        Args:
            predicted: The OCR output text
            ground_truth: The ground truth text
            
        Returns:
            Dictionary containing all calculated metrics
        """
        return {
            'edit_distance': OCRMetrics.edit_distance(predicted, ground_truth),
            'normalized_edit_distance': OCRMetrics.normalized_edit_distance(predicted, ground_truth),
            'character_accuracy': OCRMetrics.character_accuracy(predicted, ground_truth),
            'word_accuracy': OCRMetrics.word_accuracy(predicted, ground_truth),
            'sentence_accuracy': OCRMetrics.sentence_accuracy(predicted, ground_truth),
            'predicted_length': len(predicted),
            'ground_truth_length': len(ground_truth),
            'predicted_word_count': len(predicted.split()),
            'ground_truth_word_count': len(ground_truth.split()),
        }

