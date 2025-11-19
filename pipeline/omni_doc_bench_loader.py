"""Loader for OmniDocBench dataset."""

from pathlib import Path
import json
from typing import List, Dict, Optional


class OmniDocBenchLoader:
    """Load documents and ground truth from OmniDocBench dataset."""
    
    def load_ground_truth(self, gt_path: str) -> str:
        """Load ground truth text from file.
        
        Args:
            gt_path: Path to ground truth file (JSON or text)
            
        Returns:
            Ground truth text content
        """
        gt_file = Path(gt_path)
        
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        if gt_path.endswith('.json'):
            with open(gt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Try common keys for text content
                if isinstance(data, dict):
                    return data.get('text', '') or data.get('content', '') or data.get('gt', '') or str(data)
                elif isinstance(data, list):
                    # If it's a list, try to extract text from elements
                    return '\n'.join(str(item) for item in data)
                else:
                    return str(data)
        else:
            # Assume text file
            with open(gt_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _extract_text_from_layout_dets(self, layout_dets: List[Dict]) -> str:
        """Extract all text from layout_dets annotations.
        
        Args:
            layout_dets: List of layout detection dictionaries
            
        Returns:
            Combined text from all text-containing elements
        """
        texts = []
        for det in layout_dets:
            # Skip ignored elements
            if det.get('ignore', False):
                continue
            
            # Extract text from main field
            if 'text' in det and det['text']:
                texts.append(det['text'])
            
            # Extract text from line_with_spans
            if 'line_with_spans' in det:
                for line in det['line_with_spans']:
                    if isinstance(line, dict) and 'text' in line and line['text']:
                        texts.append(line['text'])
        
        return '\n'.join(texts)
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load OmniDocBench dataset structure.
        
        Args:
            dataset_path: Path to OmniDocBench dataset directory
            
        Returns:
            List of dictionaries containing document info (id, doc_path, gt_text)
        """
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        # Find the annotations JSON file
        annotations_file = dataset_dir / 'annotations' / 'OmniDocBench.json'
        if not annotations_file.exists():
            raise FileNotFoundError(
                f"OmniDocBench annotations file not found: {annotations_file}\n"
                f"Expected structure: {dataset_path}/annotations/OmniDocBench.json"
            )
        
        # Find the images directory
        images_dir = dataset_dir / 'images'
        if not images_dir.exists():
            raise FileNotFoundError(
                f"OmniDocBench images directory not found: {images_dir}\n"
                f"Expected structure: {dataset_path}/images/"
            )
        
        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"Found {len(annotations)} annotation entries")
        
        # Build a mapping of image_path -> annotation entry
        image_to_annotation = {}
        for entry in annotations:
            if 'page_info' in entry and 'image_path' in entry['page_info']:
                image_path = entry['page_info']['image_path']
                image_to_annotation[image_path] = entry
        
        print(f"Found {len(image_to_annotation)} unique image paths in annotations")
        
        # Find all image files and match with annotations
        documents = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif'}
        
        # Get all image files
        image_files = []
        for img_file in images_dir.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                image_files.append(img_file)
        
        print(f"Found {len(image_files)} image files in directory")
        
        matched_count = 0
        for img_file in image_files:
            # Try to match by filename (just the name, not full path)
            img_filename = img_file.name
            
            # Check if this image has an annotation
            if img_filename in image_to_annotation:
                entry = image_to_annotation[img_filename]
                
                # Extract ground truth text from layout_dets
                layout_dets = entry.get('layout_dets', [])
                gt_text = self._extract_text_from_layout_dets(layout_dets)
                
                documents.append({
                    'id': img_file.stem,
                    'doc_path': str(img_file.absolute()),
                    'gt_path': None,  # Ground truth is embedded, not a file
                    'gt_text': gt_text,  # Direct text content
                    'annotation_entry': entry  # Store full entry for reference
                })
                matched_count += 1
            else:
                # Image without annotation - still process but without ground truth
                documents.append({
                    'id': img_file.stem,
                    'doc_path': str(img_file.absolute()),
                    'gt_path': None,
                    'gt_text': None,
                    'annotation_entry': None
                })
        
        print(f"Matched {matched_count} images with annotations out of {len(image_files)} total images")
        
        return documents

