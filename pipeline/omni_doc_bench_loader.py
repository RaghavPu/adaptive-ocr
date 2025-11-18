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
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load OmniDocBench dataset structure.
        
        Args:
            dataset_path: Path to OmniDocBench dataset directory
            
        Returns:
            List of dictionaries containing document info (id, doc_path, gt_path)
        """
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        documents = []
        
        # Common OmniDocBench structure:
        # - dataset/images/ or dataset/documents/
        # - dataset/annotations/ or dataset/ground_truth/
        # - Or flat structure with .json/.txt ground truth files
        
        # Try to find images directory
        images_dir = None
        for possible_dir in ['images', 'documents', 'data', '']:
            test_dir = dataset_dir / possible_dir
            if test_dir.exists():
                images_dir = test_dir
                break
        
        if images_dir is None:
            images_dir = dataset_dir
        
        # Find all document files
        image_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif'}
        for img_file in images_dir.rglob('*'):
            if img_file.suffix.lower() in image_extensions:
                # Find corresponding ground truth
                gt_file = self._find_ground_truth(img_file, dataset_dir)
                
                documents.append({
                    'id': img_file.stem,
                    'doc_path': str(img_file.absolute()),
                    'gt_path': str(gt_file.absolute()) if gt_file else None
                })
        
        return documents
    
    def _find_ground_truth(self, doc_path: Path, dataset_dir: Path) -> Optional[Path]:
        """Find corresponding ground truth file for a document.
        
        Args:
            doc_path: Path to the document file
            dataset_dir: Root directory of the dataset
            
        Returns:
            Path to ground truth file if found, None otherwise
        """
        # Try different possible locations and naming conventions
        possible_dirs = [
            dataset_dir / 'annotations',
            dataset_dir / 'ground_truth',
            dataset_dir / 'labels',
            dataset_dir / 'gt',
            doc_path.parent / 'annotations',
            doc_path.parent / 'ground_truth',
        ]
        
        possible_extensions = ['.json', '.txt', '.gt', '.label']
        
        for gt_dir in possible_dirs:
            if gt_dir.exists():
                for ext in possible_extensions:
                    # Try exact match
                    gt_file = gt_dir / f"{doc_path.stem}{ext}"
                    if gt_file.exists():
                        return gt_file
                    
                    # Try with different naming (e.g., image_001 -> gt_001)
                    if '_' in doc_path.stem:
                        parts = doc_path.stem.split('_')
                        for i in range(len(parts)):
                            alt_name = '_'.join(parts[i:])
                            gt_file = gt_dir / f"{alt_name}{ext}"
                            if gt_file.exists():
                                return gt_file
        
        # Try in same directory
        for ext in possible_extensions:
            gt_file = doc_path.parent / f"{doc_path.stem}{ext}"
            if gt_file.exists():
                return gt_file
        
        return None

