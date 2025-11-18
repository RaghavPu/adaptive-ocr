"""OCR inference wrapper for DeepSeek-OCR."""

import os
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union
import tempfile

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. PDF processing will be limited. Install with: pip install PyMuPDF")


class DeepSeekOCRInference:
    """Wrapper for DeepSeek-OCR model inference."""
    
    def __init__(
        self, 
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda',
        use_flash_attention: bool = True
    ):
        """Initialize DeepSeek-OCR model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run inference on ('cuda' or 'cpu')
            use_flash_attention: Whether to use flash attention
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model_name = model_name
        
        print(f"Loading DeepSeek-OCR model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load model
        attn_impl = 'flash_attention_2' if use_flash_attention else 'sdpa'
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                use_safetensors=True
            )
        except Exception as e:
            print(f"Warning: Could not load with flash_attention_2, trying default: {e}")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True
            )
        
        self.model = self.model.eval()
        
        if self.device == 'cuda':
            self.model = self.model.cuda()
            # Use bfloat16 if available
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                self.model = self.model.to(torch.bfloat16)
            else:
                self.model = self.model.to(torch.float16)
        else:
            self.model = self.model.to(torch.float32)
    
    def process_document(
        self, 
        doc_path: Union[str, Path],
        prompt: str = "<image>\nFree OCR.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> str:
        """Process a document and return OCR text.
        
        Args:
            doc_path: Path to document (PDF or image)
            prompt: Prompt to use for OCR
            base_size: Base image size for processing
            image_size: Target image size
            crop_mode: Whether to use crop mode
            
        Returns:
            OCR text output
        """
        doc_path = Path(doc_path)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        # Handle PDF files
        if doc_path.suffix.lower() == '.pdf':
            return self._process_pdf(doc_path, prompt, base_size, image_size, crop_mode)
        else:
            # Handle image files
            return self._process_image(doc_path, prompt, base_size, image_size, crop_mode)
    
    def _process_image(
        self,
        image_path: Path,
        prompt: str,
        base_size: int,
        image_size: int,
        crop_mode: bool
    ) -> str:
        """Process a single image file.
        
        Args:
            image_path: Path to image file
            prompt: OCR prompt
            base_size: Base image size
            image_size: Target image size
            crop_mode: Whether to use crop mode
            
        Returns:
            OCR text output
        """
        try:
            # Use the model's infer method if available
            if hasattr(self.model, 'infer'):
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,
                    test_compress=False
                )
                # Extract text if result is a dict
                if isinstance(result, dict):
                    return result.get('text', '') or result.get('output', '') or str(result)
                return str(result)
            else:
                # Fallback: try to use the model directly
                # This is a simplified version - actual implementation may vary
                raise NotImplementedError(
                    "Model infer method not available. "
                    "Please ensure DeepSeek-OCR model is properly loaded."
                )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise
    
    def _process_pdf(
        self,
        pdf_path: Path,
        prompt: str,
        base_size: int,
        image_size: int,
        crop_mode: bool
    ) -> str:
        """Process a PDF file page by page.
        
        Args:
            pdf_path: Path to PDF file
            prompt: OCR prompt
            base_size: Base image size
            image_size: Target image size
            crop_mode: Whether to use crop mode
            
        Returns:
            Combined OCR text from all pages
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF"
            )
        
        doc = fitz.open(str(pdf_path))
        results = []
        
        print(f"Processing PDF with {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            print(f"Processing page {page_num + 1}/{len(doc)}")
            page = doc[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                img.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # Process the page image
                page_result = self._process_image(
                    Path(tmp_path),
                    prompt,
                    base_size,
                    image_size,
                    crop_mode
                )
                results.append(page_result)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        doc.close()
        
        # Combine results with page separators
        return '\n\n--- Page Break ---\n\n'.join(results)
    
    def process_batch(
        self,
        doc_paths: List[Union[str, Path]],
        prompt: str = "<image>\nFree OCR.",
        **kwargs
    ) -> List[str]:
        """Process multiple documents in batch.
        
        Args:
            doc_paths: List of document paths
            prompt: OCR prompt to use
            **kwargs: Additional arguments for process_document
            
        Returns:
            List of OCR text outputs
        """
        results = []
        for i, doc_path in enumerate(doc_paths):
            print(f"Processing document {i+1}/{len(doc_paths)}: {doc_path}")
            try:
                result = self.process_document(doc_path, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                results.append("")
        
        return results

