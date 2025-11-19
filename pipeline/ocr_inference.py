"""OCR inference wrapper for DeepSeek-OCR using Transformers."""

import os
import torch
import fitz  # PyMuPDF
import tempfile
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union
from transformers import AutoModel, AutoTokenizer

# Patch LlamaFlashAttention2 before model loading
# The model's custom code tries to import this, but it may not be available
# We'll create an alias to regular LlamaAttention as a fallback
try:
    from transformers.models.llama import modeling_llama
    # If LlamaFlashAttention2 doesn't exist, create it as an alias to LlamaAttention
    if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
        from transformers.models.llama.modeling_llama import LlamaAttention
        # Create LlamaFlashAttention2 as an alias to LlamaAttention
        modeling_llama.LlamaFlashAttention2 = LlamaAttention
        print("Patched LlamaFlashAttention2 to use LlamaAttention (flash-attn not available)")
except Exception as e:
    print(f"Warning: Could not patch LlamaFlashAttention2: {e}")


class DeepSeekOCRInference:
    """Wrapper for DeepSeek-OCR model inference using Transformers."""
    
    def __init__(
        self, 
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda',
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        dtype: str = 'bfloat16'
    ):
        """Initialize DeepSeek-OCR model with Transformers.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run inference on ('cuda' or 'cpu')
            tensor_parallel_size: Number of GPUs for tensor parallelism (not used with Transformers)
            max_model_len: Maximum sequence length (not used with Transformers)
            dtype: Model dtype (bfloat16, float16, or float32)
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading DeepSeek-OCR model with Transformers: {model_name}")
        print(f"Using device: {device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True
            )
            
            # Move to device and set dtype
            if device == 'cuda':
                self.model = self.model.eval().cuda()
                if dtype == 'bfloat16':
                    self.model = self.model.to(torch.bfloat16)
                elif dtype == 'float16':
                    self.model = self.model.to(torch.float16)
                elif dtype == 'float32':
                    self.model = self.model.to(torch.float32)
            else:
                self.model = self.model.eval()
            
            print("Transformers model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model with Transformers: {e}")
    
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
            base_size: Base image size for processing (used for image preprocessing)
            image_size: Target image size (used for image preprocessing)
            crop_mode: Whether to use crop mode (used for image preprocessing)
            
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
            base_size: Base image size (for preprocessing if needed)
            image_size: Target image size (for preprocessing if needed)
            crop_mode: Whether to use crop mode (for preprocessing if needed)
            
        Returns:
            OCR text output
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Use the model's infer method (as per HuggingFace documentation)
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False, 
                output_path="results/ocr_output/", 
                eval_mode=True
            )
            
            # The infer method returns the OCR text directly
            if result is None:
                return ""
            elif isinstance(result, str):
                return result.strip()
            elif isinstance(result, dict):
                return result.get('text', result.get('output', '')).strip() or ""
            else:
                return str(result).strip()
                
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise
        finally:
            pass
    
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
