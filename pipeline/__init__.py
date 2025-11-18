"""Pipeline for evaluating DeepSeek-OCR on OmniDocBench documents."""

from .document_processor import DocumentProcessor
from .ocr_inference import DeepSeekOCRInference
from .metrics import OCRMetrics
from .omni_doc_bench_loader import OmniDocBenchLoader

__all__ = [
    'DocumentProcessor',
    'DeepSeekOCRInference',
    'OCRMetrics',
    'OmniDocBenchLoader',
]

