# Adaptive Optical Compression for DeepSeek OCR

Adaptive OCR is designed to automatically select optimal compression level for DeepSeek-OCR based on document characteristics in order to reduce token usage without compromising reconstruction quality. 

## Repository Layout
```
├─ download_omnidocbench.py        # Dataset fetcher (images + annotations)
├─ evaluate/                       # CLI entrypoints
│  ├─ evaluate.py                  # Main evaluation script
│  └─ evaluate_all_models.py       # Batch runner across compression presets
├─ pipeline/
│  ├─ document_processor.py        # Orchestrates OCR + metrics
│  ├─ ocr_inference.py             # DeepSeek-OCR HF wrapper
│  ├─ metrics.py                   # Character/word/edit-distance metrics
│  └─ omni_doc_bench_loader.py     # Dataset loader w/ embedded GT text
├─ results/                        # Evaluation outputs (gitignored)
├─ calc_summary_stats.py           # Data-source counts (OmniDocBench)
├─ calc_reconstruction_acc.py      # Normalized edit distance vs compression/data_source table
├─ OmniDocBench/                   # OmniDocBench Dataset (gitignored)
└─ requirements_vm.txt             # Python dependencies
```

## Environment Setup
### VM Setup
1. **Python & CUDA**  
   - Python ≥3.10 (tested with 3.12.8).  
   - CUDA-capable GPU recommended

2. **Install dependencies**
   ```bash
   conda create -n adaptive-ocr python=3.12 -y
   conda activate adaptive-ocr
   pip install -r requirements_vm.txt
   ```

3. **Hugging Face authentication**  
   - Obtain a personal access token with dataset read permissions.  
   - Store it in a `.env` file or export `HF_TOKEN` before running download/eval scripts:
     ```bash
     echo "HF_TOKEN=hf_xxx" > .env
     ```

### Local Setup 
TODO

## Dataset Preparation
```bash
python download_omnidocbench.py
```
- Images land in `OmniDocBench/images/`, annotations in `OmniDocBench/annotations/OmniDocBench.json`.
- `pipeline/omni_doc_bench_loader.py` expects the above layout; adjust paths via CLI flags if you relocate the dataset.

## Running Evaluations

### Single Document
```bash
python evaluate/evaluate.py \
  --doc_path /path/to/page.png \
  --gt_path /path/to/ground_truth.txt \
  --output_dir results/ \
  --image_size 640 --base_size 640 --prompt "<image>\nFree OCR."
```

### OmniDocBench Subset
```bash
python evaluate/evaluate.py \
  --dataset_path OmniDocBench \
  --output_dir results/tiny \
  --image_size 512 \
  --base_size 512 \
  --max_docs 100
```

Key flags:
- `--prompt` customizes the grounding text passed to DeepSeek-OCR.
- `--model_name` lets you point to alternative checkpoints.
- `--max_docs` (optional) randomly samples documents for quick experiments.
- `--crop_mode` (optional) uses gundam mode for dynamic tiling. 
- `--no_save_individual` skips writing `individual_results.json` if storage is a concern.

### Batch Across Compression Levels
Preset image sizes live in `evaluate/evaluate_all_models.py`.
```bash
python evaluate/evaluate_all_models.py \
  --dataset_path OmniDocBench \
  --models tiny small base \
  --max_docs 200
```
Each run writes to `results/<model_key>/`.

## Outputs & Logs
- `results/<run>/aggregated_metrics.json` – macro averages (edit distance, accuracies, lengths, error counts).
- `results/<run>/individual_results.json` – per-document OCR text, metadata, and metrics.
- `results/<run>/errors.json` – only emitted when a document fails.
- `results/ocr_output/images/` – raw model dumps when `DeepSeek-OCR` saves intermediates.

## Analysis Utilities

### OmniDocBench Data-Source Breakdown
```bash
python calc_summary_stats.py \
  --dataset-dir OmniDocBench
```
Prints the total document count plus counts/percentages for each `page_info.page_attribute.data_source`.

### Normalized Edit Distance vs Compression/Data Source
```bash
python calc_reconstruction_acc.py \
  --dataset-dir OmniDocBench \
  --results-root results
```
Collects every `individual_results.json`, joins with annotations by `document_id`, and emits a table with compression levels as rows and document types (data sources) as columns (values are mean normalized edit distance). Skipped entries (missing GT, doc IDs, etc.) are summarized below the table.

## Troubleshooting
- **Model load errors** – ensure transformers==4.46 and that flash-attn dependencies are either installed or let `pipeline/ocr_inference.py` patch `LlamaFlashAttention2`.
- **Missing HF token** – both dataset download and model pulls may require authentication; set `HF_TOKEN` or run `huggingface-cli login`.
- **OOM during inference** – reduce `image_size/base_size`, enable `--crop_mode`, or switch to the `tiny` preset.
- **Empty OCR outputs** – check GPU logs and `results/errors.json`; DeepSeek-OCR sometimes returns empty strings if prompts or max lengths are invalid.

## Contributing
1. Fork & branch.
2. Keep scripts runnable via CLI and documented here.
3. Run linting/tests relevant to your changes.
4. Open a PR detailing motivation, setup, and sample outputs.

This README will evolve as new compression policies or models are integrated—please keep it updated when adding features.