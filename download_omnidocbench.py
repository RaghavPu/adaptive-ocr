#!/usr/bin/env python3
"""Download OmniDocBench dataset from HuggingFace."""

import os
import json 
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from datasets import load_dataset 
from huggingface_hub import hf_hub_download, list_repo_files

# Load .env file first, then get token
load_dotenv()  # This loads the .env file
HF_TOKEN = os.getenv('HF_TOKEN', '')

def download_omnidocbench_documents(output_dir: str):
    """Download OmniDocBench images by loading the dataset.
    
    Args:
        output_dir: Directory to save images
        
    Returns:
        Path to the images directory
    """
    
    repo_id = "opendatalab/OmniDocBench"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Listing image files from OmniDocBench repository...")
    # List all files in the images directory
    all_files = list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
    # Filter for image files in the images/ directory
    image_files = [f for f in all_files if f.startswith("images/") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.tif'))]
    print(f"Found {len(image_files)} image files in repository")
    
    print(f"Downloading {len(image_files)} document images...")
    
    # Download each image file
    downloaded_count = 0
    for image_file in tqdm(image_files, desc="Downloading images"):
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=image_file,
                repo_type="dataset",
                local_dir=str(output_path),
                token=HF_TOKEN
            )
            downloaded_count += 1
        except Exception as e:
            print(f"Warning: Failed to download {image_file}: {e}")
            continue
    
    print(f"✓ Downloaded {downloaded_count} images to {output_path}")
    return output_path
    

def download_omnidocbench_annotation(output_path: Path) -> Optional[Path]:
    """Download the OmniDocBench.json annotation file.
    
    Args:
        output_path: Directory to save the file
        
    Returns:
        Path to the downloaded file if successful, None otherwise
    """
    repo_id = "opendatalab/OmniDocBench"
    print("Downloading OmniDocBench.json annotation file...")
    json_file = hf_hub_download(
        repo_id=repo_id,
        filename="OmniDocBench.json",
        repo_type="dataset",
        local_dir=str(output_path),
        token=HF_TOKEN
    )
    print(f"✓ Downloaded OmniDocBench.json: {json_file}")
    return Path(json_file)

if __name__ == "__main__":
    download_omnidocbench_documents("OmniDocBench")
    download_omnidocbench_annotation("OmniDocBench/annotations")