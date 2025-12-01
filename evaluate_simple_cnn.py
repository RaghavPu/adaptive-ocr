#!/usr/bin/env python3
"""
Evaluate Simple CNN model by predicting optimal compression levels for images.

Calculates:
1. Total tokens saved by using predicted compression vs always using largest model
2. Accuracy drop for misclassified documents (when we predict smaller model than optimal)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import Counter
import sys

# Import the model class from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_simple_cnn import SimpleCNN, get_transforms


# DeepSeek-OCR model configurations
# Based on https://github.com/deepseek-ai/DeepSeek-OCR
MODEL_CONFIGS = {
    'tiny': {'image_size': 512, 'base_size': 512, 'crop_mode': False},
    'small': {'image_size': 640, 'base_size': 640, 'crop_mode': False},
    'base': {'image_size': 1024, 'base_size': 1024, 'crop_mode': False},
    'large': {'image_size': 1280, 'base_size': 1280, 'crop_mode': False},
    'gundam': {'image_size': 640, 'base_size': 1024, 'crop_mode': True},
}

MODEL_ORDER = ['tiny', 'small', 'base', 'large', 'gundam']

# DeepSeek-OCR vision encoder constants
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4


def calculate_vision_tokens(model_name, image_width=None, image_height=None):
    """
    Calculate the number of vision tokens for a given model configuration.

    Based on DeepSeek-OCR's token calculation formula:
    - patch_size = 16
    - downsample_ratio = 4
    - h = w = ceil((size / patch_size) / downsample_ratio)
    - For non-crop: tokens = h * (w + 1) + 1
    - For crop: global_tokens + local_tokens + 1

    Args:
        model_name: One of 'tiny', 'small', 'base', 'large', 'gundam'
        image_width: Original image width (required for gundam/crop_mode)
        image_height: Original image height (required for gundam/crop_mode)

    Returns:
        Number of vision tokens
    """
    import math

    config = MODEL_CONFIGS[model_name]
    image_size = config['image_size']
    base_size = config['base_size']
    crop_mode = config['crop_mode']

    # Calculate feature map dimensions
    h = w = math.ceil((base_size / PATCH_SIZE) / DOWNSAMPLE_RATIO)

    if not crop_mode:
        # Non-crop mode: single global view
        # Formula: h * (w + 1) + 1
        return h * (w + 1) + 1
    else:
        # Crop mode (gundam): global view + local tiles
        if image_width is None or image_height is None:
            raise ValueError("image_width and image_height required for crop_mode")

        # Global view tokens
        global_tokens = h * (w + 1)

        # Calculate number of tiles based on image dimensions
        num_width_tiles, num_height_tiles = count_tiles(
            image_width, image_height, image_size
        )

        # Local view feature dimensions
        h2 = w2 = math.ceil((image_size / PATCH_SIZE) / DOWNSAMPLE_RATIO)

        # Local view tokens
        local_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)

        return global_tokens + local_tokens + 1


def count_tiles(orig_width, orig_height, image_size=640, min_num=1, max_num=9):
    """
    Calculate optimal tile arrangement for an image.

    Based on DeepSeek-OCR's count_tiles implementation.
    Finds the tile configuration (num_width, num_height) that best matches
    the original image's aspect ratio.

    Args:
        orig_width: Original image width
        orig_height: Original image height
        image_size: Size of each tile (default 640)
        min_num: Minimum tiles per dimension (default 1)
        max_num: Maximum tiles per dimension (default 9)

    Returns:
        Tuple of (num_width_tiles, num_height_tiles)
    """
    aspect_ratio = orig_width / orig_height

    # Generate candidate tile configurations
    candidates = []
    for i in range(min_num, max_num + 1):
        for j in range(min_num, max_num + 1):
            candidates.append((i, j))

    # Find best matching aspect ratio
    best_ratio = None
    best_diff = float('inf')

    for (w_tiles, h_tiles) in candidates:
        candidate_ratio = w_tiles / h_tiles
        diff = abs(candidate_ratio - aspect_ratio)

        # Also check if image area is reasonable relative to tile area
        tile_area = (w_tiles * image_size) * (h_tiles * image_size)
        image_area = orig_width * orig_height

        # Prefer configurations where image fills at least half the tile area
        if image_area < tile_area * 0.5 and w_tiles * h_tiles > 1:
            continue

        if diff < best_diff:
            best_diff = diff
            best_ratio = (w_tiles, h_tiles)

    # Fallback to (1, 1) if no good match found
    if best_ratio is None:
        best_ratio = (1, 1)

    return best_ratio


class ImageOnlyDataset(Dataset):
    """Dataset for inference - only loads images, no labels needed."""

    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Find all images
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            self.image_files.extend(self.images_dir.glob(ext))

        self.image_files = sorted(self.image_files)
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        doc_id = img_path.stem  # filename without extension

        image = Image.open(img_path).convert('RGB')
        # Store original dimensions before transform
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)

        return image, doc_id, orig_width, orig_height


def predict(model, dataloader, device):
    """Run inference on all images."""
    model.eval()
    all_preds = []
    all_doc_ids = []
    all_probs = []
    all_dimensions = []  # Store (width, height) for each image

    class_names = MODEL_ORDER

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Predicting')
        for images, doc_ids, widths, heights in pbar:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_doc_ids.extend(doc_ids)
            # Store dimensions as list of tuples
            for w, h in zip(widths.tolist(), heights.tolist()):
                all_dimensions.append((w, h))

    return all_preds, all_probs, all_doc_ids, all_dimensions


def calculate_token_savings(predictions, class_names, dimensions):
    """
    Calculate total tokens saved vs all resolution levels.

    Args:
        predictions: List of predicted model indices
        class_names: List of model names
        dimensions: List of (width, height) tuples for each image

    Returns:
        Dictionary with token statistics
    """
    total_predicted_tokens = 0
    total_random_tokens = 0
    # Track tokens for each resolution level
    total_per_model_tokens = {model: 0 for model in MODEL_ORDER}

    per_image_tokens = []

    for pred_idx, (width, height) in zip(predictions, dimensions):
        pred_model = class_names[pred_idx]

        # Calculate tokens for predicted model
        pred_tokens = calculate_vision_tokens(pred_model, width, height)
        total_predicted_tokens += pred_tokens

        # Calculate tokens for each resolution level
        model_tokens = []
        for model_name in MODEL_ORDER:
            tokens = calculate_vision_tokens(model_name, width, height)
            total_per_model_tokens[model_name] += tokens
            model_tokens.append(tokens)

        # Calculate average tokens across all models (random baseline)
        avg_tokens = sum(model_tokens) / len(model_tokens)
        total_random_tokens += avg_tokens

        per_image_tokens.append({
            'predicted_model': pred_model,
            'predicted_tokens': pred_tokens,
            'gundam_tokens': total_per_model_tokens['gundam'],
        })

    # Calculate savings vs gundam (for backwards compatibility)
    tokens_saved_vs_gundam = total_per_model_tokens['gundam'] - total_predicted_tokens
    percent_saved_vs_gundam = 100 * tokens_saved_vs_gundam / total_per_model_tokens['gundam'] if total_per_model_tokens['gundam'] > 0 else 0

    tokens_saved_vs_random = total_random_tokens - total_predicted_tokens
    percent_saved_vs_random = 100 * tokens_saved_vs_random / total_random_tokens if total_random_tokens > 0 else 0

    return {
        'predicted_tokens': total_predicted_tokens,
        'gundam_baseline_tokens': total_per_model_tokens['gundam'],
        'random_baseline_tokens': total_random_tokens,
        'per_model_tokens': total_per_model_tokens,
        'tokens_saved_vs_gundam': tokens_saved_vs_gundam,
        'percent_saved_vs_gundam': percent_saved_vs_gundam,
        'tokens_saved_vs_random': tokens_saved_vs_random,
        'percent_saved_vs_random': percent_saved_vs_random,
        'avg_tokens_per_image': total_predicted_tokens / len(predictions) if predictions else 0,
    }


def load_results_from_folder(results_dir):
    """
    Load per-document accuracies from results folder structure.

    Expected structure:
        results_dir/
            tiny/individual_results.json
            small/individual_results.json
            base/individual_results.json
            large/individual_results.json
            gundam/individual_results.json

    Returns dict: {document_id: {model_name: character_accuracy, ...}, ...}
    """
    results_dir = Path(results_dir)
    all_accuracies = {}

    for model_name in MODEL_ORDER:
        results_file = results_dir / model_name / 'individual_results.json'
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping {model_name}")
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        for item in results:
            doc_id = item['document_id']
            if doc_id not in all_accuracies:
                all_accuracies[doc_id] = {}
            all_accuracies[doc_id][model_name] = item['character_accuracy']

    return all_accuracies


def calculate_accuracy_drop(predictions, doc_ids, results_dir, class_names):
    """
    Calculate accuracy drop for predictions vs all resolution levels.

    Compares:
    - Predicted model's accuracy for each document
    - Each individual resolution level (tiny, small, base, large, gundam)
    - Random baseline: average accuracy across all 5 models for each document
    """
    # Load accuracies from results folder
    all_accuracies = load_results_from_folder(results_dir)

    predicted_accuracies = []
    random_baseline_accuracies = []
    # Track accuracies for each resolution level
    per_model_accuracies = {model: [] for model in MODEL_ORDER}

    for pred_idx, doc_id in zip(predictions, doc_ids):
        if doc_id not in all_accuracies:
            continue

        accuracies = all_accuracies[doc_id]
        pred_model = class_names[pred_idx]

        # Skip if we don't have accuracy for the predicted model
        if pred_model not in accuracies:
            continue

        # Predicted accuracy
        pred_acc = accuracies[pred_model]
        predicted_accuracies.append(pred_acc)

        # Random baseline: average of all available model accuracies
        available_accs = [accuracies[m] for m in MODEL_ORDER if m in accuracies]
        random_baseline_accuracies.append(sum(available_accs) / len(available_accs))

        # Track accuracy for each resolution level
        for model in MODEL_ORDER:
            if model in accuracies:
                per_model_accuracies[model].append(accuracies[model])

    n = len(predicted_accuracies)
    avg_predicted = sum(predicted_accuracies) / n if n > 0 else 0
    avg_random = sum(random_baseline_accuracies) / n if n > 0 else 0

    # Calculate average accuracy for each resolution level
    avg_per_model = {}
    for model in MODEL_ORDER:
        accs = per_model_accuracies[model]
        avg_per_model[model] = sum(accs) / len(accs) if accs else 0

    return {
        'total_evaluated': n,
        'avg_predicted_accuracy': avg_predicted,
        'avg_random_baseline_accuracy': avg_random,
        'per_model_accuracies': avg_per_model,
        'accuracy_vs_random': avg_predicted - avg_random,
        'accuracy_vs_gundam': avg_predicted - avg_per_model.get('gundam', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='Run inference with Simple CNN model')
    parser.add_argument('--model_path', type=str,
                       default='simple_cnn_output/best_model.pth',
                       help='Path to saved model checkpoint')
    parser.add_argument('--images_dir', type=str,
                       default='OmniDocBench/images',
                       help='Path to images directory')
    parser.add_argument('--results_dir', type=str,
                       default=None,
                       help='Optional: Path to results folder for accuracy drop calculation')
    parser.add_argument('--output_dir', type=str,
                       default='simple_cnn_output',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    _, val_transform = get_transforms(args.image_size, augment=False)

    # Dataset
    print(f"\nLoading images from {args.images_dir}...")
    dataset = ImageOnlyDataset(args.images_dir, val_transform)

    if len(dataset) == 0:
        print("Error: No images found!")
        return

    # Data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Create model
    print("\nLoading model...")
    model = SimpleCNN(num_classes=5, dropout=0.3)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy when saved: {checkpoint.get('val_acc'):.2f}%")

    # Run inference
    print("\nRunning inference...")
    predictions, probabilities, doc_ids, dimensions = predict(model, dataloader, device)

    class_names = MODEL_ORDER

    # Count predictions
    pred_counts = Counter(class_names[p] for p in predictions)
    print(f"\nPrediction distribution:")
    for model_name in MODEL_ORDER:
        count = pred_counts.get(model_name, 0)
        pct = 100 * count / len(predictions)
        print(f"  {model_name:8s}: {count:4d} ({pct:5.1f}%)")

    # Print token counts for each model (for reference)
    print(f"\nVision tokens per model (fixed-size models):")
    for model_name in MODEL_ORDER:
        if model_name != 'gundam':
            tokens = calculate_vision_tokens(model_name)
            print(f"  {model_name:8s}: {tokens:4d} tokens")
        else:
            print(f"  {model_name:8s}: dynamic (depends on image size)")

    # Calculate token savings
    print("\n" + "="*60)
    print("TOKEN SAVINGS ANALYSIS")
    print("="*60)

    token_stats = calculate_token_savings(predictions, class_names, dimensions)
    print(f"\nWith predictions:              {token_stats['predicted_tokens']:,} tokens")
    print(f"Baseline (random selection):   {token_stats['random_baseline_tokens']:,.0f} tokens")
    print(f"\nTokens by resolution level:")
    for model in MODEL_ORDER:
        model_tokens = token_stats['per_model_tokens'][model]
        diff = model_tokens - token_stats['predicted_tokens']
        pct = 100 * diff / model_tokens if model_tokens > 0 else 0
        print(f"  {model:8s}: {model_tokens:,} tokens  (predicted saves {diff:+,} / {pct:+.1f}%)")
    print(f"\nTokens saved vs random:        {token_stats['tokens_saved_vs_random']:,.0f} ({token_stats['percent_saved_vs_random']:.1f}%)")
    print(f"Average tokens per image:      {token_stats['avg_tokens_per_image']:.1f}")

    # Calculate accuracy drop if results folder provided
    accuracy_stats = None
    if args.results_dir and Path(args.results_dir).exists():
        print("\n" + "="*60)
        print("ACCURACY ANALYSIS")
        print("="*60)

        accuracy_stats = calculate_accuracy_drop(
            predictions, doc_ids, args.results_dir, class_names
        )

        print(f"\nTotal documents evaluated: {accuracy_stats['total_evaluated']}")
        print(f"\nAverage accuracy with predictions: {accuracy_stats['avg_predicted_accuracy']*100:.2f}%")
        print(f"Average accuracy with random selection: {accuracy_stats['avg_random_baseline_accuracy']*100:.2f}%")
        print(f"\nAccuracy by resolution level:")
        for model in MODEL_ORDER:
            acc = accuracy_stats['per_model_accuracies'].get(model, 0)
            diff = accuracy_stats['avg_predicted_accuracy'] - acc
            print(f"  {model:8s}: {acc*100:.2f}%  (predicted vs {model}: {diff*100:+.2f}%)")

    # Save evaluation results (token savings and accuracy)
    eval_results = {
        'token_savings': token_stats,
    }
    if accuracy_stats:
        eval_results['accuracy'] = accuracy_stats

    eval_path = output_dir / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_path}")

    # Save predictions
    results = []
    for doc_id, pred_idx, probs, (width, height) in zip(doc_ids, predictions, probabilities, dimensions):
        pred_model = class_names[pred_idx]
        pred_tokens = calculate_vision_tokens(pred_model, width, height)
        gundam_tokens = calculate_vision_tokens('gundam', width, height)
        results.append({
            'document_id': doc_id,
            'predicted_model': pred_model,
            'confidence': float(probs[pred_idx]),
            'probabilities': {name: float(probs[i]) for i, name in enumerate(class_names)},
            'image_dimensions': {'width': width, 'height': height},
            'predicted_tokens': pred_tokens,
            'gundam_tokens': gundam_tokens,
            'tokens_saved': gundam_tokens - pred_tokens,
        })

    results_path = output_dir / 'predictions.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPredictions saved to: {results_path}")

    # Save summary
    summary = {
        'num_images': len(predictions),
        'prediction_distribution': dict(pred_counts),
        'token_savings': token_stats,
        'model_path': args.model_path,
    }
    if accuracy_stats:
        summary['accuracy_analysis'] = accuracy_stats

    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
