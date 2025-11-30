#!/usr/bin/env python3
"""
Evaluate Simple CNN model on a test dataset.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Import the model class from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_simple_cnn import SimpleCNN, DocumentDataset, get_transforms


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_doc_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for images, labels, doc_ids in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_doc_ids.extend(doc_ids)
    
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = 100 * accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_doc_ids


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Simple CNN model')
    parser.add_argument('--model_path', type=str,
                       default='simple_cnn_output/best_model.pth',
                       help='Path to saved model checkpoint')
    parser.add_argument('--labels_file', type=str,
                       default='optimal_model_labels_threshold.json',
                       help='Path to labels JSON file')
    parser.add_argument('--images_dir', type=str,
                       default='OmniDocBench/images',
                       help='Path to images directory')
    parser.add_argument('--output_dir', type=str,
                       default='simple_cnn_output',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                       default='test',
                       help='Which split to evaluate on')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print(f"\nLoading labels from {args.labels_file}...")
    with open(args.labels_file, 'r') as f:
        all_data = json.load(f)
    
    print(f"Total samples: {len(all_data)}")
    
    # Split data (same as training script)
    from sklearn.model_selection import train_test_split
    train_data, temp_data = train_test_split(
        all_data, test_size=0.2, random_state=42,
        stratify=[d['optimal_model'] for d in all_data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[d['optimal_model'] for d in temp_data]
    )
    
    # Select data split
    if args.split == 'train':
        eval_data = train_data
    elif args.split == 'val':
        eval_data = val_data
    elif args.split == 'test':
        eval_data = test_data
    else:  # all
        eval_data = all_data
    
    print(f"Evaluating on {args.split} split: {len(eval_data)} samples")
    
    # Transforms
    _, val_transform = get_transforms(args.image_size, augment=False)
    
    # Dataset
    eval_dataset = DocumentDataset(eval_data, args.images_dir, val_transform)
    
    # Data loader
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    print("\nLoading model...")
    model = SimpleCNN(num_classes=5, dropout=0.3)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation accuracy when saved: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    print("\nEvaluating...")
    eval_loss, eval_acc, eval_preds, eval_labels, eval_doc_ids = validate(
        model, eval_loader, criterion, device
    )
    
    print(f"\nEvaluation Results:")
    print(f"Loss: {eval_loss:.4f}")
    print(f"Accuracy: {eval_acc:.2f}%")
    
    # Classification report
    class_names = ['tiny', 'small', 'base', 'large', 'gundam']
    print("\nClassification Report:")
    print(classification_report(eval_labels, eval_preds, target_names=class_names))
    
    # Confusion matrix
    cm_path = output_dir / f'confusion_matrix_{args.split}.png'
    plot_confusion_matrix(eval_labels, eval_preds, class_names, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Save predictions
    predictions = []
    for doc_id, true_label, pred_label in zip(eval_doc_ids, eval_labels, eval_preds):
        predictions.append({
            'document_id': doc_id,
            'true_label': class_names[true_label],
            'predicted_label': class_names[pred_label],
            'correct': bool(true_label == pred_label)
        })
    
    results_path = output_dir / f'predictions_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {results_path}")
    
    # Save summary
    summary = {
        'split': args.split,
        'num_samples': len(eval_data),
        'loss': eval_loss,
        'accuracy': eval_acc,
        'model_path': args.model_path,
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_val_acc': checkpoint.get('val_acc', 'N/A')
    }
    
    summary_path = output_dir / f'evaluation_summary_{args.split}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()