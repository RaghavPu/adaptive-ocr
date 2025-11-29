#!/usr/bin/env python3
"""
Train EfficientNet-B0 classifier to predict optimal OCR model size.

Usage:
    python train_classifier.py --epochs 40 --batch_size 32
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DocumentDataset(Dataset):
    """Dataset for document images with optimal model labels."""
    
    def __init__(self, data_list, images_dir, transform=None):
        """
        Args:
            data_list: List of dicts with 'document_id' and 'optimal_model'
            images_dir: Path to directory containing images
            transform: Optional transform to apply
        """
        self.data = data_list
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Label mapping
        self.label_map = {'tiny': 0, 'small': 1, 'base': 2, 'large': 3}
        self.label_names = ['tiny', 'small', 'base', 'large']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        doc_id = item['document_id']
        label = self.label_map[item['optimal_model']]
        
        # Try different image extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            candidate = self.images_dir / f"{doc_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {doc_id}")
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, doc_id


def get_transforms(image_size=224, augment=True):
    """Get training and validation transforms."""
    
    if augment:
        # Heavy augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 based classifier for optimal model prediction."""
    
    def __init__(self, num_classes=4, dropout=0.4, freeze_backbone=True, unfreeze_layers=2):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,  # Remove head
            global_pool=''  # Remove pooling
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            # First freeze everything
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Then unfreeze last few layers for better learning
            # EfficientNet-B0 has blocks 0-6, we unfreeze the last ones
            if unfreeze_layers > 0:
                # Get all named modules
                all_modules = list(self.backbone.named_children())
                # Unfreeze last N blocks
                for name, module in all_modules[-unfreeze_layers:]:
                    for param in module.parameters():
                        param.requires_grad = True
                print(f"✓ Backbone mostly frozen (unfroze last {unfreeze_layers} blocks for adaptation)")
            else:
                print("✓ Backbone completely frozen (only training classifier head)")
        
        # Powerful classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features from backbone
        # Gradients will flow through but frozen params won't update
        features = self.backbone(x)
        # Global average pooling
        features = self.pool(features)
        features = features.flatten(1)
        # Classification (trainable)
        logits = self.classifier(features)
        return logits
    
    def unfreeze_backbone(self, num_layers=None):
        """Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreezes all layers.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("✓ Entire backbone unfrozen")
        else:
            # Unfreeze last N layers
            all_modules = list(self.backbone.named_children())
            for name, module in all_modules[-num_layers:]:
                for param in module.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {num_layers} blocks of backbone")


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    label_counts = Counter(labels)
    total = len(labels)
    
    weights = {}
    for label, count in label_counts.items():
        weights[label] = total / (len(label_counts) * count)
    
    weight_tensor = torch.FloatTensor([weights[i] for i in range(len(weights))])
    return weight_tensor


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_doc_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
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
    
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train optimal model classifier')
    parser.add_argument('--labels_file', type=str, 
                       default='optimal_model_labels.json',
                       help='Path to labels JSON file')
    parser.add_argument('--images_dir', type=str,
                       default='OmniDocBench/images',
                       help='Path to images directory')
    parser.add_argument('--output_dir', type=str,
                       default='classifier_output',
                       help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze backbone (only train classifier head)')
    parser.add_argument('--unfreeze_layers', type=int, default=2,
                       help='Number of last backbone layers to unfreeze (default: 2)')
    parser.add_argument('--unfreeze_epoch', type=int, default=None,
                       help='Epoch to unfreeze backbone for fine-tuning (optional)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load labels
    print(f"\nLoading labels from {args.labels_file}...")
    with open(args.labels_file, 'r') as f:
        all_data = json.load(f)
    
    # Remove gundam class (0 samples)
    all_data = [d for d in all_data if d['optimal_model'] != 'gundam']
    print(f"Total samples: {len(all_data)}")
    
    # Print class distribution
    label_counts = Counter(d['optimal_model'] for d in all_data)
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({100*count/len(all_data):.1f}%)")
    
    # Split data: 80% train, 10% val, 10% test
    train_data, temp_data = train_test_split(
        all_data, test_size=0.2, random_state=args.seed,
        stratify=[d['optimal_model'] for d in all_data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=args.seed,
        stratify=[d['optimal_model'] for d in temp_data]
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=args.image_size,
        augment=not args.no_augment
    )
    
    # Create datasets
    train_dataset = DocumentDataset(train_data, args.images_dir, train_transform)
    val_dataset = DocumentDataset(val_data, args.images_dir, val_transform)
    test_dataset = DocumentDataset(test_data, args.images_dir, val_transform)
    
    # Compute class weights
    train_labels = [train_dataset.label_map[d['optimal_model']] for d in train_data]
    class_weights = compute_class_weights(train_labels)
    print(f"\nClass weights: {class_weights.tolist()}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetClassifier(
        num_classes=4, 
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_layers=args.unfreeze_layers if args.freeze_backbone else 0
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    if args.freeze_backbone:
        print(f"Training strategy: Frozen backbone + trainable classifier")
        if args.unfreeze_epoch:
            print(f"Will unfreeze backbone at epoch {args.unfreeze_epoch}")
    else:
        print(f"Training strategy: Fine-tuning entire model")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Unfreeze backbone if specified
        if args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            print("\n" + "="*60)
            print(f"UNFREEZING BACKBONE FOR FINE-TUNING")
            print("="*60)
            model.unfreeze_backbone()
            # Reduce learning rate when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
            print("="*60 + "\n")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_doc_ids = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    class_names = ['tiny', 'small', 'base', 'large']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        output_dir / 'confusion_matrix.png'
    )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test predictions
    test_results = []
    for doc_id, true_label, pred_label in zip(test_doc_ids, test_labels, test_preds):
        test_results.append({
            'document_id': doc_id,
            'true_label': class_names[true_label],
            'predicted_label': class_names[pred_label],
            'correct': true_label == pred_label
        })
    
    with open(output_dir / 'test_predictions.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print("\nFiles created:")
    print(f"  - best_model.pth (model checkpoint)")
    print(f"  - confusion_matrix.png")
    print(f"  - training_history.json")
    print(f"  - test_predictions.json")


if __name__ == '__main__':
    main()

