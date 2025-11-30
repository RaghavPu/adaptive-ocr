#!/usr/bin/env python3
"""
Train a simple CNN from scratch for document classification.

Lightweight architecture designed for limited data (1000 samples).
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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


class SimpleCNN(nn.Module):
    """Simple CNN for document classification - train from scratch."""
    
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        
        # Simple but effective architecture
        # Input: 3 x 224 x 224
        
        # Block 1: 3 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # -> 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 32 x 56 x 56
        )
        
        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 64 x 28 x 28
        )
        
        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 128 x 14 x 14
        )
        
        # Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 256 x 7 x 7
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> 256 x 1 x 1
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class DocumentDataset(Dataset):
    """Dataset for document images."""
    
    def __init__(self, data_list, images_dir, transform=None):
        self.data = data_list
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Label mapping (5 classes now)
        self.label_map = {'tiny': 0, 'small': 1, 'base': 2, 'large': 3, 'gundam': 4}
        self.label_names = ['tiny', 'small', 'base', 'large', 'gundam']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        doc_id = item['document_id']
        label = self.label_map[item['optimal_model']]
        
        # Try different extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            candidate = self.images_dir / f"{doc_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {doc_id}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, doc_id


def get_transforms(image_size=224, augment=True):
    """Get training and validation transforms."""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    label_counts = Counter(labels)
    total = len(labels)
    
    weights = {}
    for label, count in label_counts.items():
        weights[label] = total / (len(label_counts) * count)
    
    weight_tensor = torch.FloatTensor([weights[i] for i in range(len(weights))])
    return weight_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
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


def main():
    parser = argparse.ArgumentParser(description='Train simple CNN from scratch')
    parser.add_argument('--labels_file', type=str, 
                       default='optimal_model_labels_threshold.json',
                       help='Path to labels JSON file')
    parser.add_argument('--images_dir', type=str,
                       default='OmniDocBench/images',
                       help='Path to images directory')
    parser.add_argument('--output_dir', type=str,
                       default='simple_cnn_output',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
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
    
    print(f"Total samples: {len(all_data)}")
    
    # Class distribution
    label_counts = Counter(d['optimal_model'] for d in all_data)
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({100*count/len(all_data):.1f}%)")
    
    # Split data
    train_data, temp_data = train_test_split(
        all_data, test_size=0.2, random_state=args.seed,
        stratify=[d['optimal_model'] for d in all_data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=args.seed,
        stratify=[d['optimal_model'] for d in temp_data]
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # Transforms
    train_transform, val_transform = get_transforms(args.image_size, augment=True)
    
    # Datasets
    train_dataset = DocumentDataset(train_data, args.images_dir, train_transform)
    val_dataset = DocumentDataset(val_data, args.images_dir, val_transform)
    test_dataset = DocumentDataset(test_data, args.images_dir, val_transform)
    
    # Class weights
    train_labels = [train_dataset.label_map[d['optimal_model']] for d in train_data]
    class_weights = compute_class_weights(train_labels)
    print(f"\nClass weights: {class_weights.tolist()}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    print("\nInitializing Simple CNN (training from scratch)...")
    model = SimpleCNN(num_classes=5, dropout=args.dropout)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✓ All parameters trainable (no pre-training)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=5, verbose=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
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
    print(f"Training completed! Best val acc: {best_val_acc:.2f}%")
    
    # Test evaluation
    print("\nLoading best model for testing...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_doc_ids = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    class_names = ['tiny', 'small', 'base', 'large', 'gundam']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, class_names,
                         output_dir / 'confusion_matrix.png')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test results
    test_results = []
    for doc_id, true_label, pred_label in zip(test_doc_ids, test_labels, test_preds):
        test_results.append({
            'document_id': doc_id,
            'true_label': class_names[true_label],
            'predicted_label': class_names[pred_label],
            'correct': bool(true_label == pred_label)  # Convert numpy bool to Python bool
        })
    
    with open(output_dir / 'test_predictions.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == '__main__':
    main()

