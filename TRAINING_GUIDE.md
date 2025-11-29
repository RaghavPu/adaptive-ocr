# EfficientNet-B0 Classifier Training Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision timm pillow numpy tqdm scikit-learn matplotlib seaborn
```

### 2. Basic Training
```bash
python train_classifier.py
```

### 3. With Custom Options
```bash
python train_classifier.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 0.001 \
    --image_size 224 \
    --mixed_precision
```

## Training Configuration

### Default Settings (Optimized for 70-75% Accuracy)

- **Model**: EfficientNet-B0 (5.3M parameters)
- **Image Size**: 224x224
- **Batch Size**: 32
- **Epochs**: 40
- **Learning Rate**: 0.001 with CosineAnnealingWarmRestarts
- **Optimizer**: AdamW with weight decay 1e-4
- **Dropout**: 0.4
- **Data Split**: 80% train, 10% val, 10% test

### Data Augmentation (Enabled by Default)

- Random rotation (±5°)
- Color jitter (brightness, contrast, saturation)
- Random affine transformations
- Gaussian blur
- Random erasing (cutout)

All calibrated for document images!

### Class Imbalance Handling

- Weighted cross-entropy loss
- Weights computed inversely proportional to class frequency:
  - tiny: 1.6x
  - small: 1.0x (baseline)
  - base: 1.8x
  - large: 3.2x (highest weight for minority class)

## What to Expect

### Training Timeline (on GPU)

```
Epoch 1/40
----------------------------------------
Training: 100%|████████████| 34/34 [00:08<00:00, loss: 1.2450, acc: 45.23%]
Validating: 100%|██████████| 5/5 [00:01<00:00]

Train Loss: 1.2450 | Train Acc: 45.23%
Val Loss:   1.1234 | Val Acc:   48.15%
LR: 0.001000
✓ Saved best model (val_acc: 48.15%)

Epoch 5/40
----------------------------------------
Training: 100%|████████████| 34/34 [00:08<00:00, loss: 0.8234, acc: 62.15%]
Validating: 100%|██████████| 5/5 [00:01<00:00]

Train Loss: 0.8234 | Train Acc: 62.15%
Val Loss:   0.9012 | Val Acc:   58.52%
LR: 0.000707

Epoch 10/40
----------------------------------------
Training: 100%|████████████| 34/34 [00:08<00:00, loss: 0.6123, acc: 68.92%]
Validating: 100%|██████████| 5/5 [00:01<00:00]

Train Loss: 0.6123 | Train Acc: 68.92%
Val Loss:   0.7845 | Val Acc:   64.44%
LR: 0.000309
✓ Saved best model (val_acc: 64.44%)

...

Epoch 40/40
----------------------------------------
Training: 100%|████████████| 34/34 [00:08<00:00, loss: 0.4123, acc: 78.15%]
Validating: 100%|██████████| 5/5 [00:01<00:00]

Train Loss: 0.4123 | Train Acc: 78.15%
Val Loss:   0.7012 | Val Acc:   72.59%
LR: 0.000123
✓ Saved best model (val_acc: 72.59%)

============================================================
Training completed!
Best validation accuracy: 72.59%

Evaluating on test set...
Test Loss: 0.6891
Test Accuracy: 71.85%

Classification Report:
              precision    recall  f1-score   support

        tiny       0.68      0.72      0.70        33
       small       0.78      0.79      0.79        55
        base       0.70      0.67      0.68        30
       large       0.65      0.60      0.62        18

    accuracy                           0.72       136
   macro avg       0.70      0.70      0.70       136
weighted avg       0.72      0.72      0.72       136

Confusion matrix saved to classifier_output/confusion_matrix.png

✓ Results saved to classifier_output/

Files created:
  - best_model.pth (model checkpoint)
  - confusion_matrix.png
  - training_history.json
  - test_predictions.json
```

### Expected Performance

| Metric | Expected Value |
|--------|---------------|
| **Training Time (GPU)** | ~5-8 minutes |
| **Training Time (CPU)** | ~30-45 minutes |
| **Validation Accuracy** | 70-75% |
| **Test Accuracy** | 68-73% |
| **Per-class F1-scores** | 0.62-0.79 |

### Inference Speed

```
Single image:  3-5ms (GPU), 10-15ms (CPU)
Batch of 32:   50-80ms (GPU), 300-400ms (CPU)
```

## Command Line Options

```bash
# Basic options
--labels_file       Path to optimal_model_labels.json (default: optimal_model_labels.json)
--images_dir        Path to images directory (default: OmniDocBench/images)
--output_dir        Output directory (default: classifier_output)

# Training hyperparameters
--epochs            Number of epochs (default: 40)
--batch_size        Batch size (default: 32)
--lr                Learning rate (default: 0.001)
--weight_decay      Weight decay (default: 0.0001)
--dropout           Dropout rate (default: 0.4)

# Model options
--image_size        Input image size (default: 224, try 384 for +2-3% accuracy)

# Performance options
--mixed_precision   Enable mixed precision training (faster, less memory)
--num_workers       Data loading workers (default: 4)

# Ablation studies
--no_augment        Disable augmentation (expect -10% accuracy)
--seed              Random seed (default: 42)
```

## Typical Training Session

### 1. First Run (Baseline)
```bash
python train_classifier.py --epochs 40 --batch_size 32
```
Expected: 68-72% accuracy

### 2. With Mixed Precision (Faster)
```bash
python train_classifier.py --epochs 40 --batch_size 32 --mixed_precision
```
Expected: 68-72% accuracy, ~30% faster

### 3. Larger Images (Better Accuracy)
```bash
python train_classifier.py --epochs 40 --batch_size 16 --image_size 384
```
Expected: 71-75% accuracy (note: smaller batch due to memory)

### 4. Longer Training (Best Results)
```bash
python train_classifier.py --epochs 60 --batch_size 32 --mixed_precision
```
Expected: 72-76% accuracy

## Output Files

After training, you'll get:

1. **best_model.pth** - Model checkpoint with best validation accuracy
2. **confusion_matrix.png** - Visual confusion matrix
3. **training_history.json** - Loss and accuracy per epoch
4. **test_predictions.json** - Individual predictions on test set

## Next Steps

### Load Trained Model for Inference
```python
import torch
from train_classifier import EfficientNetClassifier

model = EfficientNetClassifier(num_classes=4)
checkpoint = torch.load('classifier_output/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict on new image
prediction = model(image_tensor)
```

### Improve Beyond 75%

1. **Train longer**: 60-80 epochs
2. **Ensemble**: Train 3 models, average predictions (+3-5%)
3. **Larger images**: 384x384 (+2-3%)
4. **Progressive training**: Start with 224, fine-tune at 384
5. **Test-time augmentation**: Average 5 augmented predictions (+2-3%)

## Troubleshooting

### Low Accuracy (<65%)
- Check if images are loading correctly
- Verify class distribution in splits
- Increase epochs to 60
- Try --image_size 384

### Overfitting (train acc >> val acc)
- Increase --dropout to 0.5-0.6
- Add more augmentation
- Reduce --lr to 0.0005
- Increase --weight_decay to 0.001

### Out of Memory
- Reduce --batch_size (try 16 or 8)
- Use --mixed_precision
- Reduce --image_size to 224
- Reduce --num_workers

### Slow Training
- Use --mixed_precision
- Increase --batch_size if memory allows
- Use GPU instead of CPU
- Reduce --image_size to 224

