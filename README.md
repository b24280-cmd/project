# project
Off-Road Terrain Semantic Segmentation using DeepLabV3+ (ResNet-50 backbone) implemented with PyTorch and segmentation-models-pytorch. The project includes a full training pipeline with hybrid loss (Weighted Cross-Entropy + Dice), per-class IoU evaluation, and dataset distribution analysis.
# Off-Road Terrain Segmentation

A PyTorch implementation of semantic segmentation for off-road terrain using DeepLabV3+ with ResNet-50 backbone, featuring progressive training, advanced loss functions, and comprehensive evaluation tools.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Class Distribution Analysis](#class-distribution-analysis)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)

---

## 🎯 Overview

This project implements a state-of-the-art semantic segmentation system for off-road terrain classification. The system can identify 10 distinct terrain classes including vegetation, ground cover, obstacles, and sky.

### Segmentation Classes
1. **Background** (0)
2. **Trees** (1)
3. **Lush Bushes** (2)
4. **Dry Grass** (3)
5. **Dry Bushes** (4)
6. **Ground Clutter** (5)
7. **Logs** (6)
8. **Rocks** (7)
9. **Landscape** (8)
10. **Sky** (9)

---

## 📁 Project Structure

```
Krackhack/
│
├── 📁 ENV_SETUP/                          # Environment setup files
│
├── 📁 Offroad_Segmentation_testImages/    # Test dataset
│   ├── Color_Images/                      # RGB test images
│   └── Segmentation/                      # Ground truth masks
│
├── 📁 Offroad_Segmentation_Training_Dataset/  # Training dataset
│   ├── train/
│   │   ├── Color_Images/                  # Training RGB images
│   │   └── Segmentation/                  # Training masks
│   └── val/
│       ├── Color_Images/                  # Validation RGB images
│       └── Segmentation/                  # Validation masks
│
├── 📁 predictions/                         # Inference output directory
│
├── 📁 train_output/                        # Training artifacts
│   ├── best_model.pth                     # Best model checkpoint
│   └── final_model.pth                    # Final epoch checkpoint
│
├── 📄 analyze_class_distribution.py       # Dataset analysis tool (10 KB)
├── 📄 test_segmentation.py                # Inference script (9 KB)
└── 📄 train_segmentation.py               # Training script (19 KB)
```

---

## ✨ Features

### Training Features
- **Progressive Training**: Natural data (epochs 1-15) → Augmented data (epochs 16+)
- **Advanced Architecture**: DeepLabV3+ with ResNet-50 encoder + scSE attention
- **Modern Loss Function**: Focal Loss + Dice Loss combination
- **Smart Scheduling**: CosineAnnealingWarmRestarts with LR restart at epoch 10
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Data Augmentation**: 
  - HorizontalFlip
  - Rotation (±15°)
  - RandomResizedCrop (scale: 0.8-1.0)
  - ColorJitter
- **Real-time Monitoring**:
  - Per-class IoU tracking
  - Landscape confusion matrix
  - Best model checkpointing

### Evaluation Features
- **IoU Metrics**: Mean IoU and per-class IoU
- **Color Baseline**: RGB palette-based prediction baseline
- **Distribution Analysis**: Class distribution across splits
- **Fast Inference**: Batch processing with progress tracking

---

## 📦 Requirements

### Core Dependencies
```
torch >= 1.10.0
torchvision >= 0.11.0
segmentation-models-pytorch
albumentations
opencv-python
numpy
pillow
tqdm
```

### Installation

```bash
# Create conda environment
conda create -n terrain_seg python=3.9
conda activate terrain_seg

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install tqdm
```

---

## 📊 Dataset Structure

Each dataset split should follow this structure:

```
Dataset_Folder/
├── Color_Images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── Segmentation/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### Mask Encoding
Ground truth masks use raw pixel values that map to class IDs:
- `0` → Background
- `100` → Trees
- `200` → Lush Bushes
- `300` → Dry Grass
- `500` → Dry Bushes
- `550` → Ground Clutter
- `700` → Logs
- `800` → Rocks
- `7100` → Landscape
- `10000` → Sky

---

## 🚀 Usage

### Training

#### Basic Training
```bash
python train_segmentation.py
```

#### Custom Training
```bash
python train_segmentation.py \
    --epochs 50 \
    --batch_size 16 \
    --lr 2e-4 \
    --data_dir ./Offroad_Segmentation_Training_Dataset \
    --save_dir ./train_output \
    --img_height 384 \
    --img_width 672
```

#### Training Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 8 | Batch size for training |
| `--lr` | 1e-4 | Initial learning rate |
| `--data_dir` | `./Offroad_Segmentation_Training_Dataset` | Dataset root directory |
| `--save_dir` | `./train_output` | Output directory for checkpoints |
| `--img_height` | 384 | Image height for training |
| `--img_width` | 672 | Image width for training |

#### Training Output
```
Epoch 1/30:
  Train Loss: 0.4523, Train IoU: 0.6234
  Val Loss: 0.3891, Val IoU: 0.6789
  Per-Class IoU:
     0. Background           : 0.7234
     1. Trees                : 0.6891
     ...
  
  Landscape Confusion Report:
  Total Landscape pixels: 123,456
  Landscape confused with:
    Dry Grass           : 15.23% (18,821 pixels)
    Ground Clutter      : 10.45% (12,901 pixels)
  Correct predictions: 68.34% (84,356 pixels)
  
  ⭐ New best model saved! (Val IoU: 0.6789)
```

---

### Testing

#### Basic Inference
```bash
python test_segmentation.py
```

#### Custom Inference
```bash
python test_segmentation.py \
    --model_path ./train_output/best_model.pth \
    --data_dir ./Offroad_Segmentation_testImages \
    --batch_size 16 \
    --img_height 384 \
    --img_width 672
```

#### Inference Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./train_output/best_model.pth` | Path to trained model |
| `--data_dir` | `./Offroad_Segmentation_testImages` | Test dataset directory |
| `--batch_size` | 8 | Batch size for inference |
| `--img_height` | 384 | Image height (must match training) |
| `--img_width` | 672 | Image width (must match training) |

#### Inference Output
```
📊 Per-Class IoU:
  0. Background          : 0.7145
  1. Trees               : 0.6923
  2. Lush Bushes         : 0.5834
  ...
  9. Sky                 : 0.8123

Mean IoU: 0.6834
```

---

### Class Distribution Analysis

Analyze class distribution and compute color-palette baseline accuracy:

```bash
python analyze_class_distribution.py
```

#### Output
```
Dataset: Train
Total Images: 2857
Total Pixels: 734,803,968
  0. Background      :  12.34% (90,678,234 pixels)
  1. Trees           :  23.45% (172,345,678 pixels)
  ...

Color-Palette Baseline Accuracy: 23.45%

Per-Class Color-Palette Accuracy:
  0. Background         :  15.23%
  1. Trees              :  45.67%
  ...
```

This analysis provides:
- Class distribution across train/val/test splits
- Overall color-based prediction baseline
- Per-class color accuracy (simple RGB matching)

---

## 🏗️ Model Architecture

### DeepLabV3+ Configuration
- **Encoder**: ResNet-50 (ImageNet pretrained)
- **Decoder**: ASPP + scSE attention mechanism
- **Input Size**: 384 × 672 pixels
- **Output**: 10 segmentation classes

### Loss Function
**FocalDiceLoss** (Hybrid):
- **Focal Loss** (γ=2.0): Handles class imbalance
- **Dice Loss**: Improves boundary segmentation
- **Weights**: 50% Focal + 50% Dice

---

## 🎓 Training Strategy

### Progressive Training Pipeline
1. **Epochs 1-15**: Train on natural (unaugmented) data
   - Learn basic class representations
   - Stable gradient flow
   
2. **Epochs 16+**: Switch to augmented data
   - Heavy augmentation pipeline
   - Improve generalization
   - Learning rate warm restart

### Learning Rate Schedule
- **Optimizer**: AdamW (weight decay: 1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts
  - T_0 = 10 (restart every 10 epochs)
  - Aligns with augmentation switch
  - η_min = 1e-6

### Data Augmentation (Epochs 16+)
```python
- HorizontalFlip (p=0.5)
- Rotate ±15° (p=0.5)
- RandomResizedCrop (scale: 0.8-1.0, p=0.5)
- ColorJitter (brightness, contrast, saturation, hue)
```

---

## 📈 Results

### Expected Performance
With proper training, you should achieve:
- **Train IoU**: 0.70 - 0.75
- **Validation IoU**: 0.65 - 0.70
- **Test IoU**: 0.65 - 0.72

### Per-Class Performance (Example)
| Class | IoU | Notes |
|-------|-----|-------|
| Background | 0.71 | High accuracy |
| Trees | 0.68 | Consistent |
| Lush Bushes | 0.58 | Moderate |
| Dry Grass | 0.63 | Good |
| Dry Bushes | 0.61 | Good |
| Ground Clutter | 0.54 | Challenging |
| Logs | 0.49 | Difficult (small objects) |
| Rocks | 0.65 | Good |
| Landscape | 0.57 | Improved with confusion tracking |
| Sky | 0.81 | Excellent |

---

## 🔧 Advanced Features

### Landscape Confusion Tracking
The training script tracks which classes "Landscape" is most confused with:
```
Landscape Confusion Report:
  Landscape confused with:
    Dry Grass           : 15.23%
    Ground Clutter      : 10.45%
    Rocks               :  8.12%
  Correct predictions: 68.34%
```

### Color-Palette Baseline
The analysis script computes a naive baseline by predicting classes based on RGB color similarity to a predefined palette. This helps assess:
- Dataset difficulty
- Color consistency across classes
- Model improvement over simple heuristics

---

## 📝 Notes

### Model Checkpoints
- `best_model.pth`: Saved when validation IoU improves
- `final_model.pth`: Saved after last epoch

### Checkpoint Format
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'val_iou': float
}
```

### Memory Requirements
- **Training**: ~8GB GPU memory (batch_size=8)
- **Inference**: ~4GB GPU memory (batch_size=8)

### Speed Estimates
- **Training**: ~2-3 sec/batch (RTX 3090)
- **Inference**: ~0.5-1 sec/batch (RTX 3090)

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_segmentation.py --batch_size 4
```

**2. Pillow DLL Error (Windows)**
```bash
pip uninstall Pillow
pip install Pillow
```

**3. Model Architecture Mismatch**
Ensure your test script uses the same architecture:
```python
model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights=None,
    classes=10,
    decoder_attention_type='scse'  # Must match training!
)
```

---

## 📚 References

- **DeepLabV3+**: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **segmentation_models.pytorch**: [GitHub Repository](https://github.com/qubvel/segmentation_models.pytorch)

---

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

## 📄 License

This project is for educational and research purposes.

---

**Happy Segmenting! 🚗🌲**
