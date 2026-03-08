# Image Super-Resolution with ANNs

A clean PyTorch implementation of CNN-based image super-resolution, featuring SRCNN and ESPCN architectures alongside classical bicubic/bilinear baselines.

---

## Project Structure

```
image-super-resolution-ann/
├── data/
│   ├── hr/                    # High-resolution images (training source)
│   └── lr/                    # Low-resolution images (auto-generated)
│
├── src/
│   ├── prepare_data.py        # Create LR-HR pairs & extract patches
│   ├── baseline.py            # Bicubic / bilinear baseline + metrics
│   ├── model.py               # SRCNN and ESPCN architectures
│   ├── train.py               # Training loop with checkpointing
│   └── evaluate.py            # PSNR / SSIM evaluation
│
├── notebooks/
│   └── tutorial_experiments.ipynb
│
├── outputs/
│   ├── predictions/           # SR images from model inference
│   └── comparison_images/     # Baseline output images
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add HR images

Place your high-resolution training images in `data/hr/`.  
Any standard format is supported (`.png`, `.jpg`, `.bmp`, etc.).

> **Tip:** The [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset is commonly used for super-resolution benchmarks.

### 3. Prepare LR images and patches

```bash
python src/prepare_data.py --hr_dir data/hr --lr_dir data/lr --scale 4
```

This creates matching low-resolution images in `data/lr/` and saves training patch arrays (`data/hr_patches.npy`, `data/lr_patches.npy`).

| Argument       | Default | Description                    |
|----------------|---------|--------------------------------|
| `--hr_dir`     | data/hr | Source HR image directory      |
| `--lr_dir`     | data/lr | Output LR image directory      |
| `--scale`      | 4       | Downscale factor               |
| `--patch_size` | 96      | HR patch size (pixels)         |
| `--stride`     | 48      | Stride for patch extraction    |
| `--no_patches` | —       | Skip patch extraction          |

### 4. Run the baseline

```bash
python src/baseline.py --method bicubic --scale 4
```

### 5. Train the model

```bash
# SRCNN (default)
python src/train.py --arch srcnn --epochs 100 --batch_size 16

# ESPCN
python src/train.py --arch espcn --epochs 100 --batch_size 16
```

| Argument           | Default             | Description                     |
|--------------------|---------------------|---------------------------------|
| `--arch`           | srcnn               | `srcnn` or `espcn`              |
| `--epochs`         | 100                 | Number of training epochs       |
| `--batch_size`     | 16                  | Batch size                      |
| `--lr`             | 1e-4                | Initial learning rate           |
| `--lr_step`        | 30                  | Halve LR every N epochs         |
| `--checkpoint_dir` | outputs/checkpoints | Where to save `.pth` files      |

### 6. Evaluate

```bash
python src/evaluate.py --checkpoint outputs/checkpoints/srcnn_best.pth --arch srcnn
```

---

## Models

### SRCNN — Super-Resolution CNN *(Dong et al., 2014)*

```
Input (bicubic-upsampled LR)
  └─ Conv2d(3→64,  9×9) + ReLU   # feature extraction
  └─ Conv2d(64→32, 1×1) + ReLU   # non-linear mapping
  └─ Conv2d(32→3,  5×5)           # reconstruction
Output (SR image)
```

> Takes a bicubic-upsampled LR image as input and refines it.

### ESPCN — Efficient Sub-Pixel CNN *(Shi et al., 2016)*

```
Input (native LR resolution)
  └─ Conv2d(3→64,  5×5) + ReLU
  └─ Conv2d(64→32, 3×3) + ReLU
  └─ Conv2d(32→3·r², 3×3)
  └─ PixelShuffle(r)               # sub-pixel upsampling
Output (HR image, r× larger)
```

> Operates at LR resolution — more computationally efficient than SRCNN.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) — higher is better |
| **SSIM** | Structural Similarity Index — ranges [0, 1], higher is better |

---

## References

- Dong et al. (2014). [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)
- Shi et al. (2016). [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN](https://arxiv.org/abs/1609.05158)
