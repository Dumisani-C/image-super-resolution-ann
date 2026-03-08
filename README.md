# Image Super-Resolution with ANNs

A PyTorch implementation of CNN-based image super-resolution (2× upscaling) using the **ESPCN** architecture trained on the DIV2K dataset, with classical bicubic/bilinear baselines for comparison.

---

## Problem Statement

We want to take low-quality (low-resolution) photo images and upscale them **2×** while preserving as much visual detail as possible. The approach is supervised learning: high-resolution (HR) images are collected, downscaled by 2× to create their low-resolution (LR) counterparts, and an ANN is trained to map LR → HR.

**Motivation:** Image scaling is an important challenge in areas such as game development, where small textures or sprites often need to be enlarged. Traditional interpolation methods frequently introduce blurring and artifacts. In this project, we investigate how Artificial Neural Networks (ANNs) can be used to perform image upscaling while preserving sharp edges and fine details.

---

## Why an ANN?

Classical methods like bicubic interpolation are fixed mathematical filters — they produce smooth but blurry results because they have no knowledge of natural image structure.

A Convolutional Neural Network (CNN) is well-suited for this problem because:

- **Local receptive fields** — convolutional layers learn spatial filters that detect edges, textures, and patterns at multiple scales, exactly the information needed to reconstruct missing high-frequency detail.
- **Learned priors** — by training on thousands of HR/LR patch pairs, the network learns what sharp edges and textures *should* look like, which a fixed filter can never do.
- **End-to-end optimisation** — the ANN minimises a pixel-level reconstruction loss (L1/MAE) directly, so every layer is jointly trained to serve the upscaling objective.
- **Efficiency** — ESPCN performs all feature extraction at the cheap LR resolution and only expands to HR at the final layer, making it fast at inference time.

---

## Overall Design

### Data
- **Source:** [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 900 professionally photographed 2K-resolution images, a standard super-resolution benchmark.
- **LR creation:** Each HR image is bicubic-downsampled by 2× to create its LR counterpart (`prepare_data.py`). DIV2K also ships pre-made LR images (`DIV2K_train_LR_bicubic/X2/`).

### Preprocessing
1. Crop the large HR images into **64×64 pixel patches** (stride 64 — non-overlapping).
2. Crop the matching LR images into **32×32 pixel patches** at the corresponding location.
3. Normalise all pixel values to **[0, 1]**.
4. Save as `.npy` arrays for fast loading during training.

### Network Architecture — ESPCN

```
Input  (32×32×3)  — LR patch, 3,072 values
  └─ Conv2d(3 → 64,  5×5) + ReLU   ← feature extraction
  └─ Conv2d(64 → 32, 3×3) + ReLU   ← non-linear mapping
  └─ Conv2d(32 → 12, 3×3)           ← 12 = 3 channels × 2² (scale²)
  └─ PixelShuffle(2)                 ← rearrange to HR spatial size
Output (64×64×3)  — SR patch, 12,288 values
```

| Layer | Input shape | Output shape | Parameters |
|-------|-------------|--------------|------------|
| Conv1 | 3×32×32     | 64×32×32     | 4,864      |
| Conv2 | 64×32×32    | 32×32×32     | 18,464     |
| Conv3 | 32×32×32    | 12×32×32     | 3,468      |
| PixelShuffle | 12×32×32 | 3×64×64  | —          |

**Total: ~26,796 trainable parameters**

### Training
- **Loss:** L1 (MAE) — produces sharper outputs than MSE
- **Optimizer:** Adam, learning rate 1e-4, halved every 15 epochs
- **Epochs:** 50
- **Batch size:** 16
- **Split:** 90% train / 10% validation

### Evaluation
The trained model is compared against bicubic and bilinear baselines using:
- **PSNR** (Peak Signal-to-Noise Ratio, dB) — higher is better
- **SSIM** (Structural Similarity Index, 0–1) — higher is better

### Results — DIV2K Validation Set (100 images, 2× upscale)

| Method       | Mean PSNR (dB) ↑ | Mean SSIM ↑ |
|--------------|-----------------|------------|
| Bilinear     | 30.40           | 0.8938     |
| Bicubic      | 31.04           | 0.9015     |
| **ESPCN (50 epochs)** | **32.39** | **0.9232** |

ESPCN achieves **+1.35 dB PSNR** and **+0.022 SSIM** over the bicubic baseline after 50 epochs of L1 training on 53,692 patch pairs.

---

## Project Structure

```
image-super-resolution-ann/
├── Data/
│   ├── DIV2K_valid_HR/              # 100 HR images (source)
│   ├── DIV2K_valid_LR_bicubic/X2/   # 100 matching LR images
│   ├── DIV2K_train_LR_bicubic/X2/   # 800 training LR images
│   ├── hr_patches.npy               # extracted HR patches (generated)
│   └── lr_patches.npy               # extracted LR patches (generated)
│
├── src/
│   ├── prepare_data.py        # Extract 64×64 HR / 32×32 LR patch pairs
│   ├── baseline.py            # Bicubic / bilinear baseline + PSNR/SSIM
│   ├── model.py               # SRCNN and ESPCN architectures
│   ├── train.py               # Training loop (L1 loss, Adam, checkpointing)
│   └── evaluate.py            # PSNR / SSIM evaluation on test set
│
├── notebooks/
│   └── tutorial_experiments.ipynb   # End-to-end walkthrough + comparison story
│
├── outputs/
│   ├── checkpoints/           # Saved model weights (.pth)
│   ├── predictions/           # SR images from model inference
│   └── comparison_images/     # Baseline output images
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Demo — Upscale Any Image

Once the model is trained, upscale **any single image** with one command:

```bash
python src/upscale.py --input path/to/your_image.png
```

This produces two files next to the input:
- `your_image_sr.png` — the 2× super-resolved output
- `your_image_compare.png` — a side-by-side: **LR input | Bicubic | ESPCN**

**Example using a DIV2K image:**
```bash
python src/upscale.py --input Data/DIV2K_valid_LR_bicubic/X2/0801x2.png
# Input  : 1020×678
# Output : 2040×1356
```

| Argument        | Default                              | Description                       |
|-----------------|--------------------------------------|-----------------------------------|
| `--input`       | *(required)*                         | Path to the LR image to upscale   |
| `--checkpoint`  | outputs/checkpoints/espcn_best.pth   | Trained model weights             |
| `--output`      | `<stem>_sr.<ext>`                    | Where to save the SR image        |
| `--comparison`  | `<stem>_compare.<ext>`               | Where to save the side-by-side    |
| `--scale`       | 2                                    | Upscale factor                    |
| `--arch`        | espcn                                | `espcn` or `srcnn`                |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract training patches

```bash
python src/prepare_data.py
```

This reads from `Data/DIV2K_valid_HR/` and `Data/DIV2K_valid_LR_bicubic/X2/`, extracts 64×64 / 32×32 patch pairs, and saves them to `Data/hr_patches.npy` and `Data/lr_patches.npy`.

| Argument        | Default                           | Description                          |
|-----------------|-----------------------------------|--------------------------------------|
| `--hr_dir`      | Data/DIV2K_valid_HR               | HR image directory                   |
| `--lr_dir`      | Data/DIV2K_valid_LR_bicubic/X2    | LR image directory                   |
| `--scale`       | 2                                 | Upscale factor                       |
| `--patch_size`  | 64                                | HR patch side (pixels)               |
| `--stride`      | 64                                | Patch extraction stride              |
| `--num_images`  | 80                                | Number of HR images to use           |
| `--max_patches` | None                              | Cap total patches (None = no limit)  |

### 3. Run the baseline

```bash
python src/baseline.py --method bicubic
```

### 4. Train ESPCN

```bash
python src/train.py
```

| Argument           | Default                        | Description                  |
|--------------------|--------------------------------|------------------------------|
| `--arch`           | espcn                          | `espcn` or `srcnn`           |
| `--scale`          | 2                              | Upscale factor               |
| `--epochs`         | 50                             | Training epochs              |
| `--batch_size`     | 16                             | Batch size                   |
| `--lr`             | 1e-4                           | Initial learning rate        |
| `--lr_step`        | 15                             | Halve LR every N epochs      |
| `--checkpoint_dir` | outputs/checkpoints            | Where to save `.pth` files   |

### 5. Evaluate

```bash
python src/evaluate.py --checkpoint outputs/checkpoints/espcn_best.pth
```

---

## Models

### ESPCN — Efficient Sub-Pixel CNN *(Shi et al., 2016)* — **Primary**

Operates at native LR resolution; PixelShuffle handles the upsampling at the final layer only.

```
Input (32×32 LR)  →  Conv(5×5)+ReLU  →  Conv(3×3)+ReLU  →  Conv(3×3)  →  PixelShuffle(2)  →  Output (64×64 SR)
```

### SRCNN — Super-Resolution CNN *(Dong et al., 2014)* — Alternative

Takes a bicubic-upsampled LR image and learns to refine it.

```
Input (64×64 bicubic-upsampled LR)  →  Conv(9×9)+ReLU  →  Conv(1×1)+ReLU  →  Conv(5×5)  →  Output (64×64 SR)
```

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
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
