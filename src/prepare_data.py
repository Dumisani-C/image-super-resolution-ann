import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# DIV2K helper
# ---------------------------------------------------------------------------

def _find_lr_file(hr_path: Path, lr_dir: Path, scale: int) -> Path:
    """
    Locate the LR counterpart of an HR file.

    DIV2K names LR files as  <stem>x<scale>.png  (e.g. 0801x2.png).
    Falls back to the plain stem name if the suffixed version is absent.
    """
    suffixed = lr_dir / f"{hr_path.stem}x{scale}{hr_path.suffix}"
    if suffixed.exists():
        return suffixed
    plain = lr_dir / hr_path.name
    if plain.exists():
        return plain
    return None


# ---------------------------------------------------------------------------
# Step 1 – create LR images by downsampling (when no pre-made LR folder exists)
# ---------------------------------------------------------------------------

def create_lr_images(hr_dir: str, lr_dir: str, scale_factor: int = 2) -> None:
    """Downsample HR images to create LR counterparts via bicubic interpolation."""
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    lr_path.mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in exts])

    if not hr_images:
        print(f"No images found in {hr_dir}")
        return

    print(f"Creating LR images  (x{scale_factor} downscale)  from {len(hr_images)} HR images...")
    for img_path in hr_images:
        hr_img = Image.open(img_path).convert('RGB')
        w, h = hr_img.size
        lr_img = hr_img.resize((w // scale_factor, h // scale_factor), Image.BICUBIC)
        lr_img.save(lr_path / img_path.name)

    print(f"Saved {len(hr_images)} LR images to {lr_dir}")


# ---------------------------------------------------------------------------
# Step 2 – extract 64×64 (HR) / 32×32 (LR) training patches
# ---------------------------------------------------------------------------

def extract_patches(
    hr_dir: str,
    lr_dir: str = None,
    patch_size: int = 64,
    scale_factor: int = 2,
    stride: int = 64,
    max_patches: int = None,
    num_images: int = None,
) -> tuple:
    """
    Extract aligned HR / LR patch pairs.

    If *lr_dir* is None, LR patches are created on-the-fly by downsampling
    each HR patch (useful when no pre-made LR folder exists).

    Parameters
    ----------
    hr_dir       : folder of large HR images
    lr_dir       : folder of matching LR images (DIV2K x2 naming supported),
                   or None to generate LR by downsampling
    patch_size   : HR patch side length in pixels  (default 64)
    scale_factor : upscaling factor                (default 2)
    stride       : sliding-window stride in HR pixels (default 64 = non-overlapping)
    max_patches  : cap total patches extracted (None = no cap)
    num_images   : use only the first N HR images  (None = all)
    """
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir) if lr_dir else None

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in exts])

    if num_images:
        hr_images = hr_images[:num_images]

    lr_patch_size = patch_size // scale_factor
    hr_patches, lr_patches = [], []

    for img_path in hr_images:
        lr_file = _find_lr_file(img_path, lr_path, scale_factor) if lr_path else None

        hr_img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0

        if lr_file:
            lr_img = np.array(Image.open(lr_file).convert('RGB'), dtype=np.float32) / 255.0
        else:
            pil_hr = Image.fromarray((hr_img * 255).astype(np.uint8))
            w, h = pil_hr.size
            lr_img = np.array(
                pil_hr.resize((w // scale_factor, h // scale_factor), Image.BICUBIC),
                dtype=np.float32,
            ) / 255.0

        h_hr, w_hr = hr_img.shape[:2]
        for y in range(0, h_hr - patch_size + 1, stride):
            for x in range(0, w_hr - patch_size + 1, stride):
                hr_patches.append(hr_img[y : y + patch_size, x : x + patch_size])
                ly, lx = y // scale_factor, x // scale_factor
                lr_patches.append(lr_img[ly : ly + lr_patch_size, lx : lx + lr_patch_size])

                if max_patches and len(hr_patches) >= max_patches:
                    break
            if max_patches and len(hr_patches) >= max_patches:
                break
        if max_patches and len(hr_patches) >= max_patches:
            break

    hr_arr = np.array(hr_patches, dtype=np.float32)
    lr_arr = np.array(lr_patches, dtype=np.float32)
    print(f"Extracted {len(hr_arr)} patch pairs  |  HR: {hr_arr.shape}  LR: {lr_arr.shape}")
    return hr_arr, lr_arr


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare LR-HR pairs and extract 64×64 / 32×32 patches for 2× SR'
    )
    parser.add_argument('--hr_dir',      type=str, default='Data/DIV2K_valid_HR',
                        help='Folder of HR images')
    parser.add_argument('--lr_dir',      type=str, default='Data/DIV2K_valid_LR_bicubic/X2',
                        help='Folder of pre-made LR images (DIV2K x2 naming). '
                             'Omit to generate LR by downsampling.')
    parser.add_argument('--out_dir',     type=str, default='Data',
                        help='Directory to save hr_patches.npy / lr_patches.npy')
    parser.add_argument('--scale',       type=int, default=2,  help='Upscale factor')
    parser.add_argument('--patch_size',  type=int, default=64, help='HR patch side (pixels)')
    parser.add_argument('--stride',      type=int, default=64, help='Patch extraction stride')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Cap on total patches (None = no limit)')
    parser.add_argument('--num_images',  type=int, default=80,
                        help='Number of HR images to use for patch extraction')
    parser.add_argument('--create_lr',   action='store_true',
                        help='Also write bicubic-downsampled LR images to --lr_dir '
                             '(only useful when --lr_dir does not already exist)')
    args = parser.parse_args()

    if args.create_lr:
        create_lr_images(args.hr_dir, args.lr_dir, scale_factor=args.scale)

    hr_arr, lr_arr = extract_patches(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        patch_size=args.patch_size,
        scale_factor=args.scale,
        stride=args.stride,
        max_patches=args.max_patches,
        num_images=args.num_images,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'hr_patches.npy'), hr_arr)
    np.save(os.path.join(args.out_dir, 'lr_patches.npy'), lr_arr)
    print(f"Saved  {args.out_dir}/hr_patches.npy  and  {args.out_dir}/lr_patches.npy")
