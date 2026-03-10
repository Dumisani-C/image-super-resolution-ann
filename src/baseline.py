import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import compute_psnr, compute_ssim


def _find_lr_file(hr_path: Path, lr_dir: Path, scale: int) -> Path:
    """Resolve DIV2K-style LR filename (<stem>x<scale>.ext) or plain name."""
    suffixed = lr_dir / f"{hr_path.stem}x{scale}{hr_path.suffix}"
    if suffixed.exists():
        return suffixed
    plain = lr_dir / hr_path.name
    if plain.exists():
        return plain
    return None


def bicubic_upsample(lr_image: Image.Image, scale_factor: int) -> Image.Image:
    """Upsample a PIL Image using bicubic interpolation."""
    w, h = lr_image.size
    return lr_image.resize((w * scale_factor, h * scale_factor), Image.BICUBIC)


def bilinear_upsample(lr_image: Image.Image, scale_factor: int) -> Image.Image:
    """Upsample a PIL Image using bilinear interpolation."""
    w, h = lr_image.size
    return lr_image.resize((w * scale_factor, h * scale_factor), Image.BILINEAR)


def evaluate_baseline(
    lr_dir: str,
    hr_dir: str,
    scale_factor: int = 2,
    method: str = 'bicubic',
    output_dir: str = None,
) -> tuple:
    """
    Evaluate a classical upsampling baseline on matched LR/HR image pairs.

    Returns
    -------
    psnr_scores, ssim_scores : lists of per-image metric values
    """
    lr_path = Path(lr_dir)
    hr_path = Path(hr_dir)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    hr_files = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in exts])

    upsample_fn = bicubic_upsample if method == 'bicubic' else bilinear_upsample
    psnr_scores, ssim_scores = [], []

    for hr_file in hr_files:
        lr_file = _find_lr_file(hr_file, lr_path, scale_factor)
        if lr_file is None:
            print(f"  [skip] No matching LR file for {hr_file.name}")
            continue

        lr_img = Image.open(lr_file).convert('RGB')
        hr_img = Image.open(hr_file).convert('RGB')
        sr_img = upsample_fn(lr_img, scale_factor)

        min_w = min(sr_img.width,  hr_img.width)
        min_h = min(sr_img.height, hr_img.height)
        sr_img = sr_img.crop((0, 0, min_w, min_h))
        hr_img = hr_img.crop((0, 0, min_w, min_h))

        sr_arr = np.array(sr_img, dtype=np.float32) / 255.0
        hr_arr = np.array(hr_img, dtype=np.float32) / 255.0

        psnr = compute_psnr(hr_arr, sr_arr)
        ssim = compute_ssim(hr_arr, sr_arr)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        print(f"  {hr_file.name:<40}  PSNR={psnr:.2f} dB   SSIM={ssim:.4f}")

        if output_dir:
            sr_img.save(Path(output_dir) / hr_file.name)

    if psnr_scores:
        print(f"\n=== {method.capitalize()} Baseline (x{scale_factor}) ===")
        print(f"Mean PSNR : {np.mean(psnr_scores):.2f} dB")
        print(f"Mean SSIM : {np.mean(ssim_scores):.4f}")
    else:
        print("No image pairs found. Check --lr_dir and --hr_dir paths.")

    return psnr_scores, ssim_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bicubic / bilinear super-resolution baseline')
    parser.add_argument('--hr_dir',     type=str, default='Data/DIV2K_valid_HR')
    parser.add_argument('--lr_dir',     type=str, default='Data/DIV2K_valid_LR_bicubic/X2')
    parser.add_argument('--scale',      type=int, default=2)
    parser.add_argument('--method',     type=str, default='bicubic', choices=['bicubic', 'bilinear'])
    parser.add_argument('--output_dir', type=str, default='outputs/comparison_images')
    args = parser.parse_args()

    evaluate_baseline(args.lr_dir, args.hr_dir, args.scale, args.method, args.output_dir)
