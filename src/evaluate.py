import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from model import PixelShuffle


def _find_lr_file(hr_path, lr_dir, scale):
    """Resolve DIV2K-style LR filename (<stem>x<scale>.ext) or plain name."""
    suffixed = lr_dir / f"{hr_path.stem}x{scale}{hr_path.suffix}"
    if suffixed.exists():
        return suffixed
    plain = lr_dir / hr_path.name
    if plain.exists():
        return plain
    return None


def compute_psnr(hr, sr):
    """PSNR between HR and SR float arrays in [0, 1]."""
    return float(skimage_psnr(hr, sr, data_range=1.0))


def compute_ssim(hr, sr):
    """Mean SSIM between HR and SR float arrays in [0, 1] (RGB)."""
    return float(skimage_ssim(hr, sr, data_range=1.0, channel_axis=2))


def evaluate_model(
    checkpoint_path,
    lr_dir,
    hr_dir,
    arch='espcn',
    scale=2,
    output_dir=None,
):
    """
    Run inference with a trained model and report per-image PSNR / SSIM.

    Returns
    -------
    psnr_list, ssim_list : lists of per-image scores
    """
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'PixelShuffle': PixelShuffle},
    )
    print(f"Loaded checkpoint: {checkpoint_path}  (arch={arch}, scale=x{scale})")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    lr_path = Path(lr_dir)
    hr_path = Path(hr_dir)
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    hr_files = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in exts])

    psnr_list, ssim_list = [], []

    for hr_file in hr_files:
        lr_file = _find_lr_file(hr_file, lr_path, scale)
        if lr_file is None:
            print(f"  [skip] {hr_file.name}")
            continue

        lr_img = np.array(Image.open(lr_file).convert('RGB'), dtype=np.float32) / 255.0
        hr_img = np.array(Image.open(hr_file).convert('RGB'), dtype=np.float32) / 255.0

        lr_tensor = lr_img[np.newaxis]   # add batch dimension: (1, H, W, 3)

        if arch == 'srcnn':
            # SRCNN expects bicubic-upsampled LR at HR size
            lr_tensor = tf.image.resize(
                lr_tensor, [hr_img.shape[0], hr_img.shape[1]], method='bicubic'
            ).numpy()
            lr_tensor = np.clip(lr_tensor, 0.0, 1.0)

        sr_arr = model.predict(lr_tensor, verbose=0)[0]
        sr_arr = np.clip(sr_arr, 0.0, 1.0)

        min_h = min(sr_arr.shape[0], hr_img.shape[0])
        min_w = min(sr_arr.shape[1], hr_img.shape[1])
        sr_arr = sr_arr[:min_h, :min_w]
        hr_arr = hr_img[:min_h, :min_w]

        psnr = compute_psnr(hr_arr, sr_arr)
        ssim = compute_ssim(hr_arr, sr_arr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f"  {hr_file.name:<40}  PSNR={psnr:.2f} dB   SSIM={ssim:.4f}")

        if output_dir:
            sr_pil = Image.fromarray((sr_arr * 255).astype(np.uint8))
            sr_pil.save(Path(output_dir) / f"sr_{hr_file.name}")

    if psnr_list:
        print(f"\n=== Model Evaluation ({arch.upper()}, x{scale}) ===")
        print(f"Mean PSNR : {np.mean(psnr_list):.2f} dB")
        print(f"Mean SSIM : {np.mean(ssim_list):.4f}")
    else:
        print("No image pairs evaluated. Check --lr_dir and --hr_dir.")

    return psnr_list, ssim_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate super-resolution model (PSNR / SSIM)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .keras model file')
    parser.add_argument('--hr_dir',     type=str, default='Data/DIV2K_valid_HR')
    parser.add_argument('--lr_dir',     type=str, default='Data/DIV2K_valid_LR_bicubic/X2')
    parser.add_argument('--arch',       type=str, default='espcn', choices=['srcnn', 'espcn'])
    parser.add_argument('--scale',      type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='outputs/predictions')
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        arch=args.arch,
        scale=args.scale,
        output_dir=args.output_dir,
    )
