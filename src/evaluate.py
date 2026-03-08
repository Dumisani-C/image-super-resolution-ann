import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from torchvision import transforms

from model import build_model


def _find_lr_file(hr_path: Path, lr_dir: Path, scale: int) -> Path:
    """Resolve DIV2K-style LR filename (<stem>x<scale>.ext) or plain name."""
    suffixed = lr_dir / f"{hr_path.stem}x{scale}{hr_path.suffix}"
    if suffixed.exists():
        return suffixed
    plain = lr_dir / hr_path.name
    if plain.exists():
        return plain
    return None


def compute_psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    """PSNR between HR and SR float arrays in [0, 1]."""
    return float(skimage_psnr(hr, sr, data_range=1.0))


def compute_ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    """Mean SSIM between HR and SR float arrays in [0, 1] (RGB)."""
    return float(skimage_ssim(hr, sr, data_range=1.0, channel_axis=2))


def evaluate_model(
    checkpoint_path: str,
    lr_dir: str,
    hr_dir: str,
    arch: str = 'srcnn',
    scale: int = 4,
    output_dir: str = None,
    device: torch.device = None,
) -> tuple:
    """
    Run inference with a trained model and report per-image PSNR / SSIM.

    Returns
    -------
    psnr_list, ssim_list : lists of per-image scores
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(arch, scale_factor=scale).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}  (arch={arch}, scale=x{scale})")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    lr_path = Path(lr_dir)
    hr_path = Path(hr_dir)
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    hr_files = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in exts])

    to_tensor = transforms.ToTensor()
    psnr_list, ssim_list = [], []

    for hr_file in hr_files:
        lr_file = _find_lr_file(hr_file, lr_path, scale)
        if lr_file is None:
            print(f"  [skip] {hr_file.name}")
            continue

        lr_img = Image.open(lr_file).convert('RGB')
        hr_img = Image.open(hr_file).convert('RGB')

        lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

        if arch == 'srcnn':
            lr_tensor = transforms.functional.resize(
                lr_tensor.squeeze(0),
                [hr_img.height, hr_img.width],
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ).unsqueeze(0)

        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(0.0, 1.0)

        sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
        sr_arr = np.array(sr_img,  dtype=np.float32) / 255.0
        hr_arr = np.array(hr_img,  dtype=np.float32) / 255.0

        min_h = min(sr_arr.shape[0], hr_arr.shape[0])
        min_w = min(sr_arr.shape[1], hr_arr.shape[1])
        sr_arr = sr_arr[:min_h, :min_w]
        hr_arr = hr_arr[:min_h, :min_w]

        psnr = compute_psnr(hr_arr, sr_arr)
        ssim = compute_ssim(hr_arr, sr_arr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f"  {hr_file.name:<40}  PSNR={psnr:.2f} dB   SSIM={ssim:.4f}")

        if output_dir:
            sr_img.save(Path(output_dir) / f"sr_{hr_file.name}")

    if psnr_list:
        print(f"\n=== Model Evaluation ({arch.upper()}, x{scale}) ===")
        print(f"Mean PSNR : {np.mean(psnr_list):.2f} dB")
        print(f"Mean SSIM : {np.mean(ssim_list):.4f}")
    else:
        print("No image pairs evaluated. Check --lr_dir and --hr_dir.")

    return psnr_list, ssim_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate super-resolution model (PSNR / SSIM)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
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
