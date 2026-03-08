import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


def upscale(
    input_path: str,
    checkpoint_path: str,
    output_path: str = None,
    comparison_path: str = None,
    scale: int = 2,
    arch: str = 'espcn',
) -> None:
    """
    Upscale a single image using a trained ESPCN/SRCNN model.

    Parameters
    ----------
    input_path      : path to the low-resolution input image
    checkpoint_path : path to the .pth model checkpoint
    output_path     : where to save the SR image  (default: <stem>_sr.<ext>)
    comparison_path : where to save a side-by-side comparison  (default: <stem>_compare.<ext>)
    scale           : upscale factor (default 2)
    arch            : model architecture — 'espcn' or 'srcnn'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(arch, scale_factor=scale).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.eval()
    print(f"Loaded  : {checkpoint_path}  ({arch.upper()}, x{scale})")

    lr_img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = lr_img.size
    print(f"Input   : {input_path}  ({orig_w}×{orig_h})")

    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    if arch == 'srcnn':
        lr_tensor = transforms.functional.resize(
            lr_tensor.squeeze(0),
            [orig_h * scale, orig_w * scale],
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ).unsqueeze(0)

    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0.0, 1.0)

    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    sr_w, sr_h = sr_img.size

    input_stem = Path(input_path).stem
    input_ext  = Path(input_path).suffix or '.png'

    if output_path is None:
        output_path = str(Path(input_path).parent / f"{input_stem}_sr{input_ext}")
    sr_img.save(output_path)
    print(f"SR image: {output_path}  ({sr_w}×{sr_h})")

    bicubic_img = lr_img.resize((orig_w * scale, orig_h * scale), Image.BICUBIC)

    padding = 8
    comp_w = orig_w + sr_w + bicubic_img.width + padding * 4
    comp_h = max(orig_h, sr_h, bicubic_img.height) + padding * 2 + 24
    comparison = Image.new('RGB', (comp_w, comp_h), (30, 30, 30))

    x = padding
    comparison.paste(lr_img,       (x, padding)); x += orig_w + padding
    comparison.paste(bicubic_img,  (x, padding)); x += bicubic_img.width + padding
    comparison.paste(sr_img,       (x, padding))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        labels = [
            (padding,                                           comp_h - 20, 'LR Input'),
            (orig_w + padding * 2,                             comp_h - 20, 'Bicubic'),
            (orig_w + bicubic_img.width + padding * 3,         comp_h - 20, f'{arch.upper()} (ours)'),
        ]
        for lx, ly, text in labels:
            draw.text((lx, ly), text, fill=(220, 220, 220))
    except Exception:
        pass

    if comparison_path is None:
        comparison_path = str(Path(input_path).parent / f"{input_stem}_compare{input_ext}")
    comparison.save(comparison_path)
    print(f"Compare : {comparison_path}  (LR | Bicubic | {arch.upper()})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upscale a single image with a trained ESPCN/SRCNN model'
    )
    parser.add_argument('--input',       type=str, required=True,
                        help='Path to the low-resolution input image')
    parser.add_argument('--checkpoint',  type=str,
                        default='outputs/checkpoints/espcn_best.pth',
                        help='Path to .pth checkpoint')
    parser.add_argument('--output',      type=str, default=None,
                        help='Output path for the SR image  (default: <stem>_sr.<ext>)')
    parser.add_argument('--comparison',  type=str, default=None,
                        help='Output path for the side-by-side comparison image')
    parser.add_argument('--scale',       type=int, default=2)
    parser.add_argument('--arch',        type=str, default='espcn',
                        choices=['espcn', 'srcnn'])
    args = parser.parse_args()

    upscale(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        comparison_path=args.comparison,
        scale=args.scale,
        arch=args.arch,
    )
