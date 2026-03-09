import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import PixelShuffle


def upscale(
    input_path,
    checkpoint_path,
    output_path=None,
    comparison_path=None,
    scale=2,
    arch='espcn',
):
    """
    Upscale a single image using a trained ESPCN/SRCNN model.

    Parameters
    ----------
    input_path      : path to the low-resolution input image
    checkpoint_path : path to the .keras model file
    output_path     : where to save the SR image  (default: <stem>_sr.<ext>)
    comparison_path : where to save a side-by-side comparison  (default: <stem>_compare.<ext>)
    scale           : upscale factor (default 2)
    arch            : model architecture — 'espcn' or 'srcnn'
    """
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'PixelShuffle': PixelShuffle},
    )
    print(f"Loaded  : {checkpoint_path}  ({arch.upper()}, x{scale})")

    lr_img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = lr_img.size
    print(f"Input   : {input_path}  ({orig_w}×{orig_h})")

    lr_arr = np.array(lr_img, dtype=np.float32) / 255.0
    lr_tensor = lr_arr[np.newaxis]   # (1, H, W, 3)

    if arch == 'srcnn':
        lr_tensor = tf.image.resize(
            lr_tensor, [orig_h * scale, orig_w * scale], method='bicubic'
        ).numpy()
        lr_tensor = np.clip(lr_tensor, 0.0, 1.0)

    sr_arr = model.predict(lr_tensor, verbose=0)[0]
    sr_arr = np.clip(sr_arr, 0.0, 1.0)
    sr_img = Image.fromarray((sr_arr * 255).astype(np.uint8))
    sr_w, sr_h = sr_img.size

    input_stem = Path(input_path).stem
    input_ext  = Path(input_path).suffix or '.png'

    if output_path is None:
        output_path = str(Path(input_path).parent / f"{input_stem}_sr{input_ext}")
    sr_img.save(output_path)
    print(f"SR image: {output_path}  ({sr_w}×{sr_h})")

    # Build side-by-side comparison: LR Input | Bicubic | ESPCN
    bicubic_img = lr_img.resize((orig_w * scale, orig_h * scale), Image.BICUBIC)

    padding = 8
    comp_w = orig_w + sr_w + bicubic_img.width + padding * 4
    comp_h = max(orig_h, sr_h, bicubic_img.height) + padding * 2 + 24
    comparison = Image.new('RGB', (comp_w, comp_h), (30, 30, 30))

    x = padding
    comparison.paste(lr_img,      (x, padding)); x += orig_w + padding
    comparison.paste(bicubic_img, (x, padding)); x += bicubic_img.width + padding
    comparison.paste(sr_img,      (x, padding))

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison)
        labels = [
            (padding,                                    comp_h - 20, 'LR Input'),
            (orig_w + padding * 2,                       comp_h - 20, 'Bicubic'),
            (orig_w + bicubic_img.width + padding * 3,   comp_h - 20, f'{arch.upper()} (ours)'),
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
    parser.add_argument('--input',      type=str, required=True,
                        help='Path to the low-resolution input image')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/checkpoints/espcn_best.keras',
                        help='Path to .keras model file')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output path for the SR image  (default: <stem>_sr.<ext>)')
    parser.add_argument('--comparison', type=str, default=None,
                        help='Output path for the side-by-side comparison image')
    parser.add_argument('--scale',      type=int, default=2)
    parser.add_argument('--arch',       type=str, default='espcn',
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
