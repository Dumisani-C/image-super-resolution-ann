import os
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from model import build_model, PixelShuffle


class SRDataset:
    """
    Super-resolution dataset.

    Supports two data sources (in priority order):
      1. Pre-extracted NumPy patch arrays  (Data/hr_patches.npy, Data/lr_patches.npy)
      2. Image directories                 (Data/hr/, Data/lr/)
    """

    def __init__(
        self,
        hr_patches_path=None,
        lr_patches_path=None,
        hr_dir=None,
        lr_dir=None,
        scale_factor=2,
        architecture='espcn',
    ):
        self.architecture = architecture
        self.scale_factor = scale_factor

        if hr_patches_path and lr_patches_path:
            self.hr_data = np.load(hr_patches_path).astype(np.float32)
            self.lr_data = np.load(lr_patches_path).astype(np.float32)
        elif hr_dir and lr_dir:
            self.hr_data, self.lr_data = self._load_from_dirs(hr_dir, lr_dir)
        else:
            raise ValueError("Provide either (hr_patches_path, lr_patches_path) or (hr_dir, lr_dir).")

    def _load_from_dirs(self, hr_dir, lr_dir):
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        hr_files = sorted([f for f in Path(hr_dir).iterdir() if f.suffix.lower() in exts])
        lr_files = sorted([f for f in Path(lr_dir).iterdir() if f.suffix.lower() in exts])
        assert len(hr_files) == len(lr_files), (
            f"HR and LR directories must have the same number of images "
            f"({len(hr_files)} vs {len(lr_files)})"
        )
        hr_data, lr_data = [], []
        for hf, lf in zip(hr_files, lr_files):
            hr_data.append(np.array(Image.open(hf).convert('RGB'), dtype=np.float32) / 255.0)
            lr_data.append(np.array(Image.open(lf).convert('RGB'), dtype=np.float32) / 255.0)
        return np.array(hr_data), np.array(lr_data)

    def get_tf_datasets(self, batch_size=16, val_split=0.1):
        """
        Split into train/val and return tf.data.Dataset objects.
        For SRCNN, the LR patches are bicubic-upsampled to match HR size.
        """
        n = len(self.hr_data)
        val_size = max(1, int(n * val_split))
        train_size = n - val_size

        lr_input = self.lr_data
        if self.architecture == 'srcnn':
            # SRCNN takes bicubic-upsampled LR (same size as HR)
            hr_h, hr_w = self.hr_data.shape[1], self.hr_data.shape[2]
            lr_input = tf.image.resize(
                self.lr_data, [hr_h, hr_w], method='bicubic'
            ).numpy().astype(np.float32)
            lr_input = np.clip(lr_input, 0.0, 1.0)

        print(f"Samples — train: {train_size}  val: {val_size}")

        train_ds = (
            tf.data.Dataset.from_tensor_slices((lr_input[:train_size], self.hr_data[:train_size]))
            .shuffle(buffer_size=1000, seed=42)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((lr_input[train_size:], self.hr_data[train_size:]))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_ds, val_ds


def train(args):
    patch_hr = Path('Data/hr_patches.npy')
    patch_lr = Path('Data/lr_patches.npy')

    if patch_hr.exists() and patch_lr.exists():
        print("Loading pre-extracted patch arrays...")
        dataset = SRDataset(
            hr_patches_path=str(patch_hr),
            lr_patches_path=str(patch_lr),
            scale_factor=args.scale,
            architecture=args.arch,
        )
    else:
        print("Loading images from directories...")
        dataset = SRDataset(
            hr_dir=args.hr_dir,
            lr_dir=args.lr_dir,
            scale_factor=args.scale,
            architecture=args.arch,
        )

    train_ds, val_ds = dataset.get_tf_datasets(batch_size=args.batch_size)

    # Build and compile the model — same pattern as the tutorial
    model = build_model(args.arch, scale_factor=args.scale)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.MeanAbsoluteError(),       # L1 loss — sharper than MSE
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse')],
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(args.checkpoint_dir, f'{args.arch}_best.keras')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    print(f"\nTraining {args.arch.upper()} for up to {args.epochs} epochs\n{'─'*60}")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    final_ckpt = os.path.join(args.checkpoint_dir, f'{args.arch}_final.keras')
    model.save(final_ckpt)
    print(f"\nDone. Final model saved: {final_ckpt}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train super-resolution ANN')
    parser.add_argument('--arch',           type=str,   default='espcn',
                        choices=['srcnn', 'espcn'])
    parser.add_argument('--hr_dir',         type=str,   default='Data/DIV2K_valid_HR')
    parser.add_argument('--lr_dir',         type=str,   default='Data/DIV2K_valid_LR_bicubic/X2')
    parser.add_argument('--scale',          type=int,   default=2)
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=16)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str,   default='outputs/checkpoints')
    args = parser.parse_args()

    train(args)
