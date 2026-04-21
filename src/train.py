import os
import sys
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, PixelShuffle


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_pair(lr, hr):
    """
    Apply identical random spatial augmentations to an LR/HR pair.

    Uses tf.cond with stateful scalar ops — fixed graph structure so TF
    does not accumulate new nodes across batches (prevents progressive
    slowdown from graph growth).
    """
    do_flip_lr = tf.random.uniform(()) > 0.5
    do_flip_ud = tf.random.uniform(()) > 0.5
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)

    lr = tf.cond(do_flip_lr, lambda: tf.image.flip_left_right(lr), lambda: lr)
    hr = tf.cond(do_flip_lr, lambda: tf.image.flip_left_right(hr), lambda: hr)

    lr = tf.cond(do_flip_ud, lambda: tf.image.flip_up_down(lr), lambda: lr)
    hr = tf.cond(do_flip_ud, lambda: tf.image.flip_up_down(hr), lambda: hr)

    lr = tf.image.rot90(lr, k)
    hr = tf.image.rot90(hr, k)

    return lr, hr


# ---------------------------------------------------------------------------
# Perceptual loss wrapper
# ---------------------------------------------------------------------------

class SRPerceptualModel(tf.keras.Model):
    """
    Wraps an SR model and adds VGG16 perceptual loss during training.

    The inner SR model is saved to checkpoints (not this wrapper), so
    upscale.py and evaluate.py remain fully compatible.

    Loss = L1_pixel + perceptual_weight * L2_vgg_features
    """

    def __init__(self, sr_model, perceptual_weight=0.1):
        super().__init__()
        self.sr_model = sr_model
        self.perceptual_weight = perceptual_weight
        self._mse_metric = tf.keras.metrics.MeanSquaredError(name='mse')

        # VGG feature extractor — block2_conv2 captures low/mid-level textures
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        self.vgg = tf.keras.Model(
            vgg.input,
            vgg.get_layer('block2_conv2').output,
            name='vgg_features',
        )

    @property
    def metrics(self):
        return [self._mse_metric]

    def call(self, inputs, training=False):
        return self.sr_model(inputs, training=training)

    def train_step(self, data):
        lr, hr = data
        preprocess = tf.keras.applications.vgg16.preprocess_input

        with tf.GradientTape() as tape:
            sr = self.sr_model(lr, training=True)
            pixel_loss = tf.reduce_mean(tf.abs(hr - sr))
            sr_feat = self.vgg(preprocess(sr * 255.0), training=False)
            hr_feat = self.vgg(preprocess(hr * 255.0), training=False)
            # Normalise by mean feature magnitude so perceptual loss stays
            # proportional to pixel loss regardless of layer activation scale.
            scale = tf.stop_gradient(tf.reduce_mean(tf.abs(hr_feat)) + 1e-8)
            perc_loss = tf.reduce_mean(tf.square(hr_feat - sr_feat)) / scale
            total_loss = pixel_loss + self.perceptual_weight * perc_loss

        grads = tape.gradient(total_loss, self.sr_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.sr_model.trainable_variables))
        self._mse_metric.update_state(hr, sr)

        return {
            'loss': total_loss,
            'pixel_loss': pixel_loss,
            'perceptual_loss': perc_loss,
            'mse': self._mse_metric.result(),
        }

    def test_step(self, data):
        lr, hr = data
        sr = self.sr_model(lr, training=False)
        pixel_loss = tf.reduce_mean(tf.abs(hr - sr))
        self._mse_metric.update_state(hr, sr)
        return {
            'loss': pixel_loss,
            'mse': self._mse_metric.result(),
        }

    def save(self, filepath, **kwargs):
        """Save only the inner SR model for compatibility with upscale.py / evaluate.py."""
        self.sr_model.save(filepath, **kwargs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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

    def get_tf_datasets(self, batch_size=16, val_split=0.1, augment=True):
        """
        Split into train/val and return tf.data.Dataset objects.
        For SRCNN, the LR patches are bicubic-upsampled to match HR size.
        Augmentation (random flips + rot90) is applied to the training set only.
        """
        n = len(self.hr_data)
        val_size = max(1, int(n * val_split))
        train_size = n - val_size

        lr_input = self.lr_data
        if self.architecture == 'srcnn':
            hr_h, hr_w = self.hr_data.shape[1], self.hr_data.shape[2]
            lr_input = tf.image.resize(
                self.lr_data, [hr_h, hr_w], method='bicubic'
            ).numpy().astype(np.float32)
            lr_input = np.clip(lr_input, 0.0, 1.0)

        print(f"Samples — train: {train_size}  val: {val_size}  augment: {augment}")

        train_ds = tf.data.Dataset.from_tensor_slices(
            (lr_input[:train_size], self.hr_data[:train_size])
        ).shuffle(buffer_size=1000, seed=42)

        if augment:
            train_ds = train_ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = (
            tf.data.Dataset.from_tensor_slices((lr_input[train_size:], self.hr_data[train_size:]))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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

    augment = not getattr(args, 'no_augment', False)
    train_ds, val_ds = dataset.get_tf_datasets(batch_size=args.batch_size, augment=augment)

    # Build or resume the inner SR model
    resume_path = getattr(args, 'resume', None)
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        sr_model = tf.keras.models.load_model(
            resume_path,
            custom_objects={'PixelShuffle': PixelShuffle},
        )
    else:
        sr_model = build_model(args.arch, scale_factor=args.scale)
    sr_model.summary()

    perceptual_weight = getattr(args, 'perceptual_weight', 0.1)

    if perceptual_weight > 0:
        print(f"Using perceptual loss  (weight={perceptual_weight})")
        model = SRPerceptualModel(sr_model, perceptual_weight=perceptual_weight)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))
    else:
        model = sr_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.MeanAbsoluteError(),
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

    print(f"\nTraining {args.arch.upper()} for up to {args.epochs} epochs\n{'-'*60}")
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
    parser.add_argument('--arch',               type=str,   default='espcn',
                        choices=['srcnn', 'espcn'])
    parser.add_argument('--hr_dir',             type=str,   default='Data/DIV2K_valid_HR')
    parser.add_argument('--lr_dir',             type=str,   default='Data/DIV2K_valid_LR_bicubic/X2')
    parser.add_argument('--scale',              type=int,   default=2)
    parser.add_argument('--epochs',             type=int,   default=100)
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--lr',                 type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir',     type=str,   default='outputs/checkpoints')
    parser.add_argument('--resume',             type=str,   default=None,
                        help='Path to a .keras checkpoint to resume training from')
    parser.add_argument('--perceptual_weight',  type=float, default=0.1,
                        help='Weight for VGG perceptual loss (0 = pixel loss only)')
    parser.add_argument('--no_augment',         action='store_true',
                        help='Disable random flip/rotation augmentation')
    args = parser.parse_args()

    train(args)
