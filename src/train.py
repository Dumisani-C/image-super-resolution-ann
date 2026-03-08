import os
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

from model import build_model


class SRDataset(Dataset):
    """
    Super-resolution dataset.

    Supports two data sources (in priority order):
      1. Pre-extracted NumPy patch arrays  (data/hr_patches.npy, data/lr_patches.npy)
      2. Image directories                 (data/hr/, data/lr/)
    """

    def __init__(
        self,
        hr_patches_path: str = None,
        lr_patches_path: str = None,
        hr_dir: str = None,
        lr_dir: str = None,
        scale_factor: int = 4,
        architecture: str = 'srcnn',
    ):
        self.architecture = architecture
        self.scale_factor = scale_factor

        if hr_patches_path and lr_patches_path:
            self.hr_data = np.load(hr_patches_path)
            self.lr_data = np.load(lr_patches_path)
            self.from_patches = True
        elif hr_dir and lr_dir:
            exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            self.hr_files = sorted([f for f in Path(hr_dir).iterdir() if f.suffix.lower() in exts])
            self.lr_files = sorted([f for f in Path(lr_dir).iterdir() if f.suffix.lower() in exts])
            assert len(self.hr_files) == len(self.lr_files), (
                f"HR and LR directories must contain the same number of images "
                f"({len(self.hr_files)} vs {len(self.lr_files)})"
            )
            self.from_patches = False
        else:
            raise ValueError("Provide either (hr_patches_path, lr_patches_path) or (hr_dir, lr_dir).")

    def __len__(self) -> int:
        return len(self.hr_data) if self.from_patches else len(self.hr_files)

    def __getitem__(self, idx: int):
        if self.from_patches:
            hr = torch.from_numpy(self.hr_data[idx]).permute(2, 0, 1).float()
            lr = torch.from_numpy(self.lr_data[idx]).permute(2, 0, 1).float()
        else:
            hr = transforms.ToTensor()(Image.open(self.hr_files[idx]).convert('RGB'))
            lr = transforms.ToTensor()(Image.open(self.lr_files[idx]).convert('RGB'))

        if self.architecture == 'srcnn':
            lr = transforms.functional.resize(
                lr,
                [hr.shape[1], hr.shape[2]],
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )

        return lr, hr


def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

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

    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Samples — train: {train_size}  val: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = build_model(args.arch, scale_factor=args.scale).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nTraining {args.arch.upper()} for {args.epochs} epochs\n{'─'*60}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(lr_batch), hr_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                val_loss += criterion(model(lr_batch), hr_batch).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step()

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={time.time()-t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = Path(args.checkpoint_dir) / f'{args.arch}_best.pth'
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'val_loss': val_loss},
                str(ckpt),
            )
            print(f"  -> best checkpoint saved: {ckpt}")

    final_ckpt = Path(args.checkpoint_dir) / f'{args.arch}_final.pth'
    torch.save({'epoch': args.epochs, 'model_state': model.state_dict()}, str(final_ckpt))
    print(f"\nDone. Final checkpoint: {final_ckpt}")


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
    parser.add_argument('--lr_step',        type=int,   default=15,
                        help='Halve learning rate every N epochs')
    parser.add_argument('--checkpoint_dir', type=str,   default='outputs/checkpoints')
    args = parser.parse_args()

    train(args)
