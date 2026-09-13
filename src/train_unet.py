"""
train_unet.py — Training script for the OASIS brain MRI UNet segmentation model.

Usage:
    python -m src.train_unet --oasis-root /path/to/OASIS

Prints per-epoch train/validation loss and per-class DSC to stdout in a
simple "key=value,..." format suitable for SLURM log parsing. Class order
matches SEG_PIXEL_VALUES in src/dataset.py: background, CSF, gray matter,
white matter.
"""

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.dataset import OASISSegDataset
from src.unet import UNet
from src.utils import combined_loss, dice_score_per_class

CLASS_NAMES = ["bg", "csf", "gm", "wm"]


def get_device() -> torch.device:
    """Select mps if available, else cuda if available, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UNet on OASIS brain MRI slices.")
    parser.add_argument(
        "--oasis-root", type=str, required=True,
        help="Path to the OASIS dataset root (contains keras_png_slices_{train,validate,test})",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Weight on the Dice term in combined_loss.")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/unet.pth",
        help="Path to save the trained model's state_dict.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    dice_weight: float,
    device: torch.device,
) -> float:
    """Run one training pass over `loader` with gradient updates. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = combined_loss(logits, masks, dice_weight=dice_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def validate(
    model: UNet,
    loader: DataLoader,
    dice_weight: float,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Run validation with no gradient updates. Returns (average loss, per-class DSC tensor)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    dsc_sum = torch.zeros(len(CLASS_NAMES), device=device)
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = combined_loss(logits, masks, dice_weight=dice_weight)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            dsc_sum += dice_score_per_class(logits, masks, num_classes=len(CLASS_NAMES))
            num_batches += 1

    avg_loss = total_loss / total_samples
    avg_dsc = dsc_sum / num_batches
    return avg_loss, avg_dsc


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = OASISSegDataset(args.oasis_root, split="train")
    val_dataset = OASISSegDataset(args.oasis_root, split="validate")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.dice_weight, device)
        val_loss, val_dsc = validate(model, val_loader, args.dice_weight, device)

        dsc_str = ",".join(f"dsc_{name}={score:.4f}" for name, score in zip(CLASS_NAMES, val_dsc.tolist()))
        print(f"epoch={epoch},train_loss={train_loss:.4f},val_loss={val_loss:.4f},{dsc_str}")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
