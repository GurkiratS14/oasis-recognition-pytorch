"""
train_vae.py — Training script for the OASIS brain MRI VAE.

Usage:
    python -m src.train_vae --oasis-root /path/to/OASIS

Prints per-epoch train/validation loss to stdout in a simple
"epoch,train_loss,val_loss" format suitable for SLURM log parsing.
"""

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.dataset import OASISSliceDataset
from src.vae import VAE, vae_loss

BATCH_SIZE = 32


def get_device() -> torch.device:
    """Select mps if available, else cuda if available, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on OASIS brain MRI slices.")
    parser.add_argument(
        "--oasis-root", type=str, required=True,
        help="Path to the OASIS dataset root (contains keras_png_slices_{train,validate,test})",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument(
        "--beta-max", type=float, default=1.0,
        help="Final (max) weight on the KL divergence term, reached at --anneal-epochs.",
    )
    parser.add_argument(
        "--anneal-epochs", type=int, default=10,
        help="Number of epochs over which beta is linearly annealed from 0 to beta-max.",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/vae.pth",
        help="Path to save the trained model's state_dict.",
    )
    return parser.parse_args()


def kl_weight_schedule(epoch: int, beta_max: float, anneal_epochs: int) -> float:
    """Linearly ramp beta from 0.0 at epoch 1 to beta_max at epoch anneal_epochs.

    Holds steady at beta_max for all epochs after anneal_epochs.
    """
    if anneal_epochs <= 1:
        return beta_max
    progress = min(1.0, (epoch - 1) / (anneal_epochs - 1))
    return beta_max * progress


def run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    beta: float,
    device: torch.device,
    train: bool,
) -> float:
    """Run one pass over `loader`, optionally updating weights. Returns average loss."""
    model.train(train)
    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = OASISSliceDataset(args.oasis_root, split="train")
    val_dataset = OASISSliceDataset(args.oasis_root, split="validate")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        beta = kl_weight_schedule(epoch, args.beta_max, args.anneal_epochs)
        train_loss = run_epoch(model, train_loader, optimizer, beta, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, beta, device, train=False)
        print(f"epoch={epoch},beta={beta:.4f},train_loss={train_loss:.4f},val_loss={val_loss:.4f}")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
