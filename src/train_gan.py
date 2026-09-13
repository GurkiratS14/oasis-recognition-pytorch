"""
train_gan.py — DCGAN training script for synthesizing OASIS brain MRI slices.

Usage:
    python -m src.train_gan --oasis-root /path/to/OASIS

Prints per-epoch average D/G loss to stdout in a simple "epoch,d_loss,g_loss"
format suitable for SLURM log parsing, and periodically saves a PNG grid of
images decoded from a fixed noise vector so training progression is visible.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from src.dataset import OASISSliceDataset
from src.gan import Discriminator, Generator, LATENT_DIM, weights_init

NUM_SAMPLE_IMAGES = 16  # 4x4 grid
FIXED_NOISE_SEED = 42


def get_device() -> torch.device:
    """Select mps if available, else cuda if available, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DCGAN on OASIS brain MRI slices.")
    parser.add_argument(
        "--oasis-root", type=str, required=True,
        help="Path to the OASIS dataset root (contains keras_png_slices_{train,validate,test})",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (standard DCGAN momentum).")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/gan",
        help="Directory to save model checkpoints and sample image grids.",
    )
    parser.add_argument(
        "--sample-every", type=int, default=5,
        help="Save a fixed-noise sample grid every N epochs.",
    )
    return parser.parse_args()


def save_sample_grid(generator: Generator, fixed_noise: torch.Tensor, output_path: Path) -> None:
    """Decode fixed_noise through generator and save the results as a tiled PNG grid."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images = generator(fixed_noise).cpu().numpy()
    generator.train(was_training)

    grid_size = int(images.shape[0] ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = OASISSliceDataset(args.oasis_root, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    torch.manual_seed(FIXED_NOISE_SEED)
    fixed_noise = torch.randn(NUM_SAMPLE_IMAGES, LATENT_DIM, device=device)

    for epoch in range(1, args.epochs + 1):
        d_loss_total = 0.0
        g_loss_total = 0.0
        total_samples = 0

        for real_images in train_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size, 1), 1.0, device=device)
            fake_labels = torch.full((batch_size, 1), 0.0, device=device)

            # --- Train Discriminator ---
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise)

            optimizer_d.zero_grad()
            real_logits = discriminator(real_images)
            d_loss_real = criterion(real_logits, real_labels)
            fake_logits = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # --- Train Generator (non-saturating loss) ---
            optimizer_g.zero_grad()
            fake_logits_for_g = discriminator(fake_images)
            g_loss = criterion(fake_logits_for_g, real_labels)
            g_loss.backward()
            optimizer_g.step()

            d_loss_total += d_loss.item() * batch_size
            g_loss_total += g_loss.item() * batch_size
            total_samples += batch_size

        avg_d_loss = d_loss_total / total_samples
        avg_g_loss = g_loss_total / total_samples
        print(f"epoch={epoch},d_loss={avg_d_loss:.4f},g_loss={avg_g_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_path = checkpoint_dir / f"samples_epoch{epoch}.png"
            save_sample_grid(generator, fixed_noise, sample_path)

    torch.save(generator.state_dict(), checkpoint_dir / "generator.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator.pth")
    print(f"Saved checkpoints to {checkpoint_dir}")


if __name__ == "__main__":
    main()
