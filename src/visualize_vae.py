"""
visualize_vae.py — Reconstruction and latent-manifold visualization for a
trained OASIS brain MRI VAE.

Usage:
    python -m src.visualize_vae --checkpoint-path checkpoints/vae.pth \
        --oasis-root /path/to/OASIS

Produces two PNGs in --output-dir:
    reconstructions.png — 8 real test images vs. their VAE reconstructions
    manifold.png         — a grid-size x grid-size tiling of the decoded 2D
                            latent manifold
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

from src.dataset import OASISSliceDataset
from src.vae import VAE

NUM_RECON_SAMPLES = 8
IMAGE_SIZE = 256


def get_device() -> torch.device:
    """Select mps if available, else cuda if available, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained OASIS VAE.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained VAE state_dict.")
    parser.add_argument("--oasis-root", type=str, required=True, help="Path to the OASIS dataset root.")
    parser.add_argument("--output-dir", type=str, default="outputs/vae", help="Directory to save output PNGs.")
    parser.add_argument("--grid-size", type=int, default=15, help="Size of the latent manifold grid (grid-size x grid-size).")
    return parser.parse_args()


def save_reconstructions(model: VAE, dataset: OASISSliceDataset, device: torch.device, output_path: Path) -> None:
    """Save a figure of 8 real images (top row) and their reconstructions (bottom row)."""
    loader = DataLoader(dataset, batch_size=NUM_RECON_SAMPLES, shuffle=False)
    originals = next(iter(loader)).to(device)

    with torch.no_grad():
        recons, _, _ = model(originals)

    originals = originals.cpu().numpy()
    recons = recons.cpu().numpy()

    fig, axes = plt.subplots(2, NUM_RECON_SAMPLES, figsize=(NUM_RECON_SAMPLES * 1.5, 3.5))
    for col in range(NUM_RECON_SAMPLES):
        axes[0, col].imshow(originals[col, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].axis("off")
        axes[1, col].imshow(recons[col, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].axis("off")

    axes[0, 0].set_title("Original", loc="left", fontsize=10)
    axes[1, 0].set_title("Reconstruction", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_manifold(model: VAE, device: torch.device, grid_size: int, output_path: Path) -> None:
    """Decode a grid of 2D latent points spanning the prior's probability mass and tile them."""
    probs = np.linspace(0.05, 0.95, grid_size)
    grid_coords = norm.ppf(probs)  # evenly-spaced-in-probability-mass values

    canvas = np.zeros((grid_size * IMAGE_SIZE, grid_size * IMAGE_SIZE), dtype=np.float32)

    with torch.no_grad():
        for row, y in enumerate(grid_coords):
            # Top row of the canvas corresponds to the largest latent value.
            z_row = np.array([[x, y] for x in grid_coords], dtype=np.float32)
            z_row_tensor = torch.from_numpy(z_row).to(device)
            decoded = model.decoder(z_row_tensor).cpu().numpy()  # (grid_size, 1, H, W)

            for col in range(grid_size):
                r0 = (grid_size - 1 - row) * IMAGE_SIZE
                c0 = col * IMAGE_SIZE
                canvas[r0:r0 + IMAGE_SIZE, c0:c0 + IMAGE_SIZE] = decoded[col, 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VAE().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    test_dataset = OASISSliceDataset(args.oasis_root, split="test")

    recon_path = output_dir / "reconstructions.png"
    manifold_path = output_dir / "manifold.png"

    save_reconstructions(model, test_dataset, device, recon_path)
    save_manifold(model, device, args.grid_size, manifold_path)

    print(f"Saved: {recon_path}")
    print(f"Saved: {manifold_path}")


if __name__ == "__main__":
    main()
