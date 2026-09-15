"""
demo_gan.py — Live demo script: generate a grid of brain MRI slices from
the trained GAN generator.

Usage:
    python -m src.demo_gan --checkpoint-path checkpoints/gan/generator.pth

Saves a tiled PNG grid of generated images to --output-path.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.gan import Generator, LATENT_DIM

NUM_SAMPLES = 16  # 4x4 grid


def get_device() -> torch.device:
    """Select mps if available, else cuda if available, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample grid from a trained GAN.")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/gan/generator.pth",
        help="Path to the trained Generator state_dict.",
    )
    parser.add_argument(
        "--output-path", type=str, default="outputs/gan/demo_samples.png",
        help="Path to save the generated sample grid PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model = Generator().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z = torch.randn(NUM_SAMPLES, LATENT_DIM, device=device)
        images = model(z).cpu().numpy()

    grid_size = int(NUM_SAMPLES ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()