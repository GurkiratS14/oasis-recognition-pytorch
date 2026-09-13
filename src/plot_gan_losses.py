"""
plot_gan_losses.py — Plot D/G loss curves from a train_gan.py log file.

Usage:
    python -m src.plot_gan_losses path/to/train.log --output-path outputs/gan/losses.png

Parses lines of the form "epoch=N,d_loss=X,g_loss=Y" (as printed by
src/train_gan.py) and plots both loss curves over epoch number.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_LINE_PATTERN = re.compile(r"epoch=(\d+),d_loss=([-\d.eE+]+),g_loss=([-\d.eE+]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot D/G loss curves from a train_gan.py log file.")
    parser.add_argument("log_path", type=str, help="Path to the train_gan.py log file.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output PNG.")
    return parser.parse_args()


def parse_log(log_path: str) -> Tuple[List[int], List[float], List[float]]:
    """Parse a train_gan.py log file into parallel lists of epochs, d_loss, g_loss."""
    epochs: List[int] = []
    d_losses: List[float] = []
    g_losses: List[float] = []

    with open(log_path, "r") as f:
        for line in f:
            match = LOG_LINE_PATTERN.search(line)
            if match is None:
                continue
            epoch, d_loss, g_loss = match.groups()
            epochs.append(int(epoch))
            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

    return epochs, d_losses, g_losses


def plot_losses(epochs: List[int], d_losses: List[float], g_losses: List[float], output_path: Path) -> None:
    """Plot D/G loss curves over epoch number and save as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, d_losses, label="Discriminator loss")
    ax.plot(epochs, g_losses, label="Generator loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    epochs, d_losses, g_losses = parse_log(args.log_path)

    if not epochs:
        raise ValueError(f"No 'epoch=N,d_loss=X,g_loss=Y' lines found in {args.log_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_losses(epochs, d_losses, g_losses, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
