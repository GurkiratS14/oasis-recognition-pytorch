"""
visualize_unet.py — Qualitative + per-class DSC visualization for a trained
OASIS brain MRI UNet segmentation model.

Usage:
    python -m src.visualize_unet --checkpoint-path checkpoints/unet.pth \
        --oasis-root /path/to/OASIS

Produces a PNG in --output-dir with one row per sample and 3 columns:
original MRI, ground-truth mask, predicted mask. Masks are colored with a
fixed 4-color colormap (background/CSF/gray matter/white matter) shared
between ground truth and prediction. Also prints per-class DSC for the
visualized samples.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from src.dataset import OASISSegDataset
from src.unet import UNet
from src.utils import dice_score_per_class

CLASS_NAMES = ["bg", "csf", "gm", "wm"]
NUM_CLASSES = len(CLASS_NAMES)
# Fixed colors so class identity is visually consistent across figures and
# between ground truth / prediction columns.
MASK_CMAP = ListedColormap(["black", "red", "green", "blue"])


def get_device() -> torch.device:
    """Select mps if available, else cuda if available, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained OASIS UNet.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained UNet state_dict.")
    parser.add_argument("--oasis-root", type=str, required=True, help="Path to the OASIS dataset root.")
    parser.add_argument("--output-dir", type=str, default="outputs/unet", help="Directory to save the output PNG.")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of test images to visualize.")
    return parser.parse_args()


def save_segmentation_figure(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a num_samples x 3 grid: original MRI, ground-truth mask, predicted mask."""
    num_samples = images.shape[0]
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, 2 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    col_titles = ["Original", "Ground Truth", "Prediction"]
    for row in range(num_samples):
        axes[row, 0].imshow(images[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(masks[row], cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 2].imshow(preds[row], cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        for col in range(3):
            axes[row, col].axis("off")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    test_dataset = OASISSegDataset(args.oasis_root, split="test")
    loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=False)
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        dsc = dice_score_per_class(logits, masks, num_classes=NUM_CLASSES)

    dsc_str = ", ".join(f"{name}={score:.4f}" for name, score in zip(CLASS_NAMES, dsc.tolist()))
    print(f"Per-class DSC (visualized batch): {dsc_str}")

    output_path = output_dir / "segmentation.png"
    save_segmentation_figure(images.cpu(), masks.cpu(), preds.cpu(), output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
