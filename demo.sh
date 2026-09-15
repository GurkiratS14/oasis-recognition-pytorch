#!/bin/bash
# demo.sh — Run all three live inference demos for COMP3710 Demo 2 Part 4.
# Usage: ./demo.sh [vae|unet|gan|all]

set -e

TASK=${1:-all}

run_vae() {
    echo "=== VAE: reconstructions + manifold ==="
    python3 -m src.visualize_vae --checkpoint-path checkpoints/vae.pth --oasis-root sample_data
}

run_unet() {
    echo "=== UNet: segmentation + DSC ==="
    python3 -m src.visualize_unet --checkpoint-path checkpoints/unet.pth --oasis-root sample_data
}

run_gan() {
    echo "=== GAN: sample generation ==="
    python3 -m src.demo_gan
}

case "$TASK" in
    vae) run_vae ;;
    unet) run_unet ;;
    gan) run_gan ;;
    all) run_vae; run_unet; run_gan ;;
    *) echo "Usage: ./demo.sh [vae|unet|gan|all]"; exit 1 ;;
esac

echo "Done. Check outputs/ for saved images."