# OASIS Recognition — VAE, UNet, GAN (COMP3710 Demo 2, Part 4)

PyTorch implementations of three recognition tasks on the Preprocessed OASIS
brain MRI dataset:
- **VAE** — variational autoencoder with latent manifold visualization
- **UNet** — brain segmentation with one-hot categorical output (target >0.9 DSC per label)
- **GAN** — generative brain image synthesis

## Structure
- `src/` — core model and training logic
- `notebooks/` — demo notebooks with visualizations
- `slurm/` — Rangpur job scripts
- `prompt_logs/` — AI usage documentation per course requirements

## Dataset

- 256x256, 8-bit grayscale PNGs for both raw slices and segmentation masks.
- Pre-split into train (9666) / validate (1122) / test (546).
- Filename pairing: `case_{id}_slice_{n}.nii.png` ↔ `seg_{id}_slice_{n}.nii.png`.
- Segmentation masks: 4 classes, raw pixel values `{0, 85, 170, 255}` mapped
  to class indices `{0, 1, 2, 3}` (background, CSF, gray matter, white matter).
- Shared loader: `src/dataset.py` (`OASISSliceDataset` for raw slices,
  `OASISSegDataset` for image/mask pairs).

## Task 1 — Variational Autoencoder (`src/vae.py`, `src/train_vae.py`, `src/visualize_vae.py`)

- Encoder/decoder conv stack with a 2D latent bottleneck, chosen specifically
  so the latent manifold can be visualized directly via grid sampling rather
  than requiring a dimensionality-reduction step like UMAP.
- Trained with KL annealing (`beta`: 0 → target over the first 10 epochs) to
  address posterior collapse.
- **Finding**: an ablation across `beta_max` in {1.0, 0.1} showed the
  collapse onto a single dominant latent axis (correlating with
  ventricle/atrophy size) persists regardless of KL weight — indicating a
  structural limitation of a 2D bottleneck's capacity on this dataset, not a
  hyperparameter issue. Reconstructions remain visually strong (skull shape,
  ventricle structure) despite this; fine cortical texture is lost, as
  expected for a 2-number latent representation.
- Evidence: `outputs/vae/` (beta=1.0 run) and `outputs/vae_beta01/` (beta=0.1
  ablation) — each containing `reconstructions.png` and `manifold.png`.
- Run: `python3 -m src.train_vae --oasis-root <path> --epochs 35 --anneal-epochs 10 --beta-max 1.0`
- Visualize: `python3 -m src.visualize_vae --checkpoint-path <ckpt> --oasis-root <path>`

## Task 2 — UNet Segmentation (`src/unet.py`, `src/train_unet.py`, `src/utils.py`, `src/visualize_unet.py`)

- Standard 4-level UNet with skip connections, channel progression
  1→64→128→256→512, 1024-channel bottleneck, 4-class output (background,
  CSF, gray matter, white matter) as raw logits.
- Trained with a combined Dice + cross-entropy loss (`dice_weight=0.5`);
  per-class Dice similarity coefficient (DSC) tracked every epoch using hard
  (argmax) predictions.
- **Results (30 epochs)**: all four classes exceed the 0.9 DSC target —
  bg=0.9995, csf=0.9532, gm=0.9633, wm=0.9766. CSF (the thinnest anatomical
  structure) is expectedly the tightest margin but still clears the bar
  comfortably. Full per-epoch history in `training_logs_unet_30epoch.txt`.
- Qualitative results: `outputs/unet/segmentation.png` — original MRI,
  ground-truth mask, and predicted mask side by side for 6 test samples,
  using a fixed colormap so class identity is visually consistent between
  ground truth and prediction.
- Run: `python3 -m src.train_unet --oasis-root <path> --epochs 30 --batch-size 16`
- Visualize / live inference: `python3 -m src.visualize_unet --checkpoint-path <ckpt> --oasis-root <path>`

## Task 3 — GAN (in progress)

## Repo structure

- `src/` — models, training scripts, shared utilities (loss functions, metrics)
- `slurm/` — Rangpur SLURM job scripts (partition `comp3710`, account
  `comp3710`, `gpu:a100:1`)
- `notebooks/` — demo notebooks
- `outputs/` — saved visualizations (PNGs tracked in git; model checkpoints
  are not, per `.gitignore`)
- `prompt_logs/` — AI usage documentation per course requirements
- `training_logs_*.txt` — full per-epoch training logs, kept as evidence
  alongside git history

## Environment

- Python 3.11 (Rangpur) / 3.13 (local), PyTorch with CUDA (Rangpur) / MPS
  (local Apple Silicon) support
- Dependencies: `torch numpy matplotlib pillow scipy scikit-learn umap-learn`