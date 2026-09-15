# Reconstructed AI Prompt/Reasoning Log — OASIS Recognition Project (Part 4)

This log documents the AI-assisted development process for Tasks 1–3 (VAE,
UNet, GAN), plus demo preparation. Live Claude Code session exports were not
captured in real time during this project; this document is an accurate
secondary record reconstructed from the accompanying planning conversation,
covering every prompt given to Claude Code, why it was scoped that way, and
what came back. It is corroborated by the project's git commit history
(23 commits, incremental and individually meaningful).

---

## 1. Dataset loader (`src/dataset.py`)

**Reconnaissance first**: Before any code was written, the OASIS dataset
structure was inspected directly on Rangpur (`ls`, `file`, and a small
Python/PIL script) to determine: folder structure (pre-split train/validate/
test, ~9666/1122/546 images), filename convention
(`case_{id}_slice_{n}.nii.png` / `seg_{id}_slice_{n}.nii.png`), image
properties (256x256, 8-bit grayscale), and segmentation mask class count
(4 classes, raw pixel values `{0, 85, 170, 255}`).

**Prompt given to Claude Code**: Create a shared PyTorch `Dataset` loader
for the OASIS keras_png_slices structure, with two classes — one for raw
slices only (VAE/GAN use case) and one for (image, mask) pairs (UNet use
case) — correctly mapping the mask's raw pixel values to class indices 0-3.

**Outcome**: `OASISSliceDataset` and `OASISSegDataset`, plus a
`mask_to_class_indices` helper. Verified locally against 5 sample files
copied down from Rangpur via `scp` — confirmed correct shapes, dtypes, and
value ranges before trusting it for real training.

---

## 2. VAE (`src/vae.py`, `src/train_vae.py`, `src/visualize_vae.py`)

**Design decision**: chose a 2D latent space specifically so the manifold
could be visualized by direct grid sampling rather than requiring UMAP,
since the lab task allows either approach and 2D gives a more directly
interpretable demo result.

**Prompt 1**: Create the VAE encoder/decoder architecture and loss function
(BCE reconstruction + KL divergence, beta-weighted) for 256x256 grayscale
input with a 2D latent bottleneck. Model + loss only, no training loop.

**Review**: verified conv/deconv channel and spatial dimension math by hand
(256→128→64→32→16→8→4 and back) before accepting.

**Prompt 2**: Create the training script (train/val loop, device selection,
checkpointing, CLI args via argparse).

**First training run** (beta=1.0 fixed): loss plateaued after epoch 2
(train ~18300→16545 over 20 epochs, val loss flat ~16860-17100).

**Diagnosis via visualization**: prompted for a reconstruction comparison
grid + a 2D latent manifold grid (via `scipy.stats.norm.ppf` on
evenly-spaced probabilities). Result: reconstructions were good (captured
skull shape, ventricle structure) but the manifold showed variation along
only one latent axis — partial posterior collapse.

**Fix attempt 1**: modified `train_vae.py` to add KL annealing (`beta`
ramping 0→`beta_max` over the first N epochs). Re-ran (35 epochs,
beta_max=1.0) — same collapse pattern persisted.

**Fix attempt 2 (ablation)**: re-ran with `beta_max=0.1` (10x reduction) to
test whether the collapse was a beta-magnitude issue at all. Same collapse
pattern, nearly identical final loss (~16880s in both runs).

**Conclusion**: the collapse is a structural limitation of a 2D
bottleneck's capacity for this dataset's anatomical variation, not a
hyperparameter/annealing issue. This diagnostic process (baseline →
annealing → ablation) is the documented "substantial analysis" for this
task.

---

## 3. UNet (`src/unet.py`, `src/utils.py`, `src/train_unet.py`, `src/visualize_unet.py`)

**Prompt 1**: Create a standard 4-level UNet (channel progression
1→64→128→256→512, 1024-channel bottleneck, skip connections, 4-class raw
logit output) for 256x256 grayscale input.

**Review**: verified channel/spatial dimensions and skip-connection
concatenation shapes at every level by hand before accepting.

**Prompt 2**: Add to `src/utils.py`: a differentiable soft Dice loss, a
combined Dice+cross-entropy loss, and a non-differentiable per-class DSC
metric using hard (argmax) predictions — explicitly separating the soft
loss used for backprop from the hard metric used for reporting.

**Prompt 3**: Create the training script — train/val loop, per-class DSC
tracked every epoch, checkpointing.

**Training run** (30 epochs, batch size 16, dice_weight=0.5): all four
classes exceeded the 0.9 DSC target — bg=0.9995, csf=0.9532, gm=0.9633,
wm=0.9766. CSF (thinnest anatomical structure) was expectedly the tightest
margin, with visible noise mid-training (e.g. dropping to 0.877 at epoch 7)
before recovering.

**Prompt 4**: Create a visualization script producing a fixed-colormap
comparison grid (original / ground truth / prediction) across several test
samples, printing per-class DSC for the visualized batch — doubling as the
live-inference tool for the demo.

---

## 4. GAN (`src/gan.py`, `src/train_gan.py`, `src/plot_gan_losses.py`)

**Prompt 1**: Create a DCGAN-style Generator (100-dim noise → 256x256x1)
and Discriminator (256x256x1 → single logit), with standard DCGAN weight
initialization.

**Review**: verified dimension math through both networks before accepting.

**Prompt 2**: Create the training script — standard DCGAN alternating
update loop (train D on real+detached-fake, train G via the non-saturating
loss trick), fixed-noise sample grid saved every N epochs to track
progression.

**Training run** (50 epochs, batch size 32): discriminator loss stayed low
throughout (dipping near-zero around epochs 9-14) while generator loss
remained elevated and noisy — a discriminator-dominant pattern. However,
the actual generated sample grids (epochs 5, 25, 50) showed clear positive
progression: noise at epoch 5, recognisable and anatomically varied
brain-like slices with distinct ventricle shapes by epoch 25-50, genuine
diversity across the grid, no mode collapse — the loss curve alone
understated the actual training success.

**Prompt 3**: Create a script to parse the training log and plot both loss
curves for the write-up.

---

## 5. Rangpur/SLURM debugging (applies across all three tasks)

Significant iterative debugging was required: missing `--account=comp3710`,
incorrect GPU gres name (`gpu:1` → `gpu:a100:1`, confirmed via
`scontrol show node`), and a cluster-wide misconfiguration where all
comp3710 A100 nodes report `RealMemory=1`, making any `--mem` request fail —
fixed by omitting `--mem` entirely, relying on the partition's
`MaxMemPerNode=UNLIMITED` default. Also diagnosed Python's stdout buffering
under SLURM (fixed with `python3 -u`) after a training job appeared silent
despite running correctly.

---

## 6. Demo preparation

**Goal**: ensure live inference works during the practical demo regardless
of whether Rangpur/VPN access is available on the day.

**Steps taken**:
- Pulled all trained checkpoints (`vae.pth`, `unet.pth`, `generator.pth`,
  `discriminator.pth`) from Rangpur to the local machine via `scp`.
- Pulled a small set of real OASIS test-split files (`case_441_slice_*`,
  the actual test-split case ID, confirmed via `ls` on Rangpur) into
  `sample_data/` for local inference testing without needing the full
  dataset mounted.
- Verified all three models run correctly on local Apple Silicon (MPS):
  `visualize_unet.py` and `visualize_vae.py` produce correct outputs and
  DSC scores locally, matching Rangpur results.
- **Prompt**: Create `src/demo_gan.py`, a standalone script that loads the
  trained Generator checkpoint and saves a fresh 16-image sample grid, for
  a clean single-command GAN demo (rather than an inline Python snippet).
- Created `demo.sh`, a single shell script wrapping all three
  visualize/demo commands (`./demo.sh [vae|unet|gan|all]`), so no terminal
  commands need to be memorized during the live demo.