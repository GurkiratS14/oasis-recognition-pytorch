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