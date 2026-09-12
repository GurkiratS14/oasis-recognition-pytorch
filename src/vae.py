"""
vae.py — Variational Autoencoder for 256x256 grayscale brain MRI slices.

Input/output convention: single-channel images of shape (B, 1, 256, 256)
with pixel values in [0, 1]. Latent space is 2D, mainly so it can be
visualized directly as a scatter plot / manifold grid.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 2
# Encoder halves spatial size 6 times: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4.
BOTTLENECK_CHANNELS = 256
BOTTLENECK_SIZE = 4


class Encoder(nn.Module):
    """Conv stack that downsamples a 256x256x1 image to mu/logvar for a 2D latent."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, BOTTLENECK_CHANNELS, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(BOTTLENECK_CHANNELS),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = BOTTLENECK_CHANNELS * BOTTLENECK_SIZE * BOTTLENECK_SIZE
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images to (mu, logvar), each of shape (B, latent_dim)."""
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Mirrors Encoder: expands a 2D latent back to a 256x256x1 image in [0, 1]."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        flat_dim = BOTTLENECK_CHANNELS * BOTTLENECK_SIZE * BOTTLENECK_SIZE
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(BOTTLENECK_CHANNELS, 256, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latents (B, latent_dim) to images of shape (B, 1, 256, 256)."""
        h = self.fc(z)
        h = h.view(-1, BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        return self.deconv(h)


class VAE(nn.Module):
    """Variational Autoencoder wrapping an Encoder, reparameterization, and Decoder."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * std via the reparameterization trick, eps ~ N(0, 1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full VAE forward pass.

        Args:
            x: input images, shape (B, 1, 256, 256), values in [0, 1].

        Returns:
            recon: reconstructed images, shape (B, 1, 256, 256), values in [0, 1].
            mu: latent means, shape (B, latent_dim).
            logvar: latent log-variances, shape (B, latent_dim).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute the VAE loss: reconstruction BCE + beta-weighted KL divergence.

    The reconstruction term is binary cross-entropy summed over all pixels of
    each image and averaged over the batch. The KL term is the closed-form
    divergence between N(mu, exp(logvar)) and a standard normal prior,
    summed over latent dimensions and averaged over the batch.

    Args:
        recon: reconstructed images in [0, 1], shape (B, 1, H, W).
        target: ground-truth images in [0, 1], shape (B, 1, H, W).
        mu: latent means, shape (B, latent_dim).
        logvar: latent log-variances, shape (B, latent_dim).
        beta: weight on the KL divergence term (beta-VAE style).

    Returns:
        Scalar total loss (reconstruction + beta * KL).
    """
    batch_size = recon.shape[0]
    recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / batch_size
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + beta * kld
