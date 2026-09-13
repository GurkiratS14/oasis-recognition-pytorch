"""
gan.py — DCGAN-style Generator and Discriminator for synthesizing 256x256
grayscale brain MRI slices (single channel, pixel values in [0, 1]).
"""

import torch
import torch.nn as nn

LATENT_DIM = 100
# Generator/Discriminator feature maps meet at a 256-channel, 4x4 bottleneck.
BOTTLENECK_CHANNELS = 256
BOTTLENECK_SIZE = 4


class Generator(nn.Module):
    """Projects a latent noise vector up to a 256x256x1 image via transpose convs."""

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
        """Generate images from noise.

        Args:
            z: latent noise vectors, shape (B, latent_dim).

        Returns:
            Generated images, shape (B, 1, 256, 256), values in [0, 1].
        """
        h = self.fc(z)
        h = h.view(-1, BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        return self.deconv(h)


class Discriminator(nn.Module):
    """Downsamples a 256x256x1 image to a single real/fake logit."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 256 -> 128 (no BatchNorm on first layer)
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
        self.fc = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images as real or fake.

        Args:
            x: input images, shape (B, 1, 256, 256), values in [0, 1].

        Returns:
            Raw logits, shape (B, 1). No sigmoid is applied — pair with
            BCEWithLogitsLoss for numerical stability.
        """
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)


def weights_init(m: nn.Module) -> None:
    """Standard DCGAN weight initialization: N(0, 0.02) for conv/batchnorm weights.

    Apply via model.apply(weights_init) after instantiating Generator/Discriminator.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
