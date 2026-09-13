"""
unet.py — UNet for 4-class segmentation of 256x256 grayscale brain MRI slices.

Architecture: 4 downsampling double-conv blocks (1 -> 64 -> 128 -> 256 -> 512),
a 1024-channel bottleneck, and 4 upsampling double-conv blocks with skip
connections mirroring the encoder back down to 64 channels, followed by a
1x1 conv to NUM_CLASSES raw logits.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4


class DoubleConv(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers, same padding, preserving spatial size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """A DoubleConv followed by 2x2 max pooling. Returns (pre-pool skip, pooled output)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class Up(nn.Module):
    """Transpose-conv upsample, concatenate the encoder skip, then a DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # With 256x256 inputs and 4 poolings, sizes divide evenly and this is a
        # no-op; kept as a safeguard for odd input sizes where transpose conv
        # output can be off by a pixel relative to the skip connection.
        diff_h = skip.shape[-2] - x.shape[-2]
        diff_w = skip.shape[-1] - x.shape[-1]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        assert x.shape[-2:] == skip.shape[-2:], (
            f"Skip connection spatial mismatch: decoder {x.shape[-2:]} vs encoder {skip.shape[-2:]}"
        )
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    """UNet mapping (B, 1, 256, 256) grayscale MRI slices to (B, 4, 256, 256) class logits."""

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the UNet forward pass.

        Args:
            x: input images, shape (B, 1, 256, 256).

        Returns:
            Raw class logits, shape (B, num_classes, 256, 256). No softmax is
            applied — loss functions handle softmax/log-softmax internally.
        """
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.final_conv(x)
