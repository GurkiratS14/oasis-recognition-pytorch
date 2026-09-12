"""
dataset.py — Shared OASIS dataset loader for VAE, UNet, and GAN tasks.

Dataset layout (Rangpur): /home/groups/comp3710/OASIS/
  keras_png_slices_{train,validate,test}/       raw MRI slices, case_{id}_slice_{n}.nii.png
  keras_png_slices_seg_{train,validate,test}/   segmentation masks, seg_{id}_slice_{n}.nii.png

Images: 256x256, 8-bit grayscale.
Segmentation masks: 4 classes, raw pixel values {0, 85, 170, 255} mapped to
class indices {0, 1, 2, 3} (background, CSF, gray matter, white matter).
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Raw pixel values in the seg masks, in class-index order.
SEG_PIXEL_VALUES = [0, 85, 170, 255]
NUM_CLASSES = len(SEG_PIXEL_VALUES)


def mask_to_class_indices(mask_arr: np.ndarray) -> np.ndarray:
    """Map raw {0,85,170,255} pixel values to class indices {0,1,2,3}."""
    out = np.zeros_like(mask_arr, dtype=np.int64)
    for class_idx, pixel_val in enumerate(SEG_PIXEL_VALUES):
        out[mask_arr == pixel_val] = class_idx
    return out


class OASISSliceDataset(Dataset):
    """
    Loads raw MRI slices only (for the VAE / GAN — no labels needed).
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        root_dir: path to /home/groups/comp3710/OASIS (or local copy)
        split: 'train', 'validate', or 'test'
        """
        self.image_dir = Path(root_dir) / f"keras_png_slices_{split}"
        self.filenames = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.filenames[idx]
        img = Image.open(img_path)
        arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # add channel dim: [1, H, W]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


class OASISSegDataset(Dataset):
    """
    Loads (image, mask) pairs for the UNet segmentation task.
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.image_dir = Path(root_dir) / f"keras_png_slices_{split}"
        self.mask_dir = Path(root_dir) / f"keras_png_slices_seg_{split}"
        self.filenames = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def _mask_filename(self, image_filename: str) -> str:
        # case_001_slice_0.nii.png -> seg_001_slice_0.nii.png
        return re.sub(r"^case_", "seg_", image_filename)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        mask_filename = self._mask_filename(img_filename)

        img = Image.open(self.image_dir / img_filename)
        mask = Image.open(self.mask_dir / mask_filename)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = mask_to_class_indices(np.array(mask))

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)      # [1, H, W]
        mask_tensor = torch.from_numpy(mask_arr).long()          # [H, W], class indices

        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor