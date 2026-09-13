"""
utils.py — Loss functions and evaluation metrics for the OASIS UNet
segmentation task (4-class: background, CSF, gray matter, white matter).
"""

import torch
import torch.nn.functional as F

NUM_CLASSES = 4


def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer class-index targets (B, H, W) to one-hot (B, num_classes, H, W)."""
    return F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss, differentiable for backprop.

    Args:
        logits: raw model outputs, shape (B, num_classes, H, W).
        targets: integer class-index ground truth, shape (B, H, W).
        num_classes: number of segmentation classes.
        epsilon: smoothing constant to avoid division by zero.

    Returns:
        Scalar loss: 1 minus the mean Dice coefficient across all classes
        and the batch, computed on soft (softmax) probabilities.
    """
    probs = F.softmax(logits, dim=1)
    targets_one_hot = _one_hot(targets, num_classes)

    intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
    pred_sum = probs.sum(dim=(0, 2, 3))
    target_sum = targets_one_hot.sum(dim=(0, 2, 3))

    dice_per_class = (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
    return 1.0 - dice_per_class.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Weighted combination of soft Dice loss and cross-entropy loss.

    Args:
        logits: raw model outputs, shape (B, num_classes, H, W).
        targets: integer class-index ground truth, shape (B, H, W).
        dice_weight: weight on the Dice term; (1 - dice_weight) weights cross-entropy.

    Returns:
        Scalar loss: dice_weight * dice_loss + (1 - dice_weight) * cross_entropy.
    """
    dice = dice_loss(logits, targets)
    ce = F.cross_entropy(logits, targets)
    return dice_weight * dice + (1 - dice_weight) * ce


def dice_score_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Per-class Dice similarity coefficient (DSC) using hard predictions, for metric reporting.

    Non-differentiable: uses argmax predictions rather than soft probabilities,
    and is wrapped in torch.no_grad() since it is intended for evaluation only.

    Args:
        logits: raw model outputs, shape (B, num_classes, H, W).
        targets: integer class-index ground truth, shape (B, H, W).
        num_classes: number of segmentation classes.
        epsilon: smoothing constant to avoid division by zero.

    Returns:
        Tensor of shape (num_classes,) with the DSC score for each class.
    """
    with torch.no_grad():
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        preds_one_hot = _one_hot(preds, num_classes)
        targets_one_hot = _one_hot(targets, num_classes)

        intersection = (preds_one_hot * targets_one_hot).sum(dim=(0, 2, 3))
        pred_sum = preds_one_hot.sum(dim=(0, 2, 3))
        target_sum = targets_one_hot.sum(dim=(0, 2, 3))

        return (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
