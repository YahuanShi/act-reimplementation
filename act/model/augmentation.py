"""
Image augmentation modules for ACT training.

RandomCenterCrop:
  Automatically switches between random crop (training) and center crop (eval)
  based on self.training, so callers never need to branch on the mode.

  During training with ACT, crops are applied to images before they enter the
  ResNet backbone. The crop size should be set to the desired input resolution
  (e.g. 480 → 440 to introduce ~9% spatial jitter without losing too much
  field of view).
"""

import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF


class RandomCenterCrop(nn.Module):
    """
    Random crop during training, center crop during evaluation.

    This is an nn.Module so it participates in train()/eval() mode switching.
    When used as part of a model or passed to a DataLoader that sets training
    mode, the crop behavior changes automatically — no conditional branches
    needed in calling code.

    Args:
        size: (height, width) of the output crop
    """

    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = tuple(size)

    def forward(self, x):
        h, w = self.size
        if self.training:
            i, j, th, tw = torchvision.transforms.RandomCrop.get_params(x, (h, w))
            return TF.crop(x, i, j, th, tw)
        return TF.center_crop(x, [h, w])


class ColorJitter(nn.Module):
    """
    Color jitter augmentation that is a no-op during evaluation.

    Wraps torchvision ColorJitter to follow the same train/eval pattern
    as RandomCenterCrop.

    Args:
        brightness, contrast, saturation, hue: jitter strengths (0 = disabled)
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        super().__init__()
        self._jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def forward(self, x):
        if self.training:
            return self._jitter(x)
        return x
