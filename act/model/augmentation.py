"""
Image augmentation modules for ACT training.

RandomCenterCrop:
  Automatically switches between random crop (training) and center crop (eval)
  based on self.training, so callers never need to branch on the mode.

  During training with ACT, crops are applied to images before they enter the
  ResNet backbone. The crop can be given as a fixed size or as a ratio of the
  input size (e.g. ratio=0.9 turns 480x480 into 432x432, introducing ~10%
  spatial jitter without losing too much field of view).

  Because ACTPolicy owns these modules, the EMA copy (always in eval mode)
  and the inference pipeline automatically use the deterministic center crop,
  so train / val / deployment all see the same input resolution.

Both modules accept (N, C, H, W) batches and draw independent random
parameters per image — torchvision's transforms would otherwise apply the
same crop / jitter to the whole batch.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF


class RandomCenterCrop(nn.Module):
    """
    Random crop during training, center crop during evaluation.

    This is an nn.Module so it participates in train()/eval() mode switching.
    When used as part of a model, the crop behavior changes automatically
    with model.train() / model.eval() — no conditional branches needed in
    calling code.

    Args:
        size: (height, width) of the output crop, or an int for a square crop
        ratio: alternative to size — crop to (ratio * H, ratio * W) of the input
    """

    def __init__(self, size=None, ratio=None):
        super().__init__()
        assert (size is None) != (ratio is None), "Pass exactly one of size / ratio"
        if isinstance(size, int):
            size = (size, size)
        self.size = tuple(size) if size is not None else None
        self.ratio = ratio

    def _crop_size(self, x):
        if self.size is not None:
            return self.size
        H, W = x.shape[-2:]
        return int(round(H * self.ratio)), int(round(W * self.ratio))

    def forward(self, x):
        h, w = self._crop_size(x)
        if not self.training:
            return TF.center_crop(x, [h, w])
        if x.dim() == 3:
            i, j, th, tw = torchvision.transforms.RandomCrop.get_params(x, (h, w))
            return TF.crop(x, i, j, th, tw)
        # Independent crop offset for every image in the batch
        crops = []
        for img in x:
            i, j, th, tw = torchvision.transforms.RandomCrop.get_params(img, (h, w))
            crops.append(TF.crop(img, i, j, th, tw))
        return torch.stack(crops, dim=0)


class ColorJitter(nn.Module):
    """
    Color jitter augmentation that is a no-op during evaluation.

    Wraps torchvision ColorJitter to follow the same train/eval pattern
    as RandomCenterCrop. Expects float images in [0, 1] (i.e. it must run
    BEFORE ImageNet normalization).

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
        if not self.training:
            return x
        if x.dim() == 3:
            return self._jitter(x)
        # Independent jitter parameters for every image in the batch
        return torch.stack([self._jitter(img) for img in x], dim=0)
