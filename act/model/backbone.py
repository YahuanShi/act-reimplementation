"""
Visual backbone for the ACT model.

This module provides a ResNet18 backbone with frozen BatchNorm for extracting
visual features from camera images, following the DETR approach.

Architecture overview:
  Input image (bs, 3, H, W)
      |
      v
  ResNet18 (with FrozenBatchNorm2d)
      |
      v
  layer4 output (bs, 512, H/32, W/32)
      |
      v
  Joiner: pairs features with 2D sinusoidal position encoding

Key design choices:

1. **FrozenBatchNorm2d**: When using a pretrained ResNet with small batch sizes
   (typical in robot learning, e.g., bs=8), updating BatchNorm statistics would
   be harmful — the batch statistics would be noisy and degrade the pretrained
   features. FrozenBatchNorm2d keeps the ImageNet statistics fixed by storing
   weight, bias, running_mean, running_var as buffers (not parameters).

2. **Only layer4 output**: ACT uses return_interm_layers=False, extracting only
   the final layer4 features. This gives a 32x spatial downsampling:
   - 480x640 input -> 15x20 feature map (300 spatial tokens)
   - 240x320 input -> 8x10 feature map (80 spatial tokens)

3. **Shared backbone**: All cameras share the same backbone weights. Camera
   features are concatenated along the width dimension before feeding into
   the Transformer.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .position_encoding import PositionEmbeddingSine


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and affine parameters are fixed.

    All four parameters (weight, bias, running_mean, running_var) are registered
    as buffers, not parameters. This means:
    - They are NOT updated during training (no gradient, no optimizer step)
    - They ARE saved/loaded with the model state_dict
    - They ARE moved to GPU with .cuda()

    The forward pass computes the same transformation as regular BatchNorm:
      output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
    but in a fused form for efficiency:
      scale = weight * rsqrt(running_var + eps)
      bias_new = bias - running_mean * scale
      output = input * scale + bias_new

    Args:
        n: number of features (channels)
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Remove num_batches_tracked from state_dict if present — it's a
        # standard BN buffer that FrozenBN doesn't use
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Reshape to (1, C, 1, 1) for broadcasting with (N, C, H, W) input
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        # Fused computation: avoids separate subtraction and division
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    Base class that wraps a torchvision backbone with IntermediateLayerGetter.

    IntermediateLayerGetter hooks into specified layers of the backbone and
    returns their outputs as an OrderedDict. For ACT, we only need layer4
    (the final convolutional block).

    Args:
        backbone: a torchvision ResNet model
        train_backbone: whether to update backbone weights during training
        num_channels: number of output channels from the backbone
                      (512 for ResNet18/34, 2048 for ResNet50/101/152)
        return_interm_layers: if True, return outputs from layers 1-4;
                              if False, return only layer4
    """

    def __init__(self, backbone, train_backbone, num_channels,
                 return_interm_layers):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1",
                             "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone,
                                           return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        """
        Args:
            tensor: (bs, 3, H, W) input image

        Returns:
            OrderedDict of feature maps. With return_interm_layers=False,
            contains one entry "0" with shape (bs, num_channels, H/32, W/32).
        """
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm.

    Loads a pretrained ResNet model from torchvision, replacing all BatchNorm
    layers with FrozenBatchNorm2d to prevent statistics corruption from small
    training batches.

    Args:
        name: ResNet variant name ("resnet18", "resnet34", "resnet50", etc.)
        train_backbone: if True, backbone parameters require gradients
        return_interm_layers: if True, return all intermediate layer outputs
        dilation: if True, use dilated convolutions in the last block
                  (increases spatial resolution at the cost of receptive field)
    """

    def __init__(self, name, train_backbone, return_interm_layers, dilation):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights="IMAGENET1K_V1",
            norm_layer=FrozenBatchNorm2d)
        # ResNet18/34 have 512 channels in layer4; ResNet50+ have 2048
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels,
                         return_interm_layers)


class Joiner(nn.Sequential):
    """
    Combines a backbone with a position encoding module.

    For each feature map output by the backbone, the Joiner computes the
    corresponding 2D position encoding. This pairing is essential because
    the Transformer needs both the visual features and their spatial
    positions to perform position-aware attention.

    Args:
        backbone: a BackboneBase instance
        position_embedding: a PositionEmbeddingSine instance
    """

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor):
        """
        Args:
            tensor: (bs, 3, H, W) input image

        Returns:
            out: list of feature maps, each (bs, num_channels, H', W')
            pos: list of position encodings, each (1, d_model, H', W')
        """
        xs = self[0](tensor)  # backbone forward
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # Compute 2D position encoding matching this feature map's spatial dims
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_backbone(backbone_name, hidden_dim, lr_backbone=1e-5, dilation=False):
    """
    Build the complete backbone: ResNet + PositionEmbeddingSine -> Joiner.

    Args:
        backbone_name: ResNet variant ("resnet18", "resnet34", etc.)
        hidden_dim: Transformer hidden dimension (position encoding will be
                    hidden_dim-dimensional, split as hidden_dim//2 per axis)
        lr_backbone: backbone learning rate; if 0, backbone is frozen
        dilation: whether to use dilated convolutions

    Returns:
        Joiner model with .num_channels attribute
    """
    position_embedding = PositionEmbeddingSine(
        num_pos_feats=hidden_dim // 2, normalize=True)
    train_backbone = lr_backbone > 0
    backbone = Backbone(backbone_name, train_backbone,
                        return_interm_layers=False, dilation=dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
