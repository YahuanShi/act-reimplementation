"""
Position encodings for the ACT model.

Two types of position encoding are used in ACT:

1. **1D sinusoidal encoding table** (`get_sinusoid_encoding_table`):
   Used in the CVAE encoder to provide positional information for the
   [CLS, qpos, action_1, ..., action_T] input sequence. This is a fixed
   (non-learnable) encoding computed once and stored as a buffer.

   Formula for each (position, dimension) pair:
     - Even dims: sin(pos / 10000^(2i/d_hid))
     - Odd dims:  cos(pos / 10000^(2i/d_hid))

2. **2D sinusoidal encoding** (`PositionEmbeddingSine`):
   Used on image feature maps from the ResNet backbone. Extends the 1D
   sinusoidal encoding to 2D by computing separate encodings for x and y
   spatial coordinates, then concatenating them. This is the same approach
   used in DETR (DEtection TRansformer).

   For a feature map of shape (H, W), this produces position encodings
   of shape (1, d_model, H, W) where d_model = 2 * num_pos_feats.
"""

import math

import numpy as np
import torch
from torch import nn


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Generate a 1D sinusoidal position encoding table.

    Each row corresponds to a position in the sequence, and each column
    corresponds to a dimension of the encoding. The encoding alternates
    between sin and cos functions with geometrically increasing wavelengths,
    allowing the model to distinguish positions and learn relative offsets.

    Args:
        n_position: number of positions to encode (e.g., 1 + 1 + chunk_size
                    for [CLS] + qpos + action sequence)
        d_hid: encoding dimension (should match hidden_dim of the transformer)

    Returns:
        FloatTensor of shape (1, n_position, d_hid) — the leading dim is a
        batch dimension for easy broadcasting.
    """

    def get_position_angle_vec(position):
        # For each hidden dimension index j, compute:
        #   position / 10000^(2*(j//2) / d_hid)
        # This creates a spectrum of frequencies: low-index dims capture
        # fine-grained position info, high-index dims capture coarse info.
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

    # Apply sin to even indices (0, 2, 4, ...) and cos to odd indices (1, 3, 5, ...)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PositionEmbeddingSine(nn.Module):
    """
    2D sinusoidal position encoding for image feature maps (from DETR).

    Given a feature map of shape (bs, C, H, W), this module produces a
    position encoding of shape (1, d_model, H, W) where d_model = 2 * num_pos_feats.

    The encoding works by:
    1. Creating cumulative coordinate grids for x (width) and y (height)
    2. Optionally normalizing coordinates to [0, 2*pi]
    3. Dividing by temperature-scaled frequency bands (same as 1D sinusoidal)
    4. Applying sin/cos alternately
    5. Concatenating y-encoding and x-encoding along the channel dimension

    This gives the transformer spatial awareness of where each patch is
    located in the original image.

    Args:
        num_pos_feats: number of position features per spatial direction.
                       The total encoding dimension is 2 * num_pos_feats
                       (half for x, half for y). Should be hidden_dim // 2.
        temperature: controls the frequency spectrum (default 10000, same as
                     the original "Attention Is All You Need" paper)
        normalize: if True, normalize coordinates to [0, scale] before encoding
        scale: normalization scale (default 2*pi)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        """
        Args:
            tensor: feature map of shape (bs, C, H, W)

        Returns:
            pos: position encoding of shape (1, 2*num_pos_feats, H, W)
        """
        x = tensor

        # Create a mask of all ones (no masking — unlike DETR which may have
        # padded images, we assume all spatial positions are valid)
        not_mask = torch.ones_like(x[0, [0]])  # shape: (1, H, W)

        # Cumulative sum along height and width to get coordinate grids
        # y_embed[0, i, j] = i+1  (row index, 1-based)
        # x_embed[0, i, j] = j+1  (col index, 1-based)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            # Normalize to [0, scale] where scale is typically 2*pi
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Frequency bands: temperature^(2i / num_pos_feats) for i in [0, num_pos_feats)
        # Lower indices -> higher frequency (fine position), higher indices -> lower frequency
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Divide coordinates by frequency bands: shape (1, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Apply sin to even dims, cos to odd dims, then interleave and flatten
        # stack(..., dim=4) creates pairs [sin, cos] then flatten merges them
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # (1, H, W, num_pos_feats)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # (1, H, W, num_pos_feats)

        # Concatenate y and x encodings, then permute to (1, d_model, H, W)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
