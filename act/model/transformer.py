"""
DETR-style Transformer for the ACT model.

This module implements the Transformer encoder and decoder used in ACT,
adapted from the DETR (DEtection TRansformer) codebase. The key modification
from a standard Transformer is how position encodings are handled:

  - Position encodings are NOT added to the input embeddings directly.
  - Instead, they are added to Q and K inside each attention layer,
    leaving V unchanged. This means the position information influences
    attention weights (which tokens attend to which) but does not alter
    the content representations being aggregated.

The Transformer has two main roles in ACT:

1. **DETR Transformer** (encoder + decoder):
   - Encoder processes: [latent_token, proprio_token, image_patch_1, ..., image_patch_HW]
   - Decoder uses num_queries learnable query embeddings to cross-attend
     to the encoder memory, producing one action prediction per query.

2. **CVAE Encoder** (encoder only, training time):
   - Processes: [CLS_token, qpos_embed, action_embed_1, ..., action_embed_T]
   - The CLS token output is projected to obtain mu and logvar for the
     latent variable z.

Architecture notes:
  - ACT uses post-norm (normalize_before=False) by default
  - The decoder uses return_intermediate=True, returning outputs from all
    layers stacked along dim 0. Only the last layer output [0] is used
    for action prediction.
  - xavier_uniform_ initialization is applied to all parameters with dim > 1
"""

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Transformer(nn.Module):
    """
    Full DETR-style Transformer with encoder and decoder.

    The forward method handles two input formats:
    1. 4D src (bs, C, H, W): image features from backbone — flattened to sequence
    2. 3D src (bs, HW, C): pre-flattened features (e.g., low-dim state only)

    For the 4D case (image input), latent_input and proprio_input are prepended
    as extra tokens to the encoder input sequence, with their own learned
    position embeddings (additional_pos_embed).

    Args:
        d_model: hidden dimension of the transformer (default 512)
        nhead: number of attention heads (default 8)
        num_encoder_layers: number of encoder layers (default 6)
        num_decoder_layers: number of decoder layers (default 6)
        dim_feedforward: FFN intermediate dimension (default 2048)
        dropout: dropout rate (default 0.1)
        activation: activation function in FFN, "relu" or "gelu"
        normalize_before: if True, use pre-norm; if False, use post-norm
        return_intermediate_dec: if True, decoder returns all layer outputs
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Xavier uniform initialization for all parameters with dim > 1."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
    ):
        """
        Forward pass through the full encoder-decoder Transformer.

        Args:
            src: source features
                 - 4D (bs, C, H, W): image features from backbone
                 - 3D (bs, HW, C): pre-flattened features
            mask: source key padding mask (currently unused, pass None)
            query_embed: (num_queries, d_model) learnable action query embeddings
            pos_embed: position encoding matching src spatial dims
                       - 4D: (1, C, H, W) for image features
                       - 3D: (num_tokens, d_model) for pre-flattened
            latent_input: (bs, d_model) latent variable projected to hidden dim
            proprio_input: (bs, d_model) proprioception projected to hidden dim
            additional_pos_embed: (2, d_model) learned pos embed for latent & proprio

        Returns:
            hs: decoder output of shape (1, bs, num_queries, d_model)
                (or (num_layers, bs, num_queries, d_model) if return_intermediate)
        """
        if len(src.shape) == 4:
            # Image features: flatten (bs, C, H, W) -> (H*W, bs, C)
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

            # Prepend latent and proprio tokens with their position embeddings
            # additional_pos_embed shape: (2, d_model) -> (2, bs, d_model)
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            # Stack latent and proprio as (2, bs, d_model), prepend to src
            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            # Pre-flattened features: (bs, HW, C) -> (HW, bs, C)
            assert len(src.shape) == 3
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # Decoder target: initialized to zeros, same shape as query_embed
        # The queries learn to extract information from memory via cross-attention
        tgt = torch.zeros_like(query_embed)

        # Encode: build contextual representations of the full input
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # Decode: queries cross-attend to memory to produce action predictions
        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )

        # Transpose from (num_layers, seq, bs, d_model) to (num_layers, bs, seq, d_model)
        hs = hs.transpose(1, 2)
        return hs


class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderLayers with optional final LayerNorm.

    This encoder is used in two contexts in ACT:
    1. Inside the DETR Transformer — encodes [latent, proprio, image_patches]
    2. As the CVAE encoder — encodes [CLS, qpos, action_sequence]

    Args:
        encoder_layer: an instance of TransformerEncoderLayer (will be deep-copied)
        num_layers: number of encoder layers to stack
        norm: optional final normalization layer
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ):
        """
        Args:
            src: (seq_len, bs, d_model) input sequence
            mask: attention mask (unused in ACT)
            src_key_padding_mask: (bs, seq_len) True for positions to ignore
            pos: (seq_len, 1_or_bs, d_model) position encoding

        Returns:
            output: (seq_len, bs, d_model) encoded representations
        """
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    Stack of TransformerDecoderLayers with optional intermediate output collection.

    When return_intermediate=True, outputs from all layers are stacked along
    dim 0 and returned. ACT uses this with return_intermediate=True, but only
    uses the last layer output [0] for action prediction.

    Args:
        decoder_layer: an instance of TransformerDecoderLayer (will be deep-copied)
        num_layers: number of decoder layers to stack
        norm: final normalization layer (always applied in ACT)
        return_intermediate: if True, return all layer outputs stacked
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        """
        Args:
            tgt: (num_queries, bs, d_model) decoder input (initialized to zeros)
            memory: (seq_len, bs, d_model) encoder output
            tgt_mask: self-attention mask for queries (unused in ACT)
            memory_mask: cross-attention mask (unused in ACT)
            tgt_key_padding_mask: padding mask for queries (unused in ACT)
            memory_key_padding_mask: padding mask for memory
            pos: (seq_len, bs, d_model) position encoding for memory
            query_pos: (num_queries, bs, d_model) position encoding for queries

        Returns:
            output: (1, num_queries, bs, d_model) if not return_intermediate
                    (num_layers, num_queries, bs, d_model) if return_intermediate
        """
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer: Self-Attention -> Add&Norm -> FFN -> Add&Norm.

    Key design choice: position encoding is added to Q and K only (not V).
    This means position information affects which tokens attend to each other
    (via dot-product of Q and K), but does not modify the content being
    aggregated (V). This is important because it keeps the content
    representations position-independent — the model learns position-aware
    routing but position-independent features.

    Supports both pre-norm and post-norm configurations:
    - Post-norm (default in ACT): attention -> residual -> norm -> FFN -> residual -> norm
    - Pre-norm: norm -> attention -> residual -> norm -> FFN -> residual

    Args:
        d_model: hidden dimension
        nhead: number of attention heads
        dim_feedforward: FFN intermediate dimension
        dropout: dropout rate
        activation: "relu" or "gelu"
        normalize_before: if True, use pre-norm
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Tensor | None):
        """Add position encoding to tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Post-norm: attention -> residual -> norm -> FFN -> residual -> norm"""
        # Self-attention: add pos to Q and K only, V stays as raw content
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Pre-norm: norm -> attention -> residual -> norm -> FFN -> residual"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with three sub-layers:
    1. Self-Attention (queries attend to each other)
    2. Cross-Attention (queries attend to encoder memory)
    3. FFN

    Position encoding handling:
    - Self-attention: query_pos is added to both Q and K, so queries are
      position-aware when attending to each other
    - Cross-attention: query_pos is added to Q, memory pos is added to K,
      so queries know their own position and the spatial layout of memory

    Args:
        d_model: hidden dimension
        nhead: number of attention heads
        dim_feedforward: FFN intermediate dimension
        dropout: dropout rate
        activation: "relu" or "gelu"
        normalize_before: if True, use pre-norm
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        """Post-norm decoder layer."""
        # 1. Self-attention among queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross-attention: queries attend to encoder memory
        # Q = query content + query_pos, K = memory content + memory_pos
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        """Pre-norm decoder layer."""
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    """Create N deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
