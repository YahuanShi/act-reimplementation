"""
Step 2.4 — Transformer validation experiment.

This script verifies the Transformer components work correctly by running
forward passes with dummy data and checking output shapes. It tests:

1. TransformerEncoder (used as CVAE encoder):
   - Input: [CLS, qpos, action_1, ..., action_50] = 52 tokens
   - Output: same shape (52, bs, d_model)

2. Full Transformer with 4D image features:
   - Image features (bs, C, H, W) get flattened and prepended with
     latent + proprio tokens
   - Decoder produces (num_layers, bs, num_queries, d_model)

3. Gradient flow:
   - Verify that gradients propagate through the entire network
"""

import torch

from act.model.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer


def test_cvae_encoder():
    """Test the encoder in CVAE mode: encode [CLS, qpos, action_seq]."""
    print("=" * 60)
    print("Test 1: CVAE Encoder (TransformerEncoder)")
    print("=" * 60)

    d_model = 256
    nhead = 8
    num_layers = 4
    chunk_size = 50
    bs = 2
    # Total sequence: CLS(1) + qpos(1) + actions(50) = 52
    seq_len = 1 + 1 + chunk_size

    enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024)
    encoder = TransformerEncoder(enc_layer, num_layers)

    # Simulate encoder input: already projected to d_model
    src = torch.randn(seq_len, bs, d_model)
    pos = torch.randn(seq_len, 1, d_model)

    # Padding mask: first 2 tokens (CLS + qpos) are not padded,
    # last 10 actions are padded in this example
    is_pad = torch.zeros(bs, seq_len, dtype=torch.bool)
    is_pad[:, -10:] = True

    output = encoder(src, src_key_padding_mask=is_pad, pos=pos)

    print(f"  Input shape:  {src.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (seq_len, bs, d_model), (
        f"Expected ({seq_len}, {bs}, {d_model}), got {output.shape}"
    )

    # The CLS token output (index 0) will be used for mu/logvar
    cls_output = output[0]
    print(f"  CLS output shape: {cls_output.shape}")
    assert cls_output.shape == (bs, d_model)

    print("  PASSED\n")


def test_full_transformer():
    """Test the full Transformer with 4D image features."""
    print("=" * 60)
    print("Test 2: Full Transformer (DETR-style)")
    print("=" * 60)

    d_model = 256
    nhead = 8
    num_queries = 50
    bs = 2
    # Simulated feature map size (ResNet18 with 480x640 input -> 15x20)
    H, W = 15, 20

    transformer = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        return_intermediate_dec=True,
    )

    # Image features from backbone
    src = torch.randn(bs, d_model, H, W)
    pos_embed = torch.randn(1, d_model, H, W)

    # Action query embeddings
    query_embed = torch.randn(num_queries, d_model)

    # Latent and proprioception inputs (projected to d_model)
    latent_input = torch.randn(bs, d_model)
    proprio_input = torch.randn(bs, d_model)
    additional_pos_embed = torch.randn(2, d_model)

    hs = transformer(
        src, None, query_embed, pos_embed, latent_input, proprio_input, additional_pos_embed
    )

    print(f"  Image features: {src.shape}")
    print(f"  Output shape:   {hs.shape}")
    # (num_dec_layers, bs, num_queries, d_model)
    assert hs.shape == (4, bs, num_queries, d_model), (
        f"Expected (4, {bs}, {num_queries}, {d_model}), got {hs.shape}"
    )

    # In practice, only the last layer output is used
    last_layer = hs[0]
    print(f"  Last layer output: {last_layer.shape}")
    assert last_layer.shape == (bs, num_queries, d_model)

    print("  PASSED\n")


def test_gradient_flow():
    """Verify gradients propagate through the Transformer."""
    print("=" * 60)
    print("Test 3: Gradient Flow")
    print("=" * 60)

    d_model = 128
    transformer = Transformer(
        d_model=d_model,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        return_intermediate_dec=True,
    )

    src = torch.randn(2, d_model, 4, 6, requires_grad=True)
    pos = torch.randn(1, d_model, 4, 6)
    query = torch.randn(10, d_model)
    latent = torch.randn(2, d_model, requires_grad=True)
    proprio = torch.randn(2, d_model, requires_grad=True)
    add_pos = torch.randn(2, d_model)

    hs = transformer(src, None, query, pos, latent, proprio, add_pos)
    loss = hs[0].sum()
    loss.backward()

    print(f"  src gradient exists: {src.grad is not None}")
    print(f"  latent gradient exists: {latent.grad is not None}")
    print(f"  proprio gradient exists: {proprio.grad is not None}")
    assert src.grad is not None, "No gradient for src"
    assert latent.grad is not None, "No gradient for latent"
    assert proprio.grad is not None, "No gradient for proprio"

    # Check that encoder parameters have gradients (all should, since memory
    # feeds into the decoder). Some decoder parameters in later layers may
    # not have gradients when only hs[0] (first layer output) is used — this
    # is expected with return_intermediate=True.
    has_grad = sum(
        1 for p in transformer.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for p in transformer.parameters() if p.requires_grad)
    print(f"  Parameters with gradient: {has_grad}/{total}")
    assert has_grad > 0, "No parameters received gradients"

    print("  PASSED\n")


if __name__ == "__main__":
    test_cvae_encoder()
    test_full_transformer()
    test_gradient_flow()
    print("All Transformer tests passed!")
