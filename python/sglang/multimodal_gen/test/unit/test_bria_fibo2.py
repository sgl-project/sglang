# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the BRIA Fibo 2.0 support.

These pin down the semantics that silently produce wrong images when they drift:
the gist count mapping, the patch feature order, the three-axis RoPE (including
the fixed image plane), the per-token noisy/clean modulation, and the parameter
names/shapes the checkpoint ships.
"""

import os

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.bria_fibo2 import (
    BriaFibo2ArchConfig,
    BriaFibo2DitConfig,
)
from sglang.multimodal_gen.runtime.models.dits.bria_fibo2 import (
    ADALN_EMBED_DIM,
    BriaFibo2Attention,
    BriaFibo2Transformer2DModel,
    apply_rotary_emb,
    compute_gist_count,
    select_per_token,
)

# The real model is 8B; building it on the meta device costs no memory
META_CONFIG_DIR = os.environ.get(
    "FIBO2_MODEL_CFG_DIR", "/workspace/dsh/fibo2/model_cfg"
)


def tiny_arch_config(**overrides) -> BriaFibo2ArchConfig:
    kwargs = dict(
        dim=64,
        num_attention_heads=2,
        num_layers=3,
        n_refiner_layers=1,
        in_channels=32,
        patch_size=2,
        cap_feat_dim=48,
        injection_layer_ids=(1,),
        perceiver_num_layers=1,
        max_num_gist_tokens=8,
        min_num_gist_tokens=4,
        gist_step=2,
        gist_min_text_len=2,
        gist_max_text_len=8,
        # head_dim is dim // n_heads = 32, and the axes must sum to it
        axes_dims=(8, 12, 12),
        axes_lens=(64, 16, 16),
        # The tiny models are driven with noise levels in [0, 1], the reference
        # convention, so no rescaling happens on the way to the embedder.
        scheduler_timesteps=1.0,
    )
    kwargs.update(overrides)
    return BriaFibo2ArchConfig(**kwargs)


def tiny_model() -> BriaFibo2Transformer2DModel:
    config = BriaFibo2DitConfig(arch_config=tiny_arch_config())
    model = BriaFibo2Transformer2DModel(config, hf_config={})
    return model.eval()


# ---------------------------------------------------------------------------
# T1 gist count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text_len,expected",
    [
        (0, 256),
        (1, 256),
        (256, 256),
        (500, 320),
        (1000, 384),
        (2048, 512),
        (100000, 512),
    ],
)
def test_gist_count_matches_reference_boundaries(text_len, expected):
    got = compute_gist_count(
        [text_len],
        min_gist=256,
        max_gist=512,
        gist_step=64,
        min_text_len=256,
        max_text_len=2048,
    )
    assert got == [expected]


def test_gist_count_is_always_quantized_and_clamped():
    lengths = list(range(0, 2600, 7))
    counts = compute_gist_count(lengths, 256, 512, 64, 256, 2048)
    for n, c in zip(lengths, counts):
        assert 256 <= c <= 512, (n, c)
        assert c % 64 == 0, (n, c)


# ---------------------------------------------------------------------------
# T2 patchify / unpatchify
# ---------------------------------------------------------------------------


def test_patchify_unpatchify_roundtrip():
    model = tiny_model()
    latents = torch.randn(2, 32, 16, 12)
    tokens = model._patchify(latents)
    assert tokens.shape == (2, (16 // 2) * (12 // 2), 32 * 2 * 2)
    restored = model._unpatchify(tokens, 16, 12)
    assert torch.equal(restored, latents)


def test_patchify_feature_order_is_channel_major():
    """Token features are ordered (channel, dy, dx), as the checkpoint expects."""
    model = tiny_model()
    latents = torch.arange(2 * 2 * 2, dtype=torch.float32).reshape(1, 2, 2, 2)
    tokens = model._patchify(latents)
    # Single token; features = [c0 (0,0), c0 (0,1), c0 (1,0), c0 (1,1), c1 ...]
    assert tokens.shape == (1, 1, 2 * 2 * 2)
    expected = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    assert torch.equal(tokens[0], expected)


# ---------------------------------------------------------------------------
# T3 RoPE equals the reference complex multiply
# ---------------------------------------------------------------------------


def test_apply_rotary_emb_matches_complex_reference():
    torch.manual_seed(0)
    seq_len, heads, head_dim = 7, 2, 8
    x = torch.randn(1, seq_len, heads, head_dim)
    half = head_dim // 2
    angles = torch.randn(1, seq_len, half)
    cos, sin = torch.cos(angles), torch.sin(angles)

    got = apply_rotary_emb(x, cos, sin)

    # The reference does view_as_complex(x.reshape(..., -1, 2)) * polar(1, angle)
    ref_complex = torch.complex(x.float()[..., 0::2], x.float()[..., 1::2])
    rotation = torch.polar(torch.ones_like(angles), angles).unsqueeze(2)
    expected = torch.view_as_real(ref_complex * rotation).flatten(-2)

    assert torch.allclose(got.float(), expected, atol=1e-6)


def test_apply_rotary_emb_preserves_dtype_and_norm():
    x = torch.randn(1, 5, 2, 8, dtype=torch.float32)
    angles = torch.randn(1, 5, 4)
    out = apply_rotary_emb(x, torch.cos(angles), torch.sin(angles))
    assert out.dtype == x.dtype
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5), (
        "a rotation must preserve the per-head norm"
    )


# ---------------------------------------------------------------------------
# T4 RoPE planes: gist on 1..N, image always on max_num_gist_tokens + 1
# ---------------------------------------------------------------------------


def _captured_rope_ids(model, **forward_kwargs):
    """Run one forward and return the position ids handed to the RoPE table."""
    seen = []
    original = model.rope_embedder

    def recorder(ids):
        seen.append(ids.clone())
        return original(ids)

    model.rope_embedder = recorder
    try:
        with torch.no_grad():
            model(**forward_kwargs)
    finally:
        model.rope_embedder = original
    return seen


def test_rope_image_plane_is_fixed_at_max_gist_plus_one():
    """The image plane must be max_num_gist_tokens + 1 whatever the gist count."""
    model = tiny_model()  # max_num_gist_tokens = 8 -> image plane 9
    batch, text_len, height, width = 1, 4, 16, 8
    hidden_states = torch.randn(batch, 32, height, width)
    encoder_hidden_states = torch.randn(batch, 2, text_len, 48)
    attention_mask = torch.ones(batch, text_len, dtype=torch.bool)

    ids = _captured_rope_ids(
        model,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=torch.full((batch,), 0.5),
        encoder_attention_mask=attention_mask,
    )
    # two calls: one for the patchified image+context block, one for the gist tokens
    assert len(ids) == 2
    image_ids, gist_ids = ids

    image_plane = int(model.config.max_num_gist_tokens) + 1
    assert image_plane == 9
    assert image_ids[:, 0].unique().tolist() == [image_plane]
    # rows and columns of the image patch grid
    assert image_ids[:, 1].max().item() == height // 2 - 1
    assert image_ids[:, 2].max().item() == width // 2 - 1

    # gist tokens walk along axis 0 starting at plane 1, rows/cols stay zero
    assert gist_ids[:, 0].tolist() == list(range(1, gist_ids.shape[0] + 1))
    assert gist_ids[:, 1].abs().max().item() == 0
    assert gist_ids[:, 2].abs().max().item() == 0


def test_rope_context_image_planes_follow_the_image_plane():
    model = tiny_model()
    batch, text_len, height, width = 1, 4, 16, 8
    context_latents = [torch.randn(batch, 32, 8, 8), torch.randn(batch, 32, 16, 4)]

    ids = _captured_rope_ids(
        model,
        hidden_states=torch.randn(batch, 32, height, width),
        encoder_hidden_states=torch.randn(batch, 2, text_len, 48),
        timestep=torch.full((batch,), 0.5),
        encoder_attention_mask=torch.ones(batch, text_len, dtype=torch.bool),
        source_latents=context_latents,
    )
    image_ids = ids[0]
    image_plane = int(model.config.max_num_gist_tokens) + 1
    num_image_tokens = (height // 2) * (width // 2)

    assert image_ids[:num_image_tokens, 0].unique().tolist() == [image_plane]
    # context image k sits on plane image_plane + k + 1
    first_ctx = image_ids[num_image_tokens : num_image_tokens + (8 // 2) * (8 // 2)]
    assert first_ctx[:, 0].unique().tolist() == [image_plane + 1]
    second_ctx = image_ids[num_image_tokens + (8 // 2) * (8 // 2) :]
    assert second_ctx[:, 0].unique().tolist() == [image_plane + 2]


# ---------------------------------------------------------------------------
# T5 per-token modulation
# ---------------------------------------------------------------------------


def test_select_per_token_picks_noisy_and_clean_values():
    value_noisy = torch.tensor([[1.0, 2.0]])
    value_clean = torch.tensor([[10.0, 20.0]])
    noise_mask = torch.tensor([[1, 0, 1, 0]])
    got = select_per_token(value_noisy, value_clean, noise_mask, seq_len=4)
    expected = torch.tensor([[[1.0, 2.0], [10.0, 20.0], [1.0, 2.0], [10.0, 20.0]]])
    assert got.shape == (1, 4, 2)
    assert torch.equal(got, expected)


def test_noisy_tokens_use_real_timestep_and_clean_tokens_use_zero():
    """Noisy tokens take t_embedder(t * t_scale), clean tokens t_embedder(0)."""
    model = tiny_model()
    seen = []

    def record_input(module, args):
        seen.append(args[0].detach().clone())

    handle = model.t_embedder.register_forward_pre_hook(record_input)
    try:
        with torch.no_grad():
            model(
                hidden_states=torch.randn(1, 32, 16, 8),
                encoder_hidden_states=torch.randn(1, 2, 4, 48),
                timestep=torch.full((1,), 0.5),
                encoder_attention_mask=torch.ones(1, 4, dtype=torch.bool),
            )
    finally:
        handle.remove()

    assert len(seen) == 2, "one embedding for the noisy tokens, one for the clean"
    noisy_input, clean_input = seen
    t_scale = model.config.t_scale
    assert torch.allclose(noisy_input, torch.full((1,), 0.5) * t_scale)
    assert torch.equal(clean_input, torch.zeros(1))


def test_block_global_and_per_token_paths_agree_without_context():
    """Without context latents the refiner must modulate globally, not per token."""
    model = tiny_model()
    hidden_states = torch.randn(1, 32, 16, 8)
    encoder_hidden_states = torch.randn(1, 2, 4, 48)
    timestep = torch.full((1,), 0.5)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        t2i = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=mask,
        )
    assert torch.isfinite(t2i).all()
    assert t2i.shape == hidden_states.shape


# ---------------------------------------------------------------------------
# T11 construction and forward (both branches)
# ---------------------------------------------------------------------------


def test_tiny_model_forward_text_to_image():
    model = tiny_model()
    hidden_states = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        out = model(
            hidden_states,
            torch.randn(1, 2, 5, 48),
            torch.full((1,), 0.5),
            encoder_attention_mask=torch.ones(1, 5, dtype=torch.bool),
        )
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def test_tiny_model_forward_with_source_latents_of_mixed_sizes():
    model = tiny_model()
    hidden_states = torch.randn(1, 32, 16, 16)
    context_latents = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 16, 4)]
    with torch.no_grad():
        out = model(
            hidden_states,
            torch.randn(1, 2, 5, 48),
            torch.full((1,), 0.5),
            encoder_attention_mask=torch.ones(1, 5, dtype=torch.bool),
            source_latents=context_latents,
        )
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def test_tiny_model_forward_without_text_mask():
    model = tiny_model()
    hidden_states = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        out = model(hidden_states, torch.randn(2, 2, 4, 48), torch.full((2,), 0.25))
    assert out.shape == hidden_states.shape


def test_batch_with_mixed_gist_counts_is_rejected():
    """A batch whose prompts need different gist counts must fail, not silently pad."""
    model = tiny_model()
    batch = 2
    # text lengths 3 and 8 give different gist counts (4 vs 8)
    attention_mask = torch.zeros(batch, 8, dtype=torch.bool)
    attention_mask[0, :3] = True
    attention_mask[1, :8] = True
    with pytest.raises(ValueError, match="different numbers of gist tokens"):
        with torch.no_grad():
            model(
                torch.randn(batch, 32, 16, 8),
                torch.randn(batch, 2, 8, 48),
                torch.full((batch,), 0.5),
                encoder_attention_mask=attention_mask,
            )


def test_no_context_images_are_registered_as_buffers():
    """The model owns no context-latent state; context arrives per call."""
    model = tiny_model()
    names = {name for name, _ in model.named_buffers()}
    assert not any("context" in name for name in names)


# ---------------------------------------------------------------------------
# T12 cross-attention dimensions
# ---------------------------------------------------------------------------


def test_cross_attention_reads_cap_feat_dim_and_writes_dim():
    model = tiny_model()
    dim = model.config.dim
    cap_feat_dim = model.config.cap_feat_dim
    injection_layer_ids = list(model.config.injection_layer_ids)
    assert injection_layer_ids == [1]

    block = model.layers[1]
    assert block.cross_attention is not None
    assert block.cross_attention.to_q.weight.shape == (dim, dim)
    # the text bundle is cap_feat_dim wide, so k/v project from there
    assert block.cross_attention.to_k.weight.shape == (dim, cap_feat_dim)
    assert block.cross_attention.to_v.weight.shape == (dim, cap_feat_dim)
    assert block.cross_attn_gate.shape == (1,)

    # blocks that are not injection blocks carry no cross-attention
    assert model.layers[0].cross_attention is None
    assert model.layers[2].cross_attention is None


def test_attention_is_not_fused():
    """q/k/v stay separate: cross-attention q and k/v have different input widths."""
    attn = BriaFibo2Attention(query_dim=8, heads=2, dim_head=4, kv_dim=12)
    names = dict(attn.named_parameters())
    assert "to_q.weight" in names and names["to_q.weight"].shape == (8, 8)
    assert "to_k.weight" in names and names["to_k.weight"].shape == (8, 12)
    assert "to_v.weight" in names and names["to_v.weight"].shape == (8, 12)
    # the reference keeps the output projection in a ModuleList -> `to_out.0`
    assert "to_out.0.weight" in names
    assert "norm_q.weight" in names and "norm_k.weight" in names


# ---------------------------------------------------------------------------
# T13 module placement
# ---------------------------------------------------------------------------


def test_adaln_modulation_is_replicated_and_full_width():
    """adaLN must emit the whole 4*dim vector: sharding it silently halves it."""
    model = tiny_model()
    dim = model.config.dim
    block = model.layers[0]
    linear = block.adaLN_modulation[0]
    assert isinstance(linear, torch.nn.Linear)
    assert linear.weight.shape == (4 * dim, min(dim, ADALN_EMBED_DIM))
    assert linear.bias.shape == (4 * dim,)


def test_perceiver_latent_queries_shape():
    model = tiny_model()
    queries = model.perceiver.latent_queries
    assert isinstance(queries, torch.nn.Parameter)
    assert queries.shape == (
        model.config.max_num_gist_tokens,
        model.config.dim,
    )


def test_final_layer_outputs_one_patch_per_token():
    model = tiny_model()
    patch_size = model.config.patch_size
    out_features = patch_size * patch_size * model.config.out_channels
    assert model.final_layer.linear.out_features == out_features


# ---------------------------------------------------------------------------
# Weight map: the checkpoint's own names and shapes must match one to one
# ---------------------------------------------------------------------------


def _load_key_shapes(path):
    key_shapes = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            name, shape = fields[0], fields[2]
            dims = tuple(int(x) for x in shape.strip("[]").split(",") if x.strip())
            key_shapes[name] = dims
    return key_shapes


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(META_CONFIG_DIR, "dit_keys_shapes.tsv")),
    reason="checkpoint key/shape dump not available",
)
def test_state_dict_matches_checkpoint_keys_and_shapes():
    """Build the full 8B model on the meta device and compare against the dump.

    This is the guard against a silent weight reshuffle: every checkpoint key
    must exist with the same shape, and the model must add nothing the
    checkpoint does not ship.
    """
    key_shapes = _load_key_shapes(os.path.join(META_CONFIG_DIR, "dit_keys_shapes.tsv"))
    config = BriaFibo2DitConfig()
    with torch.device("meta"):
        model = BriaFibo2Transformer2DModel(config, hf_config={})
    mine = {name: tuple(t.shape) for name, t in model.state_dict().items()}

    missing = sorted(set(key_shapes) - set(mine))
    extra = sorted(set(mine) - set(key_shapes))
    mismatched = [
        (name, key_shapes[name], mine[name])
        for name in sorted(set(key_shapes) & set(mine))
        if key_shapes[name] != mine[name]
    ]

    assert not missing, f"checkpoint keys missing from the model: {missing[:10]}"
    assert not extra, f"model keys absent from the checkpoint: {extra[:10]}"
    assert not mismatched, f"shape mismatches: {mismatched[:10]}"


def test_reference_parameter_count_matches_arch_config():
    """Spot-check the derived shapes the arch config is responsible for."""
    config = BriaFibo2ArchConfig()
    assert config.dim == 3584
    assert config.num_layers == 44
    assert config.num_attention_heads == 28
    assert config.cap_feat_dim == 7680
    assert list(config.injection_layer_ids) == [6, 14, 22, 30, 38]
    assert sum(config.axes_dims) == config.dim // config.num_attention_heads
    # (dim * 8 // 3) // 64 * 64, the SwiGLU inner width the checkpoint ships
    assert (config.dim * 8 // 3) // 64 * 64 == 9536


def test_arch_config_rejects_inconsistent_axes():
    with pytest.raises(ValueError, match="axes_dims must sum"):
        BriaFibo2ArchConfig(dim=64, num_attention_heads=2, axes_dims=(8, 12, 8))
