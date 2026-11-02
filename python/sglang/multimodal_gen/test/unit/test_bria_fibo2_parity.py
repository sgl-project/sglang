# SPDX-License-Identifier: Apache-2.0
"""Numerical parity of the in-tree Fibo 2.0 transformer against diffusers.

The strongest check we can run without weights or a GPU: build both models from
a tiny configuration, copy diffusers' parameters into ours by name, and compare
the outputs element-wise. Because the whole forward runs, a match proves far more
than shape assertions - patchify feature order, the three-axis RoPE planes and
rotation direction, per-token noisy/clean modulation, the Perceiver and gist
construction, the injection blocks' cross-attention wiring, the ``[gist | image |
context]`` token layout, the final layer reading only image tokens, and the
unpatchify flip are all exercised at once.

The test skips when the installed diffusers predates the model (0.42.0).
"""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")


def _ref_transformer_cls():
    try:
        from diffusers.models.transformers.transformer_bria_fibo2 import (
            BriaFibo2Transformer2DModel as RefModel,
        )
    except Exception as exc:  # model not in this diffusers version
        pytest.skip(f"diffusers has no BriaFibo2 transformer: {exc}")
    return RefModel


# Small enough to run in a second on CPU, large enough to touch every branch:
# one injection block, a Perceiver layer, a two-axis-value text length.
REFERENCE_KWARGS = dict(
    in_channels=32,
    patch_size=2,
    dim=64,
    n_layers=3,
    n_refiner_layers=1,
    n_heads=2,
    norm_eps=1e-5,
    cap_feat_dim=48,
    injection_layer_ids=(1,),
    perceiver_num_layers=1,
    min_num_gist_tokens=4,
    max_num_gist_tokens=8,
    gist_step=4,
    gist_min_text_len=2,
    gist_max_text_len=8,
    rope_theta=256.0,
    t_scale=1000.0,
    # the sum must be the attention head dim, 64 // 2
    axes_dims=(8, 12, 12),
    axes_lens=(32, 8, 8),
)


def _build_reference():
    torch.manual_seed(0)
    return _ref_transformer_cls()(**REFERENCE_KWARGS).eval()


def _build_ours():
    from sglang.multimodal_gen.configs.models.dits.bria_fibo2 import (
        BriaFibo2DitConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits.bria_fibo2 import (
        BriaFibo2Transformer2DModel,
    )

    torch.manual_seed(0)
    config = BriaFibo2DitConfig()
    arch = config.arch_config
    # Reference convention: noise levels come in as [0, 1] and the embedder
    # scales them by `t_scale` itself.
    arch.scheduler_timesteps = 1.0
    arch.in_channels = REFERENCE_KWARGS["in_channels"]
    arch.patch_size = REFERENCE_KWARGS["patch_size"]
    arch.dim = REFERENCE_KWARGS["dim"]
    arch.num_layers = REFERENCE_KWARGS["n_layers"]
    arch.n_refiner_layers = REFERENCE_KWARGS["n_refiner_layers"]
    arch.num_attention_heads = REFERENCE_KWARGS["n_heads"]
    arch.norm_eps = REFERENCE_KWARGS["norm_eps"]
    arch.cap_feat_dim = REFERENCE_KWARGS["cap_feat_dim"]
    arch.injection_layer_ids = REFERENCE_KWARGS["injection_layer_ids"]
    arch.perceiver_num_layers = REFERENCE_KWARGS["perceiver_num_layers"]
    arch.min_num_gist_tokens = REFERENCE_KWARGS["min_num_gist_tokens"]
    arch.max_num_gist_tokens = REFERENCE_KWARGS["max_num_gist_tokens"]
    arch.gist_step = REFERENCE_KWARGS["gist_step"]
    arch.gist_min_text_len = REFERENCE_KWARGS["gist_min_text_len"]
    arch.gist_max_text_len = REFERENCE_KWARGS["gist_max_text_len"]
    arch.rope_theta = REFERENCE_KWARGS["rope_theta"]
    arch.t_scale = REFERENCE_KWARGS["t_scale"]
    arch.axes_dims = REFERENCE_KWARGS["axes_dims"]
    arch.axes_lens = REFERENCE_KWARGS["axes_lens"]
    arch.__post_init__()
    return BriaFibo2Transformer2DModel(config, hf_config={}).eval()


def _copy_reference_weights(reference, ours) -> None:
    """Every reference key must exist with the same shape, then be copied over."""
    ref_state = reference.state_dict()
    our_state = ours.state_dict()

    missing = sorted(set(ref_state) - set(our_state))
    extra = sorted(set(our_state) - set(ref_state))
    shape_mismatch = [
        name
        for name in set(ref_state) & set(our_state)
        if ref_state[name].shape != our_state[name].shape
    ]
    assert not missing, f"reference keys absent from our model: {missing[:5]}"
    assert not extra, f"our keys absent from the reference: {extra[:5]}"
    assert not shape_mismatch, f"shape mismatches: {shape_mismatch[:5]}"

    with torch.no_grad():
        for name, value in ref_state.items():
            our_state[name].copy_(value)


def _assert_parity(tag: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    assert expected.shape == actual.shape, (tag, expected.shape, actual.shape)
    expected_flat = expected.flatten().double()
    actual_flat = actual.flatten().double()
    max_abs = float((expected_flat - actual_flat).abs().max())
    cosine = float(
        (expected_flat * actual_flat).sum()
        / (expected_flat.norm() * actual_flat.norm())
    )
    # fp32 on CPU: anything beyond rounding means a real semantic difference
    assert max_abs < 1e-4, f"{tag}: max|delta| = {max_abs:.3e}"
    assert cosine > 0.99999, f"{tag}: cosine = {cosine:.9f}"


@pytest.fixture(scope="module")
def models():
    reference = _build_reference()
    ours = _build_ours()
    _copy_reference_weights(reference, ours)
    return reference, ours


def test_transformer_matches_diffusers_text_to_image(models):
    reference, ours = models
    batch, channels, height, width = 1, 32, 16, 16
    text_len = 6
    cap_feat_dim = REFERENCE_KWARGS["cap_feat_dim"]
    num_bundles = len(REFERENCE_KWARGS["injection_layer_ids"]) + 1

    torch.manual_seed(1)
    hidden_states = torch.randn(batch, channels, height, width)
    encoder_hidden_states = torch.randn(batch, num_bundles, text_len, cap_feat_dim)
    attention_mask = torch.ones(batch, text_len, dtype=torch.bool)
    timestep = torch.tensor([0.5])

    with torch.no_grad():
        expected = reference(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False,
        )[0]
        actual = ours(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=attention_mask,
        )
    _assert_parity("text-to-image", expected, actual)


def test_transformer_matches_diffusers_with_context_latents(models):
    """Editing conditions on clean latents of reference images of any size."""
    reference, ours = models
    batch, channels, height, width = 1, 32, 16, 16
    text_len = 6
    cap_feat_dim = REFERENCE_KWARGS["cap_feat_dim"]
    num_bundles = len(REFERENCE_KWARGS["injection_layer_ids"]) + 1

    torch.manual_seed(2)
    hidden_states = torch.randn(batch, channels, height, width)
    encoder_hidden_states = torch.randn(batch, num_bundles, text_len, cap_feat_dim)
    attention_mask = torch.ones(batch, text_len, dtype=torch.bool)
    timestep = torch.tensor([0.5])
    # two reference images, deliberately different sizes
    reference_latents = [
        torch.randn(batch, channels, height, width),
        torch.randn(batch, channels, height // 2, width),
    ]

    with torch.no_grad():
        expected = reference(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            context_latents=reference_latents,
            return_dict=False,
        )[0]
        actual = ours(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=attention_mask,
            source_latents=reference_latents,
        )
    _assert_parity("editing", expected, actual)


def test_transformer_matches_diffusers_without_text_mask(models):
    """An unpadded prompt runs without a mask; gist count comes from the length."""
    reference, ours = models
    batch, channels, height, width = 1, 32, 16, 16
    text_len = 5
    cap_feat_dim = REFERENCE_KWARGS["cap_feat_dim"]
    num_bundles = len(REFERENCE_KWARGS["injection_layer_ids"]) + 1

    torch.manual_seed(3)
    hidden_states = torch.randn(batch, channels, height, width)
    encoder_hidden_states = torch.randn(batch, num_bundles, text_len, cap_feat_dim)
    timestep = torch.tensor([0.25])

    with torch.no_grad():
        expected = reference(
            hidden_states, timestep, encoder_hidden_states, return_dict=False
        )[0]
        actual = ours(hidden_states, encoder_hidden_states, timestep)
    _assert_parity("no text mask", expected, actual)
