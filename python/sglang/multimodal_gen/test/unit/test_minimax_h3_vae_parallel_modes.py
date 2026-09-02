# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

from unittest import mock

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import VaeFastPathGate
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import (
    CausalAttention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.base_module import (
    RotaryEmbeddingND,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    create_token_ids,
    prepare_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused qk-norm+RoPE kernel needs CUDA"
)


def _init_kwargs(config: MiniMaxH3VideoVAEConfig):
    with mock.patch.object(
        AutoencoderKLLegacy, "__init__", autospec=True, return_value=None
    ) as init:
        model = MiniMaxH3VideoVAE(config)
    return model, init.call_args.kwargs


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "auto",
        "tiled",
    ],
)
def test_decode_mode_uses_released_tiled_recipe(mode):
    config = (
        MiniMaxH3VideoVAEConfig()
        if mode is None
        else MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    )
    model, kwargs = _init_kwargs(config)

    assert model.parallel_decode_mode == "tiled"
    assert kwargs["decoder_tiling"] is True
    assert kwargs["parallel_tiling"] is True
    assert kwargs["decoder_parallel"] is False


@pytest.mark.parametrize("mode", ["spatial", "spatial_shard", "patch"])
def test_unvalidated_decode_modes_are_rejected(mode):
    config = MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    with pytest.raises(ValueError, match="use tiled"):
        config.resolved_parallel_decode_mode()


def test_vit_attention_uses_local_usp_backend_dispatch():
    module = (
        "sglang.multimodal_gen.runtime.models.vaes." "minimax_h3_video_vae.attention"
    )
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(f"{module}.USPAttention", autospec=True) as usp_attention,
    ):
        Attention(heads=2, dim_head=64)

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }


def test_vit_qk_norm_supports_affine_free_rmsnorm():
    norm = nn.RMSNorm(64, elementwise_affine=False)
    hidden_states = torch.randn(1, 2, 2, 64)

    output = _apply_qk_norm(norm, hidden_states)

    assert output.shape == hidden_states.shape


@requires_cuda
def test_vit_fast_path_is_gated_and_matches_reference():
    """Gate closed: bit-identical to the original forward. Gate open: the fused
    qk-norm+RoPE kernel and cuDNN SDPA run and match to rounding level."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
        _attn_fast_compatible,
        install_qknorm_rope,
    )

    device = torch.device("cuda")
    dtype = torch.float16
    heads, dim_head, rope_dim = 4, 64, 48
    attention = Attention(
        heads=heads, dim_head=dim_head, qk_norm_type="rms_norm", eps=1e-5
    ).to(device=device, dtype=dtype)
    assert _attn_fast_compatible(attention)

    pos_embed = RotaryEmbeddingND(rope_dim, 100.0, n_dim=3, use_angle=True).to(device)
    ids = create_token_ids((2, 4, 4), device, dtype)
    ids = torch.cat([ids, torch.zeros((1, 5, 3), device=device, dtype=dtype)], 1)
    rotary = prepare_rotary_pos_emb(pos_embed(ids), dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(0)
    hidden = torch.randn((1, ids.shape[1], heads * dim_head), generator=generator)
    hidden = hidden.to(device=device, dtype=dtype)

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=dtype),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        reference = attention(hidden, rotary)
        gate = VaeFastPathGate()
        install_qknorm_rope([attention], gate)
        assert torch.equal(attention(hidden, rotary), reference)
        gate.enabled = True
        fused = attention(hidden, rotary)
    assert attention._sgl_unit_weight is not None
    assert attention._sgl_cudnn_failed is False
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=1e-2)


def test_audio_vae_attention_defaults_to_local_sdpa_and_allows_fa():
    class RecordingFA(nn.Module):
        backend = AttentionBackendEnum.FA
        dtype = torch.bfloat16

        def forward(self, query, key, value):
            self.input_dtype = query.dtype
            return query

    module = (
        "sglang.multimodal_gen.runtime.models.vaes." "minimax_h3_audio_vae.audio_vae"
    )
    recording_fa = RecordingFA()
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(
            f"{module}.USPAttention", autospec=True, return_value=recording_fa
        ) as usp_attention,
    ):
        attention = CausalAttention(in_dim=64, out_dim=32, num_heads=2)
        output = attention(torch.randn(1, 4, 64))

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["causal"] is True
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }
    assert recording_fa.input_dtype == torch.bfloat16
    assert output.dtype == torch.float32
