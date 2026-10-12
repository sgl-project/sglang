# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
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
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
    _attn_fast_compatible,
    _fused_qknorm_rope,
    _install_fast_attention,
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
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    apply_rotary_pos_emb_qk,
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
    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
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
        _install_fast_attention([attention], gate)
        assert torch.equal(attention(hidden, rotary), reference)
        gate.enabled = True
        fused = attention(hidden, rotary)
    assert attention._sgl_unit_weight is not None, "fused qk-norm+RoPE did not run"
    assert attention._sgl_cudnn_failed is False, "cuDNN SDPA fell back"
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=1e-2)


@requires_cuda
def test_vit_rope_batched_tiles_match_per_tile():
    """Stacked decoder tiles share one RoPE row; the fused (quality) and native
    (exact) RoPE kernels over the flattened batch are bit-identical to one
    tile at a time."""
    device, dtype = torch.device("cuda"), torch.float16
    tiles, heads, dim_head, rope_dim = 3, 4, 64, 48
    pos_embed = RotaryEmbeddingND(rope_dim, 100.0, n_dim=3, use_angle=True).to(device)
    ids = create_token_ids((2, 4, 4), device, dtype)
    rotary = prepare_rotary_pos_emb(pos_embed(ids), dtype=dtype)
    batched_rotary = prepare_rotary_pos_emb(pos_embed(ids), dtype=dtype, batch=tiles)
    generator = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(
        (tiles, ids.shape[1], heads, 3 * dim_head), generator=generator
    ).to(device=device, dtype=dtype)
    attn = SimpleNamespace(
        dim_head=dim_head, _sgl_unit_weight=None, norm_q=SimpleNamespace(eps=1e-5)
    )

    per_tile = qkv.clone()
    for tile in range(tiles):
        query, key, _ = per_tile[tile : tile + 1].chunk(3, dim=-1)
        assert _fused_qknorm_rope(attn, query, key, rotary)
    query, key, _ = qkv.chunk(3, dim=-1)
    assert _fused_qknorm_rope(attn, query, key, batched_rotary)
    assert torch.equal(qkv, per_tile)

    query, key, _ = qkv.chunk(3, dim=-1)
    batched = apply_rotary_pos_emb_qk(query, key, batched_rotary)
    for tile in range(tiles):
        query, key, _ = qkv[tile : tile + 1].chunk(3, dim=-1)
        for got, want in zip(batched, apply_rotary_pos_emb_qk(query, key, rotary)):
            assert torch.equal(got[tile : tile + 1], want)


def test_audio_vae_attention_defaults_to_local_sdpa_and_allows_fa():
    class RecordingFA(nn.Module):
        backend = AttentionBackendEnum.FA
        dtype = torch.bfloat16

        def forward(self, query, key, value):
            self.input_dtype = query.dtype
            return query

    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae"
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_audio_snake_first_call_matches_repeated_calls():
    # a fresh process prevents earlier tests from warming a profiling JIT graph
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import torch
                from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import Snake1d

                torch.manual_seed(42)
                activation = Snake1d(64).cuda().eval()
                with torch.inference_mode():
                    activation.alpha.uniform_(0.1, 2.0)
                    x = torch.randn(2, 64, 4096, device="cuda")
                    original_x = x.clone()
                    original_alpha = activation.alpha.clone()
                    first = activation(x)
                    for _ in range(4):
                        torch.testing.assert_close(activation(x), first, rtol=0, atol=0)
                    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
                    torch.testing.assert_close(activation.alpha, original_alpha, rtol=0, atol=0)
                """),
        ],
        check=True,
        timeout=120,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_neox_rope_matches_per_tile_rows():
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
        prepare_rotary_pos_emb,
    )

    cos = torch.randn(1, 5, 1, 8, device="cuda", dtype=torch.float16)
    sin = torch.randn(1, 5, 1, 8, device="cuda", dtype=torch.float16)
    packed = prepare_rotary_pos_emb((cos, sin), dtype=torch.float16)
    query = torch.randn(3, 5, 2, 16, device="cuda", dtype=torch.float16)
    key = torch.randn(3, 5, 2, 16, device="cuda", dtype=torch.float16)
    expanded = (
        packed[0].expand(3, -1, -1, -1),
        packed[1].expand(3, -1, -1, -1),
        *packed[2:],
    )
    batched_q, batched_k = apply_rotary_pos_emb_qk(query.clone(), key.clone(), expanded)
    rows_q = []
    rows_k = []
    for index in range(3):
        row_q, row_k = apply_rotary_pos_emb_qk(
            query[index : index + 1],
            key[index : index + 1],
            packed,
        )
        rows_q.append(row_q)
        rows_k.append(row_k)

    torch.testing.assert_close(batched_q, torch.cat(rows_q, dim=0))
    torch.testing.assert_close(batched_k, torch.cat(rows_k, dim=0))


def _h3_shaped_decoder():
    decoder = ViT3DDecoder(
        patch_size=1,
        patch_size_t=1,
        in_channels=4,
        out_channels=3,
        num_layers=2,
        heads=4,
        dim_head=64,
        norm_type="rms_norm",
        norm_affine=True,
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
        ffn_activation_fn="silu",
        ffn_use_gated=True,
        rope_dim_ratio=0.75,
        rope_theta=100.0,
        num_register_tokens=0,
    ).cuda()
    decoder.eval()
    for block in decoder.transformer_blocks:
        block.scale1.data.normal_(0, 0.1)
        block.scale2.data.normal_(0, 0.1)
    decoder.prepare_autocast_linear_weights(torch.float16)
    return decoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_decoder_matches_unfused_forward():
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    decoder = _h3_shaped_decoder()
    latent = torch.randn(2, 4, 1, 2, 2, device="cuda", dtype=torch.float16)
    previous = os.environ.get("MINIMAX_H3_VAE_DECODER_FUSED_NORM")
    from sglang.kernels.ops import diffusion as diffusion_ops

    try:
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.float16),
            set_forward_context(current_timestep=0, attn_metadata=None),
            mock.patch.object(
                diffusion_ops,
                "h3_vae_scale_add_rmsnorm",
                wraps=diffusion_ops.h3_vae_scale_add_rmsnorm,
            ) as scale_add,
            mock.patch.object(
                diffusion_ops,
                "h3_vae_qk_rmsnorm_rope",
                wraps=diffusion_ops.h3_vae_qk_rmsnorm_rope,
            ) as qk_rope,
        ):
            os.environ["MINIMAX_H3_VAE_DECODER_FUSED_NORM"] = "0"
            eager = decoder(latent)
            os.environ["MINIMAX_H3_VAE_DECODER_FUSED_NORM"] = "1"
            fused = decoder(latent)
            assert scale_add.call_count == 3
            assert qk_rope.call_count == 2
    finally:
        if previous is None:
            os.environ.pop("MINIMAX_H3_VAE_DECODER_FUSED_NORM", None)
        else:
            os.environ["MINIMAX_H3_VAE_DECODER_FUSED_NORM"] = previous

    torch.testing.assert_close(fused, eager, rtol=1e-3, atol=1e-3)
