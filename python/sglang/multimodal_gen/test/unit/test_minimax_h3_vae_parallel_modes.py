# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

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
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    attention as vit_attention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
    _sdpa_attention,
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
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

# torch.cuda.is_available() is also true on ROCm, where these JIT/Triton
# kernels do not compile; gate on the platform instead.
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fused VAE kernels need CUDA"
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gfx950_vit_attention_takes_flash_for_half_and_math_for_fp32():
    """gfx950's fused SDPA corrupts the decoder: half inputs on the GPU
    (autocast included) go to aiter's flash kernel as [B, S, H, D], FP32 to
    the math replica."""
    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
    q, k, v = (torch.randn(1, 5, 2, 8, device="cuda") for _ in range(3))
    flash = mock.Mock(side_effect=lambda q, k, v, **_: torch.full_like(q, 7.0))
    with (
        mock.patch(f"{module}._FORCE_ROCM_MATH_SDPA", True),
        mock.patch(f"{module}._rocm_flash_attn", flash),
    ):
        out_half = _sdpa_attention(q.half(), k.half(), v.half())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _sdpa_attention(q, k, v)
        out_fp32 = _sdpa_attention(q, k, v)

    assert [c.args[0].dtype for c in flash.call_args_list] == [
        torch.float16,
        torch.bfloat16,
    ]
    assert flash.call_args.args[0].shape == q.shape
    assert flash.call_args.kwargs == {"dropout_p": 0.0, "causal": False}
    assert torch.equal(out_half, torch.full_like(out_half, 7.0))
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, k, v))
    ).transpose(1, 2)
    torch.testing.assert_close(out_fp32, ref)


def test_gfx950_vit_attention_keeps_the_math_replica_on_cpu():
    """aiter's kernel is GPU-only: a CPU decode under BF16 autocast stays on
    the math replica."""
    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
    q, k, v = (torch.randn(1, 5, 2, 8) for _ in range(3))
    flash = mock.Mock()
    with (
        mock.patch(f"{module}._FORCE_ROCM_MATH_SDPA", True),
        mock.patch(f"{module}._rocm_flash_attn", flash),
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        out = _sdpa_attention(q, k, v)
        math = vit_attention._math_sdpa(*(t.transpose(1, 2) for t in (q, k, v)))

    flash.assert_not_called()
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, math.transpose(1, 2))


@pytest.mark.skipif(
    vit_attention._rocm_flash_attn is None, reason="requires aiter on gfx950"
)
def test_gfx950_vit_flash_attention_matches_math_at_decoder_shape():
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 1797, 32, 64, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    flash = _sdpa_attention(q, k, v)
    with mock.patch.object(vit_attention, "_rocm_flash_attn", None):
        math = _sdpa_attention(q, k, v)
    # both accumulate in FP32; they differ by the FP16 rounding of the output
    torch.testing.assert_close(flash, math, atol=2e-3, rtol=2e-3)


def test_vit_qk_norm_supports_affine_free_rmsnorm():
    norm = nn.RMSNorm(64, elementwise_affine=False)
    hidden_states = torch.randn(1, 2, 2, 64)

    output = _apply_qk_norm(norm, hidden_states)

    assert output.shape == hidden_states.shape


@requires_cuda
@pytest.mark.parametrize(
    "shape,rope_dim,dtype",
    [((2, 4, 4), 48, torch.float16), ((8, 8), 32, torch.bfloat16)],
)
def test_vit_fast_path_is_gated_and_matches_reference(shape, rope_dim, dtype):
    """Gate closed: bit-identical to the original forward. Gate open: the fused
    qk-norm+RoPE kernel and cuDNN SDPA run and match to rounding level."""
    device = torch.device("cuda")
    heads, dim_head = 4, 64
    attention = Attention(
        heads=heads, dim_head=dim_head, qk_norm_type="rms_norm", eps=1e-5
    ).to(device=device, dtype=dtype)
    assert _attn_fast_compatible(attention)

    n_dim = len(shape)
    pos_embed = RotaryEmbeddingND(rope_dim, 100.0, n_dim=n_dim, use_angle=True).to(
        device
    )
    ids = create_token_ids(shape, device, dtype)
    ids = torch.cat([ids, torch.zeros((1, 5, n_dim), device=device, dtype=dtype)], 1)
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
        gate.enabled = False
        assert torch.equal(attention(hidden, rotary), reference)
    assert attention._sgl_unit_weight is not None, "fused qk-norm+RoPE did not run"
    assert attention._sgl_cudnn_failed is False, "cuDNN SDPA fell back"
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize(
    "shape,rope_dim,dtype",
    [((2, 4, 4), 48, torch.float16), ((8, 8), 32, torch.bfloat16)],
)
def test_vit_rope_batched_tiles_match_per_tile(shape, rope_dim, dtype):
    """Stacked decoder tiles share one RoPE row; the fused (quality) and native
    (exact) RoPE kernels over the flattened batch are bit-identical to one
    tile at a time."""
    device = torch.device("cuda")
    tiles, heads, dim_head = 3, 4, 64
    pos_embed = RotaryEmbeddingND(rope_dim, 100.0, n_dim=len(shape), use_angle=True).to(
        device
    )
    ids = create_token_ids(shape, device, dtype)
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
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm conv1d replica")
@pytest.mark.parametrize(
    "channels, taps, dilation", [(8, 11, 5), (16, 7, 3), (64, 3, 1)]
)
def test_rocm_bigvgan_conv1d_replays_fp64_naive_order(channels, taps, dilation):
    """The Triton conv1d sums x * w in FP64, input channel outer and tap inner,
    then rounds once; an FP64 loop in that order gives the same bytes, and the
    gated BigVGAN conv returns the eager conv's bytes either way."""
    from torch.nn.utils.parametrizations import weight_norm

    from sglang.kernels.ops.diffusion import conv1d_fp64acc
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae import (
        bigvgan,
    )

    torch.manual_seed(0)
    padding = dilation * (taps - 1) // 2
    x = torch.randn(2, channels, 4099, device="cuda")
    conv = weight_norm(
        nn.Conv1d(channels, channels, taps, padding=padding, dilation=dilation)
    ).cuda()
    w = conv.weight.detach()

    xp = nn.functional.pad(x, (padding, padding)).double()
    ref = torch.zeros(2, channels, x.shape[-1], dtype=torch.float64, device="cuda")
    for ic in range(channels):
        for tap in range(taps):
            start = tap * dilation
            window = xp[:, ic : ic + 1, start : start + x.shape[-1]]
            ref += window * w[None, :, ic, tap, None].double()
    out = conv1d_fp64acc(x, w, padding=padding, dilation=dilation)
    assert torch.equal(out, ref.float())

    with torch.inference_mode():
        assert torch.equal(bigvgan._conv1d(conv, x), conv(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm conv replica")
@pytest.mark.parametrize("fp64, taps_ascending", [(True, True), (False, False)])
def test_rocm_depthwise_conv_transpose1d_replays_miopen_orders(fp64, taps_ascending):
    """The naive solver's FP64 sum (taps ascending) and the col2im FP32 sum of
    FP32 products (taps descending), each against a per-tap loop in that order."""
    from sglang.kernels.ops.diffusion import conv_transpose1d_depthwise

    torch.manual_seed(0)
    x = torch.randn(2, 16, 4099, device="cuda")
    w = torch.randn(16, 1, 12, device="cuda")
    stride = 2
    dtype = torch.float64 if fp64 else torch.float32
    ref = torch.zeros(2, 16, 4098 * stride + 12, dtype=dtype, device="cuda")
    for tap in range(12) if taps_ascending else reversed(range(12)):
        ref[..., tap : tap + 4098 * stride + 1 : stride] += x.to(dtype) * w[
            None, :, 0, tap, None
        ].to(dtype)
    out = conv_transpose1d_depthwise(
        x, w, stride=stride, fp64=fp64, taps_ascending=taps_ascending
    )
    assert torch.equal(out, ref.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm conv replica")
@pytest.mark.parametrize("channels", [8, 64, 256])
def test_rocm_alias_free_resampling_keeps_eager_bytes(channels):
    """Each depthwise filter shape returns the eager conv's bytes, through the
    replica of MIOpen's order it matched at first sight, or eager."""
    from sglang.multimodal_gen.runtime.models.vaes import alias_free

    torch.manual_seed(0)
    x = torch.randn(2, channels, 20261, device="cuda")
    up = alias_free.UpSample1d(2, 12).cuda()
    down = alias_free.DownSample1d(2, 12).cuda()
    with torch.inference_mode():
        with mock.patch.object(
            alias_free.current_platform, "is_rocm", return_value=False
        ):
            eager = (up(x), down(x))
        alias_free._ROCM_DEPTHWISE_REPLICA.clear()
        for _ in range(2):  # first sight, then the adopted path
            assert torch.equal(up(x), eager[0])
            assert torch.equal(down(x), eager[1])
    assert len(alias_free._ROCM_DEPTHWISE_REPLICA) == 2


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_decoder_runs_only_under_the_fast_path_gate():
    """The fused residual add + RMSNorm is not bit-exact, so it runs only while
    the decode-scoped gate is on (quality lossless/high); exact keeps the
    unfused blocks."""
    decoder = _h3_shaped_decoder()
    latent = torch.randn(2, 4, 1, 2, 2, device="cuda", dtype=torch.float16)
    from sglang.kernels.ops import diffusion as diffusion_ops

    gate = VaeFastPathGate()
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.float16),
        set_forward_context(current_timestep=0, attn_metadata=None),
        mock.patch.object(
            diffusion_ops,
            "h3_vae_scale_add_rmsnorm",
            wraps=diffusion_ops.h3_vae_scale_add_rmsnorm,
        ) as scale_add,
    ):
        ungated = decoder(latent)
        decoder._sgl_gate = gate
        eager = decoder(latent)
        assert scale_add.call_count == 0
        gate.enabled = True
        fused = decoder(latent)
        assert scale_add.call_count == 3

    assert torch.equal(eager, ungated)
    torch.testing.assert_close(fused, eager, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_decoder_is_disabled_off_cuda_and_rocm():
    decoder = _h3_shaped_decoder()
    latent = torch.randn(2, 4, 1, 2, 2, device="cuda", dtype=torch.float16)
    from sglang.kernels.ops import diffusion as diffusion_ops

    gate = VaeFastPathGate()
    gate.enabled = True
    decoder._sgl_gate = gate
    vae_dir = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae"
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.float16),
        set_forward_context(current_timestep=0, attn_metadata=None),
        mock.patch(f"{vae_dir}.vae_vit.current_platform.is_cuda", return_value=False),
        mock.patch(f"{vae_dir}.vae_vit.current_platform.is_rocm", return_value=False),
        mock.patch.object(diffusion_ops, "h3_vae_scale_add_rmsnorm") as scale_add,
    ):
        decoder(latent)
    scale_add.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_residual_adds_match_the_eager_fp32_add(dtype):
    """Both fused kernels store the residual as eager computes it: a rounded
    FP32 multiply, then the add (no FMA; on ROCm the launch disables fusion)."""
    from sglang.kernels.ops.diffusion import (
        h3_vae_rmsnorm,
        h3_vae_scale_add_rmsnorm,
        try_fused_scaled_residual_add_exact,
    )

    torch.manual_seed(0)
    residual = torch.randn(3, 1797, 2048, device="cuda") * 4
    update = torch.randn(3, 1797, 2048, device="cuda").to(dtype)
    scale = torch.randn(2048, device="cuda")
    weight = torch.randn(2048, device="cuda")
    expected = residual + update * scale
    with torch.no_grad():
        assert torch.equal(
            try_fused_scaled_residual_add_exact(residual, update, scale), expected
        )
        added, normed = h3_vae_scale_add_rmsnorm(
            residual.clone(), update, scale, weight, 1e-5
        )
    assert torch.equal(added, expected)
    reference = nn.functional.rms_norm(expected, (2048,), weight, 1e-5)
    torch.testing.assert_close(normed, reference, rtol=1e-5, atol=1e-5)

    # stored in the autocast dtype, the rows round exactly as the linear's input
    # cast; only ROCm stores it (vae_vit.py), and only there is it bit-equal
    if not current_platform.is_rocm():
        return
    with torch.no_grad():
        _, normed_half = h3_vae_scale_add_rmsnorm(
            residual.clone(), update, scale, weight, 1e-5, out_dtype=dtype
        )
        first = h3_vae_rmsnorm(residual, weight, 1e-5)
        first_half = h3_vae_rmsnorm(residual, weight, 1e-5, out_dtype=dtype)
    assert torch.equal(normed_half, normed.to(dtype))
    assert first.dtype == torch.float32
    assert torch.equal(first_half, first.to(dtype))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm aten replica")
@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_qk_rms_norm_replays_aten(width, dtype):
    """Strided per-head rows of a fused qkv normalize to aten's RMSNorm bytes,
    and the VAE's qk norm returns them through the gate."""
    from sglang.kernels.ops.diffusion import aten_rms_norm_rocm

    torch.manual_seed(0)
    qkv = torch.randn(1, 777, 8, 3 * width, device="cuda") * 3
    qkv = (qkv + torch.randn(1, 777, 8, 1, device="cuda")).to(dtype)
    q = qkv[..., :width]
    ref = nn.functional.rms_norm(q, (width,), eps=1e-5)
    assert torch.equal(aten_rms_norm_rocm(q, 1e-5), ref)

    norm = nn.RMSNorm(width, eps=1e-5, elementwise_affine=False)
    with torch.inference_mode():
        assert torch.equal(_apply_qk_norm(norm, q), ref)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm aten replica")
@pytest.mark.parametrize("width,rot", [(64, 48), (64, 64), (128, 128)])
def test_rocm_qk_norm_rope_replays_both_norms_and_the_rope(width, rot):
    """One launch gives the bytes of the two aten RMSNorms plus sgl_kernel's
    NeoX rope (BF16 products and sums each rounded), and the VAE attention
    takes it through its gate."""
    dtype = torch.bfloat16
    torch.manual_seed(0)
    qkv = torch.randn(2, 777, 8, 3 * width, device="cuda") * 3
    qkv = (qkv + torch.randn(2, 777, 8, 1, device="cuda")).to(dtype)
    query, key, _ = torch.chunk(qkv, 3, dim=-1)
    angle = torch.rand(777, rot // 2, device="cuda") * 50
    cache = torch.cat((angle.cos(), angle.sin()), -1).to(dtype).contiguous()
    positions = torch.arange(777, device="cuda").repeat(2)
    rope = (None, None, cache, positions)
    norm = nn.RMSNorm(width, eps=1e-5, elementwise_affine=False)
    attn = SimpleNamespace(norm_q=norm, norm_k=norm)

    with torch.inference_mode():
        expected = apply_rotary_pos_emb_qk(
            _apply_qk_norm(norm, query), _apply_qk_norm(norm, key), rope
        )
        fused = vit_attention._rocm_qk_norm_rope(attn, query, key, rope)

    assert fused is not None
    assert all(map(torch.equal, fused, expected))
    assert vit_attention._ROCM_QK_NORM_ROPE.is_verified((width, rot, dtype))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm aten replica")
def test_rocm_decoder_attention_takes_the_fused_qk_norm_rope():
    # H3's rope_dim_ratio 0.75 rotates 48 of the 64 head dims
    decoder = _h3_shaped_decoder()
    latent = torch.randn(2, 4, 1, 2, 2, device="cuda")
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        decoder(latent)
    assert vit_attention._ROCM_QK_NORM_ROPE.is_verified((64, 48, torch.bfloat16))
