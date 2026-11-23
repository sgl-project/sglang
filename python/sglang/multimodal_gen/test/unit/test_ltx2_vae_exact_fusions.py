# SPDX-License-Identifier: Apache-2.0
"""LTX video VAE exact fast paths must reproduce the eager modules bit for bit.

Two fusions sit in ``LTX2VideoCausalConv3d``: the conv bias (+ residual)
epilogue, and the ``PerChannelRMSNorm`` + SiLU prologue written straight into
the temporally padded conv input. The oracle is the same module with both
kernels unmounted, which runs the original aten chain.
"""

from contextlib import contextmanager
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.vaes import ltx_2_vae as vae_module
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
    LTX2VideoCausalConv3d,
    LTX2VideoDecoder3d,
    LTX2VideoResnetBlock3d,
    PerChannelRMSNorm,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (activation dtype, autocast dtype or None)
PRECISIONS = [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float32, None),
]


@contextmanager
def _eager():
    with mock.patch.multiple(
        vae_module,
        pixel_norm_silu=None,
        can_use_pixel_norm_silu=None,
        conv_bias_epilogue=None,
        can_use_conv_bias_epilogue=None,
    ):
        yield


@contextmanager
def _precision(autocast_dtype):
    with (
        torch.no_grad(),
        torch.autocast(
            "cuda",
            dtype=autocast_dtype or torch.bfloat16,
            enabled=autocast_dtype is not None,
        ),
    ):
        yield


@contextmanager
def _counting(name):
    calls = []
    real = getattr(vae_module, name)

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    with mock.patch.object(vae_module, name, spy):
        yield calls


def _module(module, weight_dtype, channels_last=True):
    module = module.to("cuda", weight_dtype).eval()
    if channels_last:
        module = module.to(memory_format=torch.channels_last_3d)
    return module


def _weight_dtype(x_dtype, autocast_dtype):
    return autocast_dtype or x_dtype


def _x(shape, dtype, channels_last):
    x = torch.randn(shape, device="cuda").to(dtype)
    if channels_last:
        return x.contiguous(memory_format=torch.channels_last_3d)
    return x.contiguous()


def _reference_block(block, x, causal=True, conv_cache=None, cache_key=None):
    """``LTX2VideoResnetBlock3d.forward`` as written before the fusions."""
    key = (lambda n: None) if cache_key is None else (lambda n: f"{cache_key}.{n}")
    h = F.silu(block.norm1(x))
    h = block.conv1(h, causal=causal, conv_cache=conv_cache, cache_key=key("conv1"))
    h = F.silu(block.norm2(h))
    h = block.conv2(h, causal=causal, conv_cache=conv_cache, cache_key=key("conv2"))
    if block.norm3 is not None:
        x = block.norm3(x.movedim(1, -1)).movedim(-1, 1)
    if block.conv_shortcut is not None:
        x = block.conv_shortcut(x)
    return h + x


def test_cpu_block_matches_the_original_chain():
    torch.manual_seed(0)
    for in_channels, out_channels in ((8, 8), (8, 12)):
        block = LTX2VideoResnetBlock3d(in_channels, out_channels).eval()
        x = torch.randn(1, in_channels, 4, 5, 6)
        with torch.no_grad():
            assert torch.equal(block(x), _reference_block(block, x))
            cache, ref_cache = {}, {}
            for chunk in (x[:, :, :1], x[:, :, 1:]):
                out = block(chunk, conv_cache=cache, cache_key="b")
                ref = _reference_block(
                    block, chunk, conv_cache=ref_cache, cache_key="b"
                )
                assert torch.equal(out, ref)
            assert cache.keys() == ref_cache.keys()


@cuda_only
@pytest.mark.parametrize("x_dtype,autocast_dtype", PRECISIONS)
@pytest.mark.parametrize("x_channels_last", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_conv_norm_prologue_and_epilogue_match_eager(
    x_dtype, autocast_dtype, x_channels_last, causal
):
    torch.manual_seed(0)
    conv = _module(
        LTX2VideoCausalConv3d(24, 40), _weight_dtype(x_dtype, autocast_dtype)
    )
    norm = PerChannelRMSNorm()
    x = _x((2, 24, 5, 7, 9), x_dtype, x_channels_last)
    with _precision(autocast_dtype):
        with _eager():
            ref = conv(F.silu(norm(x)), causal=causal)
            residual = torch.randn_like(ref)
            ref_res = ref + residual
        with (
            _counting("pixel_norm_silu") as norms,
            _counting("conv_bias_epilogue") as biases,
        ):
            out = conv(x, causal=causal, norm=norm)
            out_res = conv(x, causal=causal, norm=norm, residual=residual)
    assert len(norms) == 2 and len(biases) == 2
    assert not vae_module._NORM_SILU_FUSION.disabled
    assert out.dtype == ref.dtype and out.stride() == ref.stride()
    assert torch.equal(out, ref)
    assert torch.equal(out_res, ref_res)


@cuda_only
@pytest.mark.parametrize("x_dtype,autocast_dtype", PRECISIONS)
def test_cached_causal_chunks_match_eager(x_dtype, autocast_dtype):
    torch.manual_seed(0)
    conv = _module(
        LTX2VideoCausalConv3d(16, 16), _weight_dtype(x_dtype, autocast_dtype)
    )
    norm = PerChannelRMSNorm()
    chunks = [_x((1, 16, t, 6, 8), x_dtype, t % 2 == 0) for t in (1, 3, 1, 2)]
    cache, ref_cache = {}, {}
    with _precision(autocast_dtype):
        for chunk in chunks:
            with _eager():
                ref = conv(F.silu(norm(chunk)), conv_cache=ref_cache, cache_key="conv")
            with _counting("pixel_norm_silu") as norms:
                out = conv(chunk, conv_cache=cache, cache_key="conv", norm=norm)
            assert len(norms) == 1
            assert torch.equal(out, ref)
            # The eager cache holds the fp32 activation; the conv rounds it to
            # the dtype the fused cache already stores.
            assert torch.equal(
                cache["conv"].float(), ref_cache["conv"].to(cache["conv"].dtype).float()
            )


@cuda_only
@pytest.mark.parametrize("in_channels,out_channels", [(32, 32), (32, 48)])
@pytest.mark.parametrize("x_channels_last", [True, False])
def test_resnet_block_matches_eager(in_channels, out_channels, x_channels_last):
    torch.manual_seed(0)
    block = _module(LTX2VideoResnetBlock3d(in_channels, out_channels), torch.bfloat16)
    x = _x((1, in_channels, 4, 10, 12), torch.bfloat16, x_channels_last)
    with _precision(torch.bfloat16):
        with _eager():
            ref = _reference_block(block, x)
        with (
            _counting("pixel_norm_silu") as norms,
            _counting("conv_bias_epilogue") as biases,
        ):
            out = block(x)
    assert len(norms) == 2 and len(biases) == 2
    assert out.stride() == ref.stride()
    assert torch.equal(out, ref)


def _decoder(spatial_padding_mode="zeros", is_causal=False):
    torch.manual_seed(0)
    decoder = LTX2VideoDecoder3d(
        in_channels=16,
        out_channels=3,
        block_out_channels=(32, 64),
        spatio_temporal_scaling=(True, True),
        layers_per_block=(1, 1, 1),
        patch_size=2,
        is_causal=is_causal,
        inject_noise=(False, False, False),
        upsample_residual=(True, True),
        upsample_factor=(2, 2),
        upsample_type=("spatial", "temporal"),
        spatial_padding_mode=spatial_padding_mode,
    )
    return _module(decoder, torch.bfloat16)


@cuda_only
@pytest.mark.parametrize("spatial_padding_mode", ["zeros", "reflect"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_decoder_matches_eager(spatial_padding_mode, is_causal):
    decoder = _decoder(spatial_padding_mode, is_causal)
    z = torch.randn(1, 16, 3, 6, 8, device="cuda")
    with _precision(torch.bfloat16):
        with _eager():
            ref = decoder(z)
        with _counting("pixel_norm_silu") as norms:
            out = decoder(z)
    assert len(norms) == 7  # 3 resnet blocks x 2 + norm_out
    assert not vae_module._NORM_SILU_FUSION.disabled
    assert torch.equal(out, ref)


@cuda_only
def test_streamed_decoder_chunks_match_eager():
    decoder = _decoder(is_causal=True)
    chunks = [torch.randn(1, 16, t, 6, 8, device="cuda") for t in (1, 2, 1)]
    cache, ref_cache = {}, {}
    with _precision(torch.bfloat16):
        for chunk in chunks:
            with _eager():
                ref = decoder(chunk, causal=True, conv_cache=ref_cache)
            out = decoder(chunk, causal=True, conv_cache=cache)
            assert torch.equal(out, ref)
    assert cache.keys() == ref_cache.keys()


@cuda_only
def test_unsupported_precision_and_layout_stay_eager():
    torch.manual_seed(0)
    norm = PerChannelRMSNorm()
    x = _x((1, 16, 3, 6, 8), torch.bfloat16, True)
    # bf16 without autocast keeps the norm chain in bf16: not the fused contract.
    conv = _module(LTX2VideoCausalConv3d(16, 16), torch.bfloat16)
    # NCDHW weights: eager pads in NCDHW, so the conv input layout would change.
    plain = _module(LTX2VideoCausalConv3d(16, 16), torch.bfloat16, channels_last=False)
    with torch.no_grad(), _counting("pixel_norm_silu") as norms:
        out = conv(x, norm=norm)
        plain_out = plain(x, norm=norm)
        with _eager():
            assert torch.equal(out, conv(F.silu(norm(x))))
            assert torch.equal(plain_out, plain(F.silu(norm(x))))
    assert not norms


@cuda_only
def test_mismatching_kernel_falls_back_to_eager():
    torch.manual_seed(0)
    conv = _module(LTX2VideoCausalConv3d(16, 16), torch.bfloat16)
    norm = PerChannelRMSNorm()
    x = _x((1, 16, 3, 6, 8), torch.bfloat16, True)

    def wrong(x, eps, out):
        return out.fill_(1.0)

    gate = BitExactFusionGate("test LTX pixel norm", per_signature=True)
    with _precision(torch.bfloat16):
        with _eager():
            ref = conv(F.silu(norm(x)))
        with mock.patch.multiple(
            vae_module, pixel_norm_silu=wrong, _NORM_SILU_FUSION=gate
        ):
            assert torch.equal(conv(x, norm=norm), ref)
            assert gate.disabled
            assert torch.equal(conv(x, norm=norm), ref)


@cuda_only
def test_timestep_conditioned_block_keeps_the_eager_chain():
    torch.manual_seed(0)
    block = _module(
        LTX2VideoResnetBlock3d(16, 16, timestep_conditioning=True), torch.bfloat16
    )
    assert isinstance(block.nonlinearity, nn.SiLU)
    x = _x((1, 16, 3, 6, 8), torch.bfloat16, True)
    temb = torch.randn(1, 64, 1, 1, 1, device="cuda")
    with _precision(torch.bfloat16), _counting("pixel_norm_silu") as norms:
        out = block(x, temb)
        with _eager():
            ref = block(x, temb)
    assert not norms
    assert torch.equal(out, ref)
