# SPDX-License-Identifier: Apache-2.0
"""S2 tests — streaming causal-VAE decode: conv_cache + upsampler trim.

The streaming decode threads a per-conv `conv_cache` dict so a chunked causal
decode (carrying the cache across chunks) is bit-comparable to a monolithic
causal decode of the whole clip. The two logic-bearing pieces are the conv leaf
(`LTX2VideoCausalConv3d`, prepend prev tail / store last k-1 frames) and the
upsampler temporal trim (`LTXVideoUpsampler3d`, drop the anchor frame ONLY at the
true clip start). FP64, CPU.
"""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
    LTX2VideoCausalConv3d,
    LTX2VideoResnetBlock3d,
    LTXVideoUpsampler3d,
)

torch.manual_seed(0)


def _chunks(t, splits):
    out, i = [], 0
    for s in splits:
        out.append(t[:, :, i : i + s])
        i += s
    return out


@pytest.mark.parametrize("splits", [[6], [3, 3], [1, 2, 3], [2, 4]])
def test_causal_conv_chunked_equals_monolithic(splits):
    conv = (
        LTX2VideoCausalConv3d(in_channels=4, out_channels=5, kernel_size=3)
        .double()
        .eval()
    )
    x = torch.randn(1, 4, 6, 3, 3, dtype=torch.float64)
    with torch.no_grad():
        whole = conv(x, causal=True)
        cache = {}
        parts = [
            conv(c, causal=True, conv_cache=cache, cache_key="c")
            for c in _chunks(x, splits)
        ]
    chunked = torch.cat(parts, dim=2)
    assert chunked.shape == whole.shape
    torch.testing.assert_close(chunked, whole, atol=1e-9, rtol=0)


@pytest.mark.parametrize("splits", [[6], [3, 3], [2, 4], [1, 2, 3]])
def test_upsampler_chunked_equals_monolithic(splits):
    # residual + temporal upsample (stride[0]=2) -> exercises the trim gating.
    up = (
        LTXVideoUpsampler3d(
            in_channels=8, stride=(2, 2, 2), residual=True, upscale_factor=1
        )
        .double()
        .eval()
    )
    x = torch.randn(1, 8, 6, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        whole = up(x, causal=True)
        cache = {}
        parts = [
            up(c, causal=True, conv_cache=cache, cache_key="u")
            for c in _chunks(x, splits)
        ]
    chunked = torch.cat(parts, dim=2)
    # monolithic trims the anchor frame once; chunked must reproduce that exactly.
    assert chunked.shape == whole.shape
    torch.testing.assert_close(chunked, whole, atol=1e-9, rtol=0)


@pytest.mark.parametrize("splits", [[6], [3, 3], [2, 4]])
def test_resnet_chunked_equals_monolithic(splits):
    blk = LTX2VideoResnetBlock3d(in_channels=4, out_channels=4).double().eval()
    x = torch.randn(1, 4, 6, 3, 3, dtype=torch.float64)
    with torch.no_grad():
        whole = blk(x, causal=True)
        cache = {}
        parts = [
            blk(c, causal=True, conv_cache=cache, cache_key="r")
            for c in _chunks(x, splits)
        ]
    chunked = torch.cat(parts, dim=2)
    torch.testing.assert_close(chunked, whole, atol=1e-9, rtol=0)


def test_conv_cache_none_is_unchanged_dense_behavior():
    # Backward-compat: with no conv_cache, output == the original causal forward.
    conv = (
        LTX2VideoCausalConv3d(in_channels=4, out_channels=4, kernel_size=3)
        .double()
        .eval()
    )
    x = torch.randn(1, 4, 5, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        a = conv(x, causal=True)
        b = conv(x, causal=True, conv_cache=None, cache_key=None)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
