"""MXFP8 epilogues stay bitwise identical to FlashInfer's standalone quantizer."""

import os
import subprocess
import sys

import flashinfer
import pytest
import torch

from sglang.kernels.ops.layernorm.rmsnorm_mxfp8 import rmsnorm_mxfp8
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=100, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the MXFP8 reference quantizer is Blackwell-only",
)

DEVICE = "cuda"

HIDDEN = 5120
STREAMS = 4


# 9-128 pins the single-tile scale-swizzle bound hc_combine_norm_mxfp8 was
# widened to; 128 is the last row the swizzle addresses without a block offset.
# 8/9 and 48/49 straddle the changes of the row's grid split.
@pytest.mark.parametrize("m", [1, 2, 5, 6, 8, 9, 17, 33, 48, 49, 96, 127, 128])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_bitwise_identical_to_norm_then_quantize(m: int, scale: float, backend: str):
    from sglang.kernels.ops.layernorm.hc_combine_norm import (
        hc_combine_norm,
        hc_combine_norm_mxfp8,
    )
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    g = torch.Generator(device="cuda").manual_seed(m * 31 + int(scale * 1000))
    x = (
        torch.randn(
            (m, STREAMS * HIDDEN), device="cuda", dtype=torch.bfloat16, generator=g
        )
        * scale
    )
    pre = torch.randn(
        (m, STREAMS), device="cuda", dtype=torch.bfloat16, generator=g
    ).contiguous()
    w = torch.randn((HIDDEN,), device="cuda", dtype=torch.bfloat16, generator=g)
    eps = 1e-6

    y_ref = hc_combine_norm(x, pre, w, eps)
    q_ref, sf_ref = flashinfer_mxfp8_quantize(y_ref, True, 32, backend)
    y, q, sf = hc_combine_norm_mxfp8(x, pre, w, eps)

    assert torch.equal(y, y_ref)
    assert torch.equal(
        q.reshape(-1).view(torch.uint8), q_ref.reshape(-1).view(torch.uint8)
    )
    assert sf.shape == sf_ref.reshape(-1).shape
    assert torch.equal(sf, sf_ref.reshape(-1))


_PADDING_REPLAY = """
import torch
from sglang.kernels.ops.layernorm import hc_combine_norm as mod

alloc = mod._alloc


def poisoned_alloc(m, k, device):
    q, s = alloc(m, k, device)
    return q, s.fill_(0xFF)


mod._alloc = poisoned_alloc
for m in (1, 49, 127):
    g = torch.Generator().manual_seed(m)
    x = torch.randn((m, 4 * 5120), dtype=torch.bfloat16, generator=g)
    pre = torch.randn((m, 4), dtype=torch.bfloat16, generator=g)
    w = torch.randn((5120,), dtype=torch.bfloat16, generator=g)
    for parts in (1, 2, 4):
        mod._parts_for = lambda m, k, p=parts: p
        sf = mod.hc_combine_norm_mxfp8(x, pre, w, 1e-6)[2]
        # A live scale byte is never 0xFF and, for these inputs, never 0.
        assert not (sf == 0xFF).any(), f"m={m} parts={parts}: scale bytes unwritten"
        live = int(sf.count_nonzero())
        assert live == m * 160, f"m={m} parts={parts}: {m * 160 - live} live bytes zeroed"
"""


def test_combine_norm_mxfp8_padding_disjoint_from_live_rows():
    replay = subprocess.run(
        [sys.executable, "-c", _PADDING_REPLAY],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0, replay.stderr[-3000:]


def check(x, w, got):
    y, q, sf = got
    expected = flashinfer.norm.rmsnorm(x, w, 1e-6)
    eq, esf = flashinfer.mxfp8_quantize(expected, is_sf_swizzled_layout=True)
    assert torch.equal(y.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(q.view(torch.uint8), eq.view(torch.uint8))
    m = x.shape[0]
    g = torch.arange(40, device=x.device)
    row = torch.arange(m, device=x.device)[:, None]
    offsets = (g // 4) * 512 + ((row % 32) * 4 + row // 32) * 4 + g % 4
    assert torch.equal(sf[offsets], esf.flatten()[offsets])
    pad = torch.ones_like(sf, dtype=torch.bool)
    pad[offsets] = False
    assert torch.count_nonzero(sf[pad]) == 0


@pytest.mark.parametrize("m", [1, 5, 6, 8])
@pytest.mark.parametrize("stride", [1280, 1792])
def test_dynamic_graph(m, stride):
    torch.manual_seed(941)
    x = torch.randn((m, stride), device="cuda", dtype=torch.bfloat16)[:, :1280]
    w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        check(x, w, rmsnorm_mxfp8(x, w, 1e-6))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = rmsnorm_mxfp8(x, w, 1e-6)
    for scale in [0, 1e-4, 1, 100]:
        x.copy_(torch.randn_like(x) * scale)
        w.copy_(torch.randn_like(w))
        graph.replay()
        check(x, w, got)


def _inputs(m, stride, kind):
    x = torch.randn((m, stride), device="cuda", dtype=torch.float32)
    if kind == "row_scaled":
        x *= torch.exp(3 * torch.randn((m, 1), device="cuda"))
    elif kind == "tiny":
        x *= 1e-30
    elif kind == "huge":
        x *= 3e38
    elif kind == "ties":
        x = torch.round(x * 8) / 8
    elif kind == "signed_zeros":
        x[: m // 2] = 0
        x[m // 2 :, ::3] = -0.0
    return x.to(torch.bfloat16)[:, :1280]


@pytest.mark.parametrize("m", [1024, 4095, 4096, 16385])
@pytest.mark.parametrize("stride", [1280, 1792])
@pytest.mark.parametrize(
    "kind", ["normal", "row_scaled", "tiny", "huge", "ties", "signed_zeros"]
)
def test_k1280_matches_8warp_tl_sum(m, stride, kind, monkeypatch):
    from sglang.kernels.ops.layernorm import rmsnorm_mxfp8 as module

    torch.manual_seed(m)
    x = _inputs(m, stride, kind)
    w = (torch.randn(1280, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    got = rmsnorm_mxfp8(x, w, 1e-6)
    # the kernel as it ran before: tl.sum order, 8 warps
    monkeypatch.setattr(module, "_num_warps", lambda m, block: 8)
    for a, b in zip(got, rmsnorm_mxfp8(x, w, 1e-6)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
