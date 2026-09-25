"""MXFP8 epilogues stay bitwise identical to FlashInfer's standalone quantizer."""

import sys

import flashinfer
import pytest
import torch

from sglang.kernels.ops.layernorm.mxfp8_epilogue import rmsnorm_mxfp8
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the MXFP8 reference quantizer is Blackwell-only",
)

DEVICE = "cuda"

HIDDEN = 5120
STREAMS = 4


@pytest.mark.parametrize("m", [1, 2, 5, 6, 8])
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
