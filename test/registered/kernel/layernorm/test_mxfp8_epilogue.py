"""MXFP8 epilogues stay bitwise identical to FlashInfer's standalone quantizer."""

import sys

import flashinfer
import pytest
import torch
from flashinfer import mxfp8_quantize

from sglang.kernels.ops.attention.dsv4.wo_a_bf16 import (
    _quantize_partial,
    _wo_a_reduce,
    wo_a_bf16_small_batch,
    wo_a_bf16_small_batch_mxfp8,
)
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


def test_wo_a_partial_quant_matches_quantize():
    torch.manual_seed(0)
    for rows in range(2, 9):
        for magnitude in (0.0, 1e-37, 1e-7, 1.0, 448.0, 1e10):
            partial = torch.randn(8, rows, 2, 1024, device=DEVICE) * magnitude
            bf16 = torch.empty(rows, 2048, dtype=torch.bfloat16, device=DEVICE)
            _wo_a_reduce[(rows * 8,)](partial, bf16, rows * 2048, num_warps=4)
            expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
            actual_q, actual_s = _quantize_partial(partial)
            torch.testing.assert_close(
                actual_q.view(torch.uint8),
                expected_q.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

        x = torch.randn(rows, 64, 512, device=DEVICE, dtype=torch.bfloat16)[
            :, :16
        ].view(rows, 2, 4096)
        wo_a = torch.randn(2, 1024, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
        bf16 = wo_a_bf16_small_batch(x, wo_a).flatten(1)
        q, s = wo_a_bf16_small_batch_mxfp8(x, wo_a)
        expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)

    partial = torch.randn(8, 6, 2, 1024, device=DEVICE)
    _quantize_partial(partial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = _quantize_partial(partial)
    for _ in range(3):
        partial.normal_()
        # The replay must regenerate scale padding as well as live rows.
        s.fill_(255)
        graph.replay()
        bf16 = torch.empty(6, 2048, dtype=torch.bfloat16, device=DEVICE)
        _wo_a_reduce[(48,)](partial, bf16, 6 * 2048, num_warps=4)
        eq, es = mxfp8_quantize(bf16, True, alignment=32)
        torch.testing.assert_close(
            q.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(s, es, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
