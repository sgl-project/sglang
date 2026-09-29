"""MXFP8 epilogues stay bitwise identical to FlashInfer's standalone quantizer."""

import sys

import flashinfer
import pytest
import torch

from sglang.kernels.ops.layernorm.mxfp8_epilogue import rmsnorm_mxfp8
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


@pytest.mark.parametrize("m", [1, 8, 9, 48, 49, 96, 128, 129, 256, 384, 512])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
@pytest.mark.parametrize("pre_dtype", [torch.bfloat16, torch.float32])
def test_bitwise_identical_to_norm_then_quantize(
    m: int, scale: float, backend: str, pre_dtype
):
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
        (m, STREAMS), device="cuda", dtype=pre_dtype, generator=g
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


def assert_norm_quantization(x, weight, outputs):
    y, q, sf = outputs
    expected = flashinfer.norm.rmsnorm(x, weight, 1e-6)
    torch.testing.assert_close(y, expected, rtol=0.008, atol=1e-6)
    expected_q, expected_sf = flashinfer.mxfp8_quantize(y, is_sf_swizzled_layout=True)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    m = x.shape[0]
    groups = x.shape[1] // 32
    g = torch.arange(groups, device=x.device)
    row = torch.arange(m, device=x.device)[:, None]
    offsets = (
        (row // 128) * ((groups + 3) // 4) * 512
        + (g // 4) * 512
        + ((row % 32) * 4 + (row // 32) % 4) * 4
        + g % 4
    )
    assert torch.equal(sf[offsets], expected_sf.flatten()[offsets])
    padding = torch.ones_like(sf, dtype=torch.bool)
    padding[offsets] = False
    assert torch.count_nonzero(sf[padding]) == 0


@pytest.mark.parametrize("m", [1, 8, 9, 96, 127, 128, 129, 255, 256, 384, 511, 512])
@pytest.mark.parametrize("stride", [1280, 1792])
def test_dynamic_graph(m, stride):
    torch.manual_seed(941)
    x = torch.randn((m, stride), device="cuda", dtype=torch.bfloat16)[:, :1280]
    w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        assert_norm_quantization(x, w, rmsnorm_mxfp8(x, w, 1e-6))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = rmsnorm_mxfp8(x, w, 1e-6)
    for scale in [0, 1e-4, 1, 100]:
        x.copy_(torch.randn_like(x) * scale)
        w.copy_(torch.randn_like(w))
        got[2].fill_(255)
        graph.replay()
        assert_norm_quantization(x, w, got)


@pytest.mark.parametrize("m", [1, 128, 129, 512])
@pytest.mark.parametrize("k", [32, 64, 96, 160, 5120])
def test_partial_scale_group_padding(m, k):
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(k, device=DEVICE, dtype=torch.bfloat16)
    rmsnorm_mxfp8(x, w, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = rmsnorm_mxfp8(x, w, 1e-6)
    for _ in range(2):
        x.normal_()
        got[2].fill_(255)
        graph.replay()
        assert_norm_quantization(x, w, got)


@pytest.mark.parametrize("m", [9, 128, 129, 384, 512])
@pytest.mark.parametrize("k", [1280, 5120])
def test_fused_scales_with_gemm_consumer(m, k):
    torch.manual_seed(m + k)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(128, k, device=DEVICE, dtype=torch.bfloat16)
    norm_weight = torch.ones(k, device=DEVICE, dtype=torch.bfloat16)
    y, q, sf = rmsnorm_mxfp8(x, norm_weight, 1e-6)
    expected_q, expected_sf = flashinfer.mxfp8_quantize(y, is_sf_swizzled_layout=True)
    wq, wsf = flashinfer.mxfp8_quantize(weight, is_sf_swizzled_layout=True)

    def consume(values, scales):
        return flashinfer.mm_mxfp8(
            values, wq.T, scales, wsf, out_dtype=torch.bfloat16, backend="cute-dsl"
        )

    expected = consume(expected_q, expected_sf)
    torch.testing.assert_close(consume(q, sf), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = consume(q, sf)
    graph.replay()
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 128, 129, 512])
@pytest.mark.parametrize("k", [32, 64, 96, 160, 5120])
def test_hc_partial_scale_group_padding(m, k):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm_mxfp8

    x = torch.randn(m, STREAMS * k, device=DEVICE, dtype=torch.bfloat16)
    pre = torch.zeros(m, STREAMS, device=DEVICE, dtype=torch.float32)
    pre[:, 0] = 1
    weight = torch.randn(k, device=DEVICE, dtype=torch.bfloat16)
    hc_combine_norm_mxfp8(x, pre, weight, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = hc_combine_norm_mxfp8(x, pre, weight, 1e-6)
    for _ in range(2):
        x.normal_()
        outputs[2].fill_(255)
        graph.replay()
        assert_norm_quantization(x[:, :k], weight, outputs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
