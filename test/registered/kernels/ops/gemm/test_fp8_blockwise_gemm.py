import sys
from typing import Optional, Type

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b",
    runner_config="1-gpu-small",
)


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: Type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)
    output = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


def _test_accuracy_once(M, N, K, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()
    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)
    scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
    scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)
    scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32) * 0.001
    scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32) * 0.001
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    o = baseline_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
    o1 = fp8_blockwise_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
    rtol = 0.02
    atol = 1
    torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not is_sm120_supported(), reason="fp8_blockwise_scaled_mm requires SM120 (>= 12.0)"
)
@pytest.mark.parametrize("M", [1, 3, 5, 32, 48, 64, 127, 128, 512, 1024, 4096])
@pytest.mark.parametrize("N", [128, 512, 1024, 4096, 8192])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(M, N, K, out_dtype):
    _test_accuracy_once(M, N, K, out_dtype, "cuda")


@pytest.mark.skipif(not is_sm120_supported(), reason="requires SM120")
@pytest.mark.parametrize("M", [4, 8, 12, 16, 32, 36, 60, 64, 68])
@pytest.mark.parametrize(
    "N,K",
    [
        (34816, 5120),
        (5120, 17408),
        (16384, 5120),
        (5120, 6144),
        (14336, 5120),
        (5120, 16512),  # An odd number of K blocks: unequal split-K partitions.
        (5120, 4224),
        (5120, 4352),
        (5120, 4480),  # Unequal warp split-K partitions.
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_qwen_decode(M, N, K, out_dtype):
    """Exercise the small-M dispatch and its boundary with model-size weights."""
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).T
    sa = torch.rand(K // 128, M, device="cuda").T * 0.1
    sb = torch.rand(N // 128, K // 128, device="cuda").T * 0.1
    expected = baseline_scaled_mm(a, b, sa, sb, out_dtype)
    actual = fp8_blockwise_scaled_mm(a, b, sa, sb, out_dtype)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.skipif(not is_sm120_supported(), reason="requires SM120")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (4, 34816, 5120),
        (4, 5120, 4224),
        (12, 5120, 6144),
        (36, 5120, 17408),
        (60, 5120, 16512),
        (60, 14336, 5120),
    ],
)
def test_qwen_decode_cuda_graph(M, N, K):
    """Replays must consume current activations, weights, and block scales."""
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).T
    sa = torch.rand(K // 128, M, device="cuda").T * 0.1
    sb = torch.rand(N // 128, K // 128, device="cuda").T * 0.1
    fp8_blockwise_scaled_mm(a, b, sa, sb, torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp8_blockwise_scaled_mm(a, b, sa, sb, torch.bfloat16)
    for _ in range(3):
        a.copy_(torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn))
        b.copy_(torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).T)
        sa.uniform_(0, 0.1)
        sb.uniform_(0, 0.1)
        graph.replay()
        expected = baseline_scaled_mm(a, b, sa, sb, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
