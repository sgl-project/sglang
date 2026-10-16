"""Regression for FP16 invariant GEMM on GPUs with a 99-KiB SMEM limit."""

import pytest
import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import _matmul_persistent_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,k", [(1536, 1536), (151936, 1536), (8193, 257)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_persistent_gemm_capacity_and_batch_invariance(dtype, n, k, with_bias):
    if torch.version.hip is not None:
        pytest.skip("NVIDIA shared-memory capacity regression")
    torch.manual_seed(821913)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a_all = torch.randn(129, k, device="cuda", dtype=dtype) * 0.05
        weight = (torch.randn(n, k, device="cuda", dtype=dtype) * 0.05).T
        bias = torch.randn(n, device="cuda", dtype=dtype) * 0.01 if with_bias else None
        columns = torch.linspace(0, n - 1, min(n, 16), device="cuda").long()
        reference_weight = weight[:, columns].cpu().double()
        reference_bias = bias[columns].cpu().double() if bias is not None else 0.0
        rtol, atol = {
            torch.float16: (0.004, 0.001),
            torch.bfloat16: (0.02, 0.005),
            torch.float32: (0.001, 0.0003),
        }[dtype]
        first_row = None
        for rows in [1, 13, 63, 129]:
            a = a_all[:rows]
            actual = _matmul_persistent_triton(a, weight, dtype, bias)
            torch.cuda.synchronize()
            assert torch.isfinite(actual).all()
            reference = (a.cpu().double() @ reference_weight + reference_bias).to(dtype)
            torch.testing.assert_close(
                actual[:, columns].cpu(), reference, rtol=rtol, atol=atol
            )
            if first_row is None:
                first_row = actual[0].clone()
            else:
                assert torch.equal(first_row, actual[0])
        if dtype == torch.float16:
            a = a_all[:13]
            original = a.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _matmul_persistent_triton(a, weight, dtype, bias)
            a.add_(0.003)
            graph.replay()
            torch.cuda.synchronize()
            reference = (a.cpu().double() @ reference_weight + reference_bias).to(dtype)
            torch.testing.assert_close(
                captured[:, columns].cpu(), reference, rtol=rtol, atol=atol
            )
            a.copy_(original)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
