# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the local CuTeDSL TGV BF16 GEMM port."""

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("Requires CUDA.", allow_module_level=True)

major, minor = torch.cuda.get_device_capability()
if major != 10:
    pytest.skip(
        f"CuTeDSL TGV BF16 GEMM requires Blackwell SM100/SM103, got SM{major}{minor}.",
        allow_module_level=True,
    )

from sglang.jit_kernel.cutedsl_bf16_gemm import mm_bf16_tgv  # noqa: E402


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


@pytest.mark.parametrize("m,n,k", [(1, 1024, 1024), (64, 2048, 1024)])
@torch.inference_mode()
def test_mm_bf16_tgv_matches_torch_without_bias(m: int, n: int, k: int):
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.1

    got = mm_bf16_tgv(a, weight.T)
    ref = torch.mm(a, weight.T)

    assert got.shape == (m, n)
    assert got.dtype == torch.bfloat16
    assert _cosine_similarity(got, ref) > 0.99


@torch.inference_mode()
def test_mm_bf16_tgv_matches_torch_with_bias():
    torch.manual_seed(1)
    m, n, k = 8, 1024, 1024
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.1

    got = mm_bf16_tgv(a, weight.T, bias)
    ref = F.linear(a, weight, bias)

    assert got.shape == (m, n)
    assert got.dtype == torch.bfloat16
    assert _cosine_similarity(got, ref) > 0.99


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
