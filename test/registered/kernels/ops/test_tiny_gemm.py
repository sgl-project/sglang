import pytest
import torch

from sglang.kernels.ops.gemm.tiny_gemm import tiny_k_gemm_bf16, tiny_n_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# (n, k) shapes: skinny decode projections; (1536, 512) exercises the
# multi-wave fallback (split_n capped by the 32-thread block).
SHAPES = [(144, 7168), (896, 7168), (256, 4096), (1536, 512)]


def _ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.double() @ w.double().t()


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 7, 16])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_tiny_gemm_correctness(n, k, m, out_dtype):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    out = tiny_n_gemm_bf16(x, w, out_dtype=out_dtype)
    assert out.shape == (m, n) and out.dtype == out_dtype
    rtol, atol = (1e-3, 1e-3) if out_dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(out.double(), _ref(x, w), rtol=rtol, atol=atol)


@pytest.mark.parametrize("split_n", [1, 2, 4, 8])
def test_tiny_gemm_split_n(split_n):
    torch.manual_seed(0)
    x = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn(896, 7168, device="cuda", dtype=torch.bfloat16) / 8
    out = tiny_n_gemm_bf16(x, w, split_n=split_n, out_dtype=torch.float32)
    torch.testing.assert_close(out.double(), _ref(x, w), rtol=1e-3, atol=1e-3)


def test_tiny_gemm_out_param():
    x = torch.randn(2, 7168, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(144, 7168, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(2, 144, device="cuda", dtype=torch.float32)
    result = tiny_n_gemm_bf16(x, w, out=out)
    assert result is out
    torch.testing.assert_close(out.double(), _ref(x, w), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("n,k", [(1536, 128), (896, 256)])
@pytest.mark.parametrize("m", [1, 2, 7, 16])
@pytest.mark.parametrize("split_n", [None, 2, 4])
def test_tiny_k_gemm_correctness(n, k, m, split_n):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 4
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 4
    out = tiny_k_gemm_bf16(x, w, split_n=split_n)
    torch.testing.assert_close(out.double(), _ref(x, w), rtol=2e-2, atol=2e-2)


def test_tiny_k_gemm_strided_rows():
    # The K3 f_b input is a row-sliced view of the fused bfa output.
    torch.manual_seed(0)
    bfa = torch.randn(4, 144, device="cuda", dtype=torch.bfloat16) / 4
    x = bfa[..., :128]
    w = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16) / 4
    out = tiny_k_gemm_bf16(x, w)
    torch.testing.assert_close(out.double(), _ref(x, w), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
