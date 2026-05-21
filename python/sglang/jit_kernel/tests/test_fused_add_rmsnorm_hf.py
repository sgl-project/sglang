"""Tests for the JIT fused_add_rmsnorm_hf kernel."""

import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.fused_add_rmsnorm_hf import (
    fused_add_rmsnorm_hf,
    is_supported_fused_add_rmsnorm_hf_hidden_size,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = torch.finfo(torch.bfloat16).eps


def forward_native_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    post_residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``RMSNorm.forward_native`` for cast_x_before_out_mul=True — the path
    the JIT kernel must match."""
    sum_fp32 = x.to(torch.float32) + residual.to(torch.float32)
    if post_residual is not None:
        sum_fp32 = sum_fp32 + post_residual.to(torch.float32)
    residual_out = sum_fp32.to(x.dtype)
    variance = sum_fp32.pow(2).mean(-1, keepdim=True)
    out = w * (sum_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)
    return out, residual_out


# Loose-tolerance correctness over a wide batch / hidden range (mirrors
# test_fused_add_rmsnorm.py).
BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 4109])
HIDDEN_SIZE_LIST = get_ci_test_range(
    [32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
    [128, 512, 2048, 8192],
)


@pytest.mark.parametrize(
    "batch_size,hidden_size", list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST))
)
def test_fused_add_rmsnorm_hf_correctness(batch_size: int, hidden_size: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    w = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

    x_in = x.clone()
    residual_in = residual.clone()
    fused_add_rmsnorm_hf(x_in, residual_in, w, EPS)

    out_ref, residual_ref = forward_native_reference(x, residual, w, EPS)
    torch.testing.assert_close(x_in, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(residual_in, residual_ref, atol=1e-2, rtol=1e-2)


# Bitwise regression guard on one canonical setup (sm100, bf16). Bitwise
# varies with shape/arch; broader correctness is the loose test above.
_BITWISE_BS = 4
_BITWISE_HS = 4096


@pytest.mark.parametrize("with_post_residual", [False, True])
def test_fused_add_rmsnorm_hf_bitwise_canonical(with_post_residual: bool) -> None:
    torch.manual_seed(0)
    x = torch.randn(_BITWISE_BS, _BITWISE_HS, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(_BITWISE_BS, _BITWISE_HS, device=DEVICE, dtype=DTYPE)
    w = torch.randn(_BITWISE_HS, device=DEVICE, dtype=DTYPE)
    post_residual = (
        torch.randn(_BITWISE_BS, _BITWISE_HS, device=DEVICE, dtype=DTYPE)
        if with_post_residual
        else None
    )

    x_k = x.clone()
    residual_k = residual.clone()
    post_residual_k = post_residual.clone() if post_residual is not None else None
    fused_add_rmsnorm_hf(x_k, residual_k, w, EPS, post_residual=post_residual_k)

    out_ref, residual_ref = forward_native_reference(
        x, residual, w, EPS, post_residual=post_residual
    )
    assert torch.equal(x_k, out_ref)
    assert torch.equal(residual_k, residual_ref)
    if post_residual is not None:
        assert torch.equal(post_residual_k, post_residual)


def test_fused_add_rmsnorm_hf_inplace() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 4096, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(8, 4096, device=DEVICE, dtype=DTYPE)
    w = torch.randn(4096, device=DEVICE, dtype=DTYPE)
    x_ptr, residual_ptr = x.data_ptr(), residual.data_ptr()
    fused_add_rmsnorm_hf(x, residual, w, EPS)
    assert x.data_ptr() == x_ptr
    assert residual.data_ptr() == residual_ptr


def test_fused_add_rmsnorm_hf_empty_input() -> None:
    x = torch.empty(0, 4096, device=DEVICE, dtype=DTYPE)
    residual = torch.empty(0, 4096, device=DEVICE, dtype=DTYPE)
    w = torch.randn(4096, device=DEVICE, dtype=DTYPE)
    fused_add_rmsnorm_hf(x, residual, w, EPS)


def test_fused_add_rmsnorm_hf_shape_mismatch_raises() -> None:
    x = torch.randn(4, 4096, device=DEVICE, dtype=DTYPE)
    bad_residual = torch.randn(4, 2048, device=DEVICE, dtype=DTYPE)
    w = torch.randn(4096, device=DEVICE, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="input shape"):
        fused_add_rmsnorm_hf(x, bad_residual, w, EPS)


def test_fused_add_rmsnorm_hf_dtype_mismatch_raises() -> None:
    x = torch.randn(4, 4096, device=DEVICE, dtype=DTYPE)
    bad_residual = torch.randn(4, 4096, device=DEVICE, dtype=torch.float16)
    w = torch.randn(4096, device=DEVICE, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="input dtype"):
        fused_add_rmsnorm_hf(x, bad_residual, w, EPS)


def test_fused_add_rmsnorm_hf_3d_input_raises() -> None:
    x = torch.randn(2, 4, 4096, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(2, 4, 4096, device=DEVICE, dtype=DTYPE)
    w = torch.randn(4096, device=DEVICE, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="input must be 2D"):
        fused_add_rmsnorm_hf(x, residual, w, EPS)


def test_fused_add_rmsnorm_hf_fp32_input_raises() -> None:
    x = torch.randn(4, 4096, device=DEVICE, dtype=torch.float32)
    residual = torch.randn(4, 4096, device=DEVICE, dtype=torch.float32)
    w = torch.randn(4096, device=DEVICE, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be fp16 or bf16"):
        fused_add_rmsnorm_hf(x, residual, w, EPS)


def test_fused_add_rmsnorm_hf_unsupported_hidden_size_raises() -> None:
    x = torch.randn(4, 500, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(4, 500, device=DEVICE, dtype=DTYPE)
    w = torch.randn(500, device=DEVICE, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="unsupported hidden_size"):
        fused_add_rmsnorm_hf(x, residual, w, EPS)


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (16, False),
        (32, True),
        (64, True),
        (96, True),
        (128, True),
        (256, True),
        (288, True),
        (384, True),
        (500, False),
        (512, True),
        (3072, True),
        (4096, True),
        (6144, True),
        (8192, True),
        (4097, False),
    ],
)
def test_is_supported_hidden_size(hidden_size: int, expected: bool) -> None:
    assert is_supported_fused_add_rmsnorm_hf_hidden_size(hidden_size) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
