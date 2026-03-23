# Adapted from sgl-kernel/tests/test_norm.py

import sys

import pytest
import torch

# JIT rmsnorm: fp16/bf16 only
# - Warp norm path (one warp per token):  hidden_size in {64, 128, 256}
# - CTA norm path (multi-warp per token): hidden_size is a multiple of 256, > 256, and <=8192
RMSNORM_HIDDEN_SIZES = [64, 128, 256, 512, 1024, 3072, 3584, 4096, 8192]

# JIT fused_add_rmsnorm: fp16/bf16 only; hidden_size % 8 == 0, <=8192
FUSED_ADD_RMSNORM_HIDDEN_SIZES = [1024, 3072, 3584, 4096, 8192]

BS_LIST = [
    1,
    19,
    99,
    989,
    8192,
]  # 8192 ensures num_tokens > max_occupancy * kNumSM on any GPU


def _jit_rmsnorm(input, weight, output, eps):
    from sglang.jit_kernel.norm import rmsnorm

    rmsnorm(input, weight, output=output, eps=eps)


def _fi_rmsnorm(input, weight, out, eps):
    from flashinfer.norm import rmsnorm

    rmsnorm(input, weight, out=out, eps=eps)


def _jit_fused_add_rmsnorm(input, residual, weight, eps):
    from sglang.jit_kernel.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps)


def _fi_fused_add_rmsnorm(input, residual, weight, eps):
    from flashinfer.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps=eps)


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("hidden_size", RMSNORM_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm_jit(batch_size, hidden_size, dtype, specify_out):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # flashinfer reference
    x_ref = x.clone()
    _fi_rmsnorm(x_ref, w, out=x_ref, eps=eps)

    if specify_out:
        y = torch.empty_like(x)
        _jit_rmsnorm(x, w, output=y, eps=eps)
    else:
        y = x.clone()
        _jit_rmsnorm(y, w, output=y, eps=eps)

    torch.testing.assert_close(y, x_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("hidden_size", FUSED_ADD_RMSNORM_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_jit(batch_size, hidden_size, dtype):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    # flashinfer reference
    x_ref = x.clone()
    r_ref = residual.clone()
    _fi_fused_add_rmsnorm(x_ref, r_ref, weight, eps=eps)

    x_jit = x.clone()
    r_jit = residual.clone()
    _jit_fused_add_rmsnorm(x_jit, r_jit, weight, eps)

    torch.testing.assert_close(x_jit, x_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(r_jit, r_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (0, False),
        (64, True),
        (128, True),
        (256, True),
        (512, True),
        (8192, True),
        (16384, False),
    ],
)
def test_rmsnorm_hidden_size_support(hidden_size, expected):
    from sglang.jit_kernel.norm import _is_supported_rmsnorm_hidden_size

    assert _is_supported_rmsnorm_hidden_size(hidden_size) is expected


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (64, "RMSNormWarpKernel"),
        (128, "RMSNormWarpKernel"),
        (256, "RMSNormWarpKernel"),
        (512, "RMSNormKernel"),
        (8192, "RMSNormKernel"),
    ],
)
def test_rmsnorm_kernel_dispatch(hidden_size, expected):
    from sglang.jit_kernel.norm import _rmsnorm_kernel_class

    assert _rmsnorm_kernel_class(hidden_size) == expected


@pytest.mark.parametrize("hidden_size", [0, 16384])
def test_rmsnorm_rejects_unsupported_hidden_size(hidden_size):
    from sglang.jit_kernel.norm import rmsnorm

    x = torch.randn(1, hidden_size)
    w = torch.randn(hidden_size)

    with pytest.raises(RuntimeError, match=f"unsupported hidden_size={hidden_size}"):
        rmsnorm(x, w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
