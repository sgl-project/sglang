import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=160, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=45, suite="jit-kernel-unit-test-amd")


EPS = 1e-6
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]


def sglang_jit_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    eps: float = EPS,
) -> None:
    from sglang.kernels.ops.layernorm.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


def flashinfer_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    from flashinfer.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


def torch_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    x = input.float()
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    output.copy_((normed * weight.float()).to(output.dtype))


def reference_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    # NVIDIA uses flashinfer (the bitwise reference); flashinfer is CUDA-only,
    # so on ROCm fall back to the torch reference (matches flashinfer math).
    if is_hip():
        torch_rmsnorm(input, weight, output=output, eps=eps)
    else:
        flashinfer_rmsnorm(input, weight, output=output, eps=eps)


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
SUPPORTED_HIDDEN_SIZE_LIST = [
    64,
    128,
    256,
    512,
    *range(1024, 8192 + 1, 1024),
    1536,
    2304,
    2560,
    8704,
    12288,
    16384,
]
RMSNORM_CASES = get_ci_test_range(
    list(itertools.product(BS_LIST, SUPPORTED_HIDDEN_SIZE_LIST)),
    [
        (1, 256),
        (18, 1024),
        (38, 4096),
        (1240, 1536),
        (2500, 1024),
        (4109, 1024),
        (7807, 128),
    ],
)


@pytest.mark.parametrize("batch_size,hidden_size", RMSNORM_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype, specify_out: bool
) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)

    input_ref = input.clone()
    output_ref = torch.empty_like(input)
    reference_rmsnorm(input_ref, weight, output=output_ref)

    if specify_out:
        output_sglang = torch.empty_like(input)
        sglang_jit_rmsnorm(input, weight, output=output_sglang)
    else:
        output_sglang = input.clone()
        sglang_jit_rmsnorm(output_sglang, weight, output=output_sglang)

    torch.testing.assert_close(output_sglang, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 8192, 8704, 16384])
def test_rmsnorm_hidden_size_support(hidden_size: int) -> None:
    from sglang.kernels.ops.layernorm.norm import is_jit_rmsnorm_supported

    assert is_jit_rmsnorm_supported(hidden_size)


def _hf_semantics_reference(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """HF LlamaRMSNorm: normalize in fp32, cast to the activation dtype, then
    multiply the weight in that narrow dtype."""
    x_fp32 = x.float()
    normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return w * normed.to(x.dtype)


def _plain_semantics_reference(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """Default semantics: the weight multiply stays in fp32, cast at the end."""
    x_fp32 = x.float()
    normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * w.float()).to(x.dtype)


def _assert_picks_reference(out, want, other, label) -> None:
    """The two semantics differ by at most one ULP and the kernel's reduction
    order differs from PyTorch's `mean`, so neither reference is bit-exact.
    Assert the flag moves the output strictly closer to the matching one."""
    d_want = (out.float() - want).abs().max().item()
    d_other = (out.float() - other).abs().max().item()
    assert d_want < d_other, (
        f"{label}: output is closer to the wrong reference "
        f"(want={d_want}, other={d_other})"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_cast_x_before_out_mul_switches_semantics(dtype: torch.dtype) -> None:
    """`cast_x_before_out_mul` must change the rounding rather than be ignored.

    Regression guard: the flag was accepted and silently dropped, so callers
    asking for HF semantics got the fp32-multiply rounding instead.
    """
    from sglang.kernels.ops.layernorm.norm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(64, 4096, device=DEVICE, dtype=dtype)
    w = torch.randn(4096, device=DEVICE, dtype=dtype)
    hf = _hf_semantics_reference(x, w, EPS).float()
    plain = _plain_semantics_reference(x, w, EPS).float()
    assert (hf - plain).abs().max() > 0, "inputs do not exercise the difference"

    for flag, want, other in ((True, hf, plain), (False, plain, hf)):
        out = rmsnorm(x, w, EPS, cast_x_before_out_mul=flag)
        _assert_picks_reference(out, want, other, f"rmsnorm(cast={flag})")


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_cast_x_before_out_mul_switches_semantics(
    dtype: torch.dtype,
) -> None:
    """Same guard for the fused-add kernel, which is the path the framework
    actually routes to when HF semantics are requested."""
    from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(64, 4096, device=DEVICE, dtype=dtype)
    res = torch.randn_like(x)
    w = torch.randn(4096, device=DEVICE, dtype=dtype)
    summed = (x.float() + res.float()).to(dtype)
    hf = _hf_semantics_reference(summed, w, EPS).float()
    plain = _plain_semantics_reference(summed, w, EPS).float()
    assert (hf - plain).abs().max() > 0, "inputs do not exercise the difference"

    for flag, want, other in ((True, hf, plain), (False, plain, hf)):
        xi, ri = x.clone(), res.clone()
        fused_add_rmsnorm(xi, ri, w, EPS, cast_x_before_out_mul=flag)
        _assert_picks_reference(xi, want, other, f"fused_add(cast={flag})")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
