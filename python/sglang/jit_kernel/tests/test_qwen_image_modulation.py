import sys

import pytest
import torch
import triton

from sglang.jit_kernel.diffusion.triton.norm import norm_infer
from sglang.jit_kernel.diffusion.triton.scale_shift import (
    fuse_layernorm_scale_shift_gate_select01_kernel,
    fuse_residual_layernorm_scale_shift_gate_select01_kernel,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = get_ci_test_range(
    [torch.float16, torch.bfloat16, torch.float32], [torch.float16, torch.bfloat16]
)
BATCH_SIZES = get_ci_test_range([1, 2, 4], [1, 2])
SEQ_LENS = get_ci_test_range([6, 33, 128, 257], [6, 128])
HIDDEN_SIZES = get_ci_test_range([512, 1024, 1536, 3072], [512, 3072])
EPS = 1e-6


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 5e-2, 5e-2


def _make_modulation_tensors(batch_size: int, hidden_size: int, dtype: torch.dtype):
    scale0 = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    shift0 = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    gate0 = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    scale1 = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    shift1 = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    gate1 = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    return scale0, shift0, gate0, scale1, shift1, gate1


def _baseline_select01_modulation(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
):
    normalized = norm_infer(
        x.view(-1, x.shape[-1]),
        weight,
        bias,
        eps=eps,
        is_rms_norm=False,
    ).view_as(x)
    return _apply_select01_modulation(
        normalized, scale0, shift0, gate0, scale1, shift1, gate1, index
    )


def _baseline_residual_select01_modulation(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
):
    residual_out = residual + residual_gate * x
    normalized = norm_infer(
        residual_out.view(-1, residual_out.shape[-1]),
        weight,
        bias,
        eps=eps,
        is_rms_norm=False,
    ).view_as(residual_out)
    output, gate_out = _apply_select01_modulation(
        normalized, scale0, shift0, gate0, scale1, shift1, gate1, index
    )
    return output, residual_out, gate_out


def _apply_select01_modulation(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
):
    idx = index.bool().unsqueeze(-1)
    scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
    return x * (1 + scale) + shift, gate


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_fused_layernorm_scale_shift_gate_select01(
    dtype, batch_size, seq_len, hidden_size
):
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    bias = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    index = torch.randint(0, 2, (batch_size, seq_len), device=DEVICE, dtype=torch.int32)
    scale0, shift0, gate0, scale1, shift1, gate1 = _make_modulation_tensors(
        batch_size, hidden_size, dtype
    )

    out_ref, gate_ref = _baseline_select01_modulation(
        x,
        weight,
        bias,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        EPS,
    )
    out_fused, gate_fused = fuse_layernorm_scale_shift_gate_select01_kernel(
        x.contiguous(),
        weight=weight,
        bias=bias,
        scale0=scale0,
        shift0=shift0,
        gate0=gate0,
        scale1=scale1,
        shift1=shift1,
        gate1=gate1,
        index=index,
        eps=EPS,
    )

    atol, rtol = _tol(dtype)
    triton.testing.assert_close(out_ref, out_fused, atol=atol, rtol=rtol)
    triton.testing.assert_close(gate_ref, gate_fused, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_fused_residual_layernorm_scale_shift_gate_select01(
    dtype, batch_size, seq_len, hidden_size
):
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=dtype)
    residual = torch.randn_like(x)
    residual_gate = torch.randn_like(x)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    bias = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    index = torch.randint(0, 2, (batch_size, seq_len), device=DEVICE, dtype=torch.int32)
    scale0, shift0, gate0, scale1, shift1, gate1 = _make_modulation_tensors(
        batch_size, hidden_size, dtype
    )

    out_ref, residual_ref, gate_ref = _baseline_residual_select01_modulation(
        x,
        residual,
        residual_gate,
        weight,
        bias,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        EPS,
    )
    out_fused, residual_fused, gate_fused = (
        fuse_residual_layernorm_scale_shift_gate_select01_kernel(
            x.contiguous(),
            residual=residual.contiguous(),
            residual_gate=residual_gate.contiguous(),
            weight=weight,
            bias=bias,
            scale0=scale0,
            shift0=shift0,
            gate0=gate0,
            scale1=scale1,
            shift1=shift1,
            gate1=gate1,
            index=index,
            eps=EPS,
        )
    )

    atol, rtol = _tol(dtype)
    triton.testing.assert_close(out_ref, out_fused, atol=atol, rtol=rtol)
    triton.testing.assert_close(residual_ref, residual_fused, atol=atol, rtol=rtol)
    triton.testing.assert_close(gate_ref, gate_fused, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
