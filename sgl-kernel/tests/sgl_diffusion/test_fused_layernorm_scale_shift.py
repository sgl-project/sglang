from typing import Optional

import pytest
import torch
from sgl_kernel import (
    fused_layernorm_scale_shift,
    fused_scale_residual_layernorm_scale_shift,
)


# Reference fused output: compute LN in fp32, then apply scale/shift in fp32, cast back
@torch.no_grad()
def fused_layernorm_scale_shift_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
):
    x32 = x.float()
    s32 = scale.float()
    sh32 = shift.float()
    mean = x32.mean(dim=1, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (x32 - mean) * inv_std
    if weight is not None:
        w32 = weight.float()
        y_ln32 = y_ln32 * w32
        if bias is not None:
            b32 = bias.float()
            y_ln32 = y_ln32 + b32
    if s32.ndim == 4:
        M = x32.shape[0]
        B, F, _, _ = s32.shape
        S = M // (B * F)
        y_gt_fused = torch.empty_like(y_ln32)
        for m in range(M):
            b = (m // (F * S)) % B
            s_in_b = m - b * F * S
            f = s_in_b // S
            y_gt_fused[m] = y_ln32[m] * (1.0 + s32[b, f, 0]) + sh32[b, f, 0]
    else:
        y_gt_fused = y_ln32 * (1.0 + s32) + sh32
    return y_gt_fused.to(dtype)


@torch.no_grad()
def fused_scale_residual_layernorm_scale_shift_ref(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
):
    r32 = residual.float()
    x32 = x.float()
    g32 = gate.float() if gate is not None else 1
    s32 = scale.float()
    sh32 = shift.float()
    M, N = x.shape
    if gate is not None and gate.ndim == 3:
        B, _, N = gate.shape
        S = M // B
        x32 = x32.view(B, S, N)
        r32 = r32.view(B, S, N)
    if gate is not None and gate.ndim == 4:
        out32 = torch.empty_like(x32)
        B, F, _, _ = gate.shape
        S = M // (B * F)
        for m in range(M):
            b = (m // (F * S)) % B
            s_in_b = m - b * F * S
            f = s_in_b // S
            out32[m] = r32[m] + x32[m] * g32[b, f, 0]
    else:
        out32 = r32 + g32 * x32
    out32 = out32.view(M, N)
    mean = out32.mean(dim=1, keepdim=True)
    var = (out32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (out32 - mean) * inv_std
    if weight is not None:
        w32 = weight.float()
        y_ln32 = y_ln32 * w32
        if bias is not None:
            b32 = bias.float()
            y_ln32 = y_ln32 + b32
    if scale.ndim == 4:
        M = x32.shape[0]
        B, F, _, _ = s32.shape
        S = M // (B * F)
        y_ref = torch.empty_like(y_ln32)
        for m in range(M):
            b = (m // (F * S)) % B
            s_in_b = m - b * F * S
            f = s_in_b // S
            y_ref[m] = y_ln32[m] * (1.0 + s32[b, f, 0]) + sh32[b, f, 0]
    else:
        y_ref = y_ln32 * (1.0 + s32) + sh32
    residual_ref = out32
    return y_ref.to(dtype), residual_ref.to(dtype)


@torch.no_grad()
def run_case_fused_accuracy(
    dtype=torch.float32,
    M: int = 128,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)

    y_dev_fused = fused_layernorm_scale_shift(x, weight, bias, scale, shift, eps)
    y_gt_fused = fused_layernorm_scale_shift_ref(
        x, weight, bias, scale, shift, eps, dtype
    )
    return y_dev_fused, y_gt_fused


@torch.no_grad()
def run_case_fused_no_affine_accuracy(
    dtype=torch.float32,
    M: int = 128,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)

    # CUDA fused LayerNorm without affine weights (gamma=1, beta=0) + scale/shift
    y_dev_fused = fused_layernorm_scale_shift(x, None, None, scale, shift, eps)
    y_gt_fused = fused_layernorm_scale_shift_ref(
        x, None, None, scale, shift, eps, dtype
    )
    return y_dev_fused, y_gt_fused


@torch.no_grad()
def run_case_fused_4d_scale_accuracy(
    dtype=torch.float32,
    B: int = 2,
    F: int = 3,
    S: int = 4,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * F * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale4d = torch.randn(B, F, 1, N, device=device, dtype=dtype)
    shift4d = torch.randn(B, F, 1, N, device=device, dtype=dtype)

    # CUDA 4D scale/shift fused
    y_dev_fused = fused_layernorm_scale_shift(x, weight, bias, scale4d, shift4d, eps)
    y_ref = fused_layernorm_scale_shift_ref(
        x, weight, bias, scale4d, shift4d, eps, dtype
    )
    return y_dev_fused, y_ref


@torch.no_grad()
def run_case_residual_gate_int(
    dtype=torch.float32,
    B: int = 2,
    S: int = 5,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)

    # gate == 1 (no gate tensor)
    y_dev, residual_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, None, weight, bias, scale, shift, eps
    )
    y_ref, residual_ref = fused_scale_residual_layernorm_scale_shift_ref(
        residual, x, None, weight, bias, scale, shift, eps, dtype
    )
    return y_dev, residual_out_dev, y_ref, residual_ref


@torch.no_grad()
def run_case_residual_gate_3d(
    dtype=torch.float32,
    B: int = 2,
    S: int = 5,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    gate = torch.randn(B, 1, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)

    y_dev, residual_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, eps
    )
    y_ref, residual_ref = fused_scale_residual_layernorm_scale_shift_ref(
        residual, x, gate, weight, bias, scale, shift, eps, dtype
    )
    return y_dev, residual_out_dev, y_ref, residual_ref


@torch.no_grad()
def run_case_residual_gate_4d(
    dtype=torch.float32,
    B: int = 2,
    F: int = 3,
    S: int = 4,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * F * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    gate4d = torch.randn(B, F, 1, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale4d = torch.randn(B, F, 1, N, device=device, dtype=dtype)
    shift4d = torch.randn(B, F, 1, N, device=device, dtype=dtype)

    y_dev, residual_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, gate4d, weight, bias, scale4d, shift4d, eps
    )
    y_ref, residual_ref = fused_scale_residual_layernorm_scale_shift_ref(
        residual, x, gate4d, weight, bias, scale4d, shift4d, eps, dtype
    )
    return y_dev, residual_out_dev, y_ref, residual_ref


@torch.no_grad()
def run_case_residual_gate_fully_expanded(
    dtype=torch.float32,
    B: int = 1,
    S: int = 115200,
    N: int = 3072,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    # Fully expanded tensors as seen by kernel (mimicking Python-side expansion)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)
    gate = torch.randn(M, N, device=device, dtype=dtype)

    y_dev, residual_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, eps
    )
    y_ref, residual_ref = fused_scale_residual_layernorm_scale_shift_ref(
        residual, x, gate, weight, bias, scale, shift, eps, dtype
    )
    return y_dev, residual_out_dev, y_ref, residual_ref


@torch.no_grad()
def run_case_residual_gate_3d_scalar(
    dtype=torch.float32,
    B: int = 1,
    S: int = 32760,
    N: int = 1536,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    # 3D Gate [B, 1, N]
    gate = torch.randn(B, 1, N, device=device, dtype=dtype)

    # Scalar Scale/Shift
    scale = torch.zeros(1, device=device, dtype=dtype)
    shift = torch.zeros(1, device=device, dtype=dtype)

    y_dev, residual_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, eps
    )
    y_ref, residual_ref = fused_scale_residual_layernorm_scale_shift_ref(
        residual, x, gate, weight, bias, scale, shift, eps, dtype
    )
    return y_dev, residual_out_dev, y_ref, residual_ref


@torch.no_grad()
def run_case_residual_gate_3d_scalar_no_affine(
    dtype=torch.float32,
    B: int = 1,
    S: int = 32760,
    N: int = 1536,
    eps: float = 1e-5,
):
    device = "cuda"
    M = B * S
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)

    # 3D Gate [B, 1, N]
    gate = torch.randn(B, 1, N, device=device, dtype=dtype)

    # Scalar Scale/Shift
    scale = torch.zeros(1, device=device, dtype=dtype)
    shift = torch.zeros(1, device=device, dtype=dtype)

    y_dev, residual_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, gate, None, None, scale, shift, eps
    )
    y_ref, residual_ref = fused_scale_residual_layernorm_scale_shift_ref(
        residual, x, gate, None, None, scale, shift, eps, dtype
    )
    return y_dev, residual_out_dev, y_ref, residual_ref


@torch.no_grad()
def run_case_broadcast(
    dtype=torch.float32,
    M: int = 128,
    N: int = 1024,
    scale_shape=None,
    shift_shape=None,
    gate_shape=None,
    eps: float = 1e-5,
):
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    # Initialize scale/shift with specified shapes or zeros [M, N] if not provided
    if scale_shape:
        scale = torch.randn(*scale_shape, device=device, dtype=dtype)
    else:
        scale = torch.zeros(M, N, device=device, dtype=dtype)

    if shift_shape:
        shift = torch.randn(*shift_shape, device=device, dtype=dtype)
    else:
        shift = torch.zeros(M, N, device=device, dtype=dtype)

    gate = None
    if gate_shape:
        gate = torch.randn(*gate_shape, device=device, dtype=dtype)

    # Test pure fused_layernorm_scale_shift (only valid if gate is None)
    if gate is None:
        y_dev_fused = fused_layernorm_scale_shift(x, weight, bias, scale, shift, eps)

        # Reference
        x32 = x.float()
        w32 = weight.float()
        b32 = bias.float()
        s32 = scale.float()
        sh32 = shift.float()
        mean = x32.mean(dim=1, keepdim=True)
        var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
        inv_std = (var + eps).sqrt().reciprocal()
        y_ln32 = (x32 - mean) * inv_std
        y_ln32 = y_ln32 * w32 + b32
        # Broadcast scale/shift if they are [1, N] or [1, 1, N]
        if s32.ndim == 3 and s32.size(0) == 1 and s32.size(1) == 1:
            s32 = s32.view(1, N)
        if sh32.ndim == 3 and sh32.size(0) == 1 and sh32.size(1) == 1:
            sh32 = sh32.view(1, N)

        y_gt_fused = (y_ln32 * (1.0 + s32) + sh32).to(dtype)
        torch.testing.assert_close(
            y_dev_fused, y_gt_fused, atol=_tol(dtype), rtol=_tol(dtype)
        )

        # Test no-affine variant
        y_dev_no_affine = fused_layernorm_scale_shift(x, None, None, scale, shift, eps)
        y_ln32_no_affine = (x32 - mean) * inv_std
        y_gt_no_affine = (y_ln32_no_affine * (1.0 + s32) + sh32).to(dtype)
        torch.testing.assert_close(
            y_dev_no_affine, y_gt_no_affine, atol=_tol(dtype), rtol=_tol(dtype)
        )

    # Test residual + gate + fused
    y_dev_res, res_out_dev = fused_scale_residual_layernorm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, eps
    )

    # Reference
    x32 = x.float()
    r32 = residual.float()
    w32 = weight.float()
    b32 = bias.float()
    s32 = scale.float()
    sh32 = shift.float()

    if gate is not None:
        g32 = gate.float()
        # Broadcast gate if needed
        if g32.ndim == 3 and g32.size(0) == 1 and g32.size(1) == 1:
            g32 = g32.view(1, N)
        out32 = r32 + x32 * g32
    else:
        out32 = r32 + x32

    mean = out32.mean(dim=1, keepdim=True)
    var = (out32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (out32 - mean) * inv_std
    y_ln32 = y_ln32 * w32 + b32

    # Broadcast scale/shift if needed
    if s32.ndim == 3 and s32.size(0) == 1 and s32.size(1) == 1:
        s32 = s32.view(1, N)
    if sh32.ndim == 3 and sh32.size(0) == 1 and sh32.size(1) == 1:
        sh32 = sh32.view(1, N)

    y_ref = (y_ln32 * (1.0 + s32) + sh32).to(dtype)
    residual_ref = out32.to(dtype)

    torch.testing.assert_close(y_dev_res, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        res_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


# -------------------------
# PyTest entrypoints below
# -------------------------

CASES = [
    (20, 3072),
    (128, 3072),
    (256, 3072),
    (512, 3072),
    (1024, 3072),
    (2000, 3072),
    (2048, 3072),
    # Hunyuan
    (115200, 3072),
    (5, 3072),
    # Wan
    (32760, 1536),
    # Qwen
    (2025, 3072),
    (9, 3072),
    (6, 3072),
]


def _tol(dtype: torch.dtype):
    return 2e-5 if dtype == torch.float32 else 5e-2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", CASES)
def test_fused_layernorm_scale_shift_2d(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if (N % 4) != 0:
        pytest.skip("Vectorized kernel requires N % 4 == 0")
    y_dev, y_ref = run_case_fused_accuracy(dtype=dtype, M=M, N=N, eps=1e-5)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", CASES)
def test_fused_layernorm_scale_shift_no_affine_2d(dtype, M, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if (N % 4) != 0:
        pytest.skip("Vectorized kernel requires N % 4 == 0")
    y_dev, y_ref = run_case_fused_no_affine_accuracy(dtype=dtype, M=M, N=N, eps=1e-5)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,F,S,N", [(2, 3, 4, 1024), (12, 24, 1, 2048)])
def test_fused_layernorm_scale_shift_4d(dtype, B, F, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, y_ref = run_case_fused_4d_scale_accuracy(
        dtype=dtype, B=B, F=F, S=S, N=N, eps=1e-5
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(2, 5, 1024), (16, 32, 4096), (1, 32760, 1536)])
def test_residual_gate_int(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_int(
        dtype=dtype, B=B, S=S, N=N, eps=1e-5
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(2, 5, 1024), (12, 24, 2048)])
def test_residual_gate_3d(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_3d(
        dtype=dtype, B=B, S=S, N=N, eps=1e-5
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(1, 115200, 3072), (1, 5, 3072)])
def test_residual_gate_fully_expanded(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = (
        run_case_residual_gate_fully_expanded(dtype=dtype, B=B, S=S, N=N, eps=1e-5)
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(1, 32760, 1536)])
def test_residual_gate_3d_scalar(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_3d_scalar(
        dtype=dtype, B=B, S=S, N=N, eps=1e-5
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(1, 32760, 1536)])
def test_residual_gate_3d_scalar_no_affine(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = (
        run_case_residual_gate_3d_scalar_no_affine(dtype=dtype, B=B, S=S, N=N, eps=1e-5)
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,F,S,N", [(2, 3, 4, 1024), (12, 24, 1, 2048)])
def test_residual_gate_4d(dtype, B, F, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_4d(
        dtype=dtype, B=B, F=F, S=S, N=N, eps=1e-5
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(
        residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N", [(128, 1024)])
@pytest.mark.parametrize(
    "broadcast_dims",
    [
        ([1, 1024], [1, 1024], None),  # scale/shift 1xN, no gate
        ([1, 1, 1024], [1, 1, 1024], None),  # scale/shift 1x1xN, no gate
        ([1, 1024], [1, 1024], [1, 1024]),  # all 1xN
        ([1, 1, 1024], [1, 1, 1024], [1, 1, 1024]),  # all 1x1xN
        ([128, 1024], [128, 1024], [1, 1024]),  # gate broadcast only
        ([1, 1024], [1, 1024], [128, 1024]),  # scale/shift broadcast only
        ([1, 1, 1024], [1, 1, 1024], [128, 1024]),  # scale/shift 3d broadcast
    ],
)
def test_broadcast(dtype, M, N, broadcast_dims):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    scale_shape, shift_shape, gate_shape = broadcast_dims
    run_case_broadcast(
        dtype=dtype,
        M=M,
        N=N,
        scale_shape=scale_shape,
        shift_shape=shift_shape,
        gate_shape=gate_shape,
        eps=1e-5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
