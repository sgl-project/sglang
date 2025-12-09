import torch
import pytest
import sgl_kernel

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

    y_dev_fused = sgl_kernel.fused_layernorm_scale_shift(x, weight, bias, scale, shift)

    # Reference fused output: compute LN in fp32, then apply scale/shift in fp32, cast back
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
    y_gt_fused = (y_ln32 * (1.0 + s32) + sh32).to(dtype)

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
    y_dev_fused = sgl_kernel.fused_layernorm_scale_shift(x, weight, bias, scale4d, shift4d)

    # Reference in fp32
    x32 = x.float()
    w32 = weight.float()
    b32 = bias.float()
    sc4 = scale4d.float()
    sh4 = shift4d.float()
    mean = x32.mean(dim=1, keepdim=True)
    var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (x32 - mean) * inv_std
    y_ln32 = y_ln32 * w32 + b32

    # Map rows m -> (b,f) with frame_seqlen = S
    y_ref = torch.empty_like(y_ln32)
    for m in range(M):
        b = (m // (F * S)) % B
        s_in_b = m - b * F * S
        f = s_in_b // S
        y_ref[m] = y_ln32[m] * (1.0 + sc4[b, f, 0]) + sh4[b, f, 0]
    y_ref = y_ref.to(dtype)

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
    y_dev, residual_out_dev = torch.ops.sgl_kernel.fused_scale_residual_layernorm_scale_shift(
        residual, x, weight, bias, scale, shift, None
    )

    # Reference
    out32 = (residual.float() + x.float())  # gate==1
    mean = out32.mean(dim=1, keepdim=True)
    var = (out32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (out32 - mean) * inv_std
    y_ln32 = y_ln32 * weight.float() + bias.float()
    y_ref = (y_ln32 * (1.0 + scale.float()) + shift.float()).to(dtype)
    residual_ref = out32.to(dtype)

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

    y_dev, residual_out_dev = torch.ops.sgl_kernel.fused_scale_residual_layernorm_scale_shift(
        residual, x, weight, bias, scale, shift, gate
    )

    # Reference: reshape to [B,S,N] for broadcasting gate[b,1,N]
    x32 = x.float().view(B, S, N)
    r32 = residual.float().view(B, S, N)
    g32 = gate.float().view(B, 1, N)
    out32 = (r32 + x32 * g32).view(M, N)
    mean = out32.mean(dim=1, keepdim=True)
    var = (out32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (out32 - mean) * inv_std
    y_ln32 = y_ln32 * weight.float() + bias.float()
    y_ref = (y_ln32 * (1.0 + scale.float()) + shift.float()).to(dtype)
    residual_ref = out32.to(dtype)

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

    y_dev, residual_out_dev = torch.ops.sgl_kernel.fused_scale_residual_layernorm_scale_shift(
        residual, x, weight, bias, scale4d, shift4d, gate4d
    )

    # Reference: map rows m -> (b,f)
    x32 = x.float()
    r32 = residual.float()
    g4 = gate4d.float()
    out32 = torch.empty_like(x32)
    for m in range(M):
        b = (m // (F * S)) % B
        s_in_b = m - b * F * S
        f = s_in_b // S
        out32[m] = r32[m] + x32[m] * g4[b, f, 0]
    mean = out32.mean(dim=1, keepdim=True)
    var = (out32 - mean).pow(2).mean(dim=1, keepdim=True)
    inv_std = (var + eps).sqrt().reciprocal()
    y_ln32 = (out32 - mean) * inv_std
    y_ln32 = y_ln32 * weight.float() + bias.float()
    sc4 = scale4d.float()
    sh4 = shift4d.float()
    y_ref = torch.empty_like(y_ln32)
    for m in range(M):
        b = (m // (F * S)) % B
        s_in_b = m - b * F * S
        f = s_in_b // S
        y_ref[m] = y_ln32[m] * (1.0 + sc4[b, f, 0]) + sh4[b, f, 0]
    y_ref = y_ref.to(dtype)
    residual_ref = out32.to(dtype)

    return y_dev, residual_out_dev, y_ref, residual_ref


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
]

def _tol(dtype: torch.dtype):
    return 2e-5 if dtype == torch.float32 else 2e-1

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
@pytest.mark.parametrize("B,F,S,N", [(2, 3, 4, 1024)])
def test_fused_layernorm_scale_shift_4d(dtype, B, F, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, y_ref = run_case_fused_4d_scale_accuracy(dtype=dtype, B=B, F=F, S=S, N=N, eps=1e-5)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(2, 5, 1024)])
def test_residual_gate_int(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_int(dtype=dtype, B=B, S=S, N=N, eps=1e-5)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype))

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,N", [(2, 5, 1024)])
def test_residual_gate_3d(dtype, B, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_3d(dtype=dtype, B=B, S=S, N=N, eps=1e-5)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype))

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,F,S,N", [(2, 3, 4, 1024)])
def test_residual_gate_4d(dtype, B, F, S, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    y_dev, residual_out_dev, y_ref, residual_ref = run_case_residual_gate_4d(dtype=dtype, B=B, F=F, S=S, N=N, eps=1e-5)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(residual_out_dev, residual_ref, atol=_tol(dtype), rtol=_tol(dtype))

if __name__ == "__main__":
    pytest.main([__file__])

