from typing import Optional

import pytest
import torch
from einops import rearrange

from sglang.jit_kernel.diffusion.norm_fusion.fused_norm_scale_shift import (
    fused_norm_scale_shift,
)
from sglang.jit_kernel.diffusion.norm_fusion.fused_scale_residual_norm_scale_shift import (
    fused_scale_residual_norm_scale_shift,
)


def _tol(dtype: torch.dtype):
    return 2e-5 if dtype == torch.float32 else 5e-2


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# --- Reference implementations ---


def _apply_norm(x32: torch.Tensor, norm_type: str, eps: float) -> torch.Tensor:
    """Apply LayerNorm or RMSNorm in fp32."""
    if norm_type == "layer":
        mean = x32.mean(dim=-1, keepdim=True)
        var = (x32 - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x32 - mean) * (var + eps).rsqrt()
    else:  # rms
        mean_sq = (x32 * x32).mean(dim=-1, keepdim=True)
        return x32 * (mean_sq + eps).rsqrt()


def _apply_scale_shift(
    y_ln32: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Apply scale/shift with proper broadcasting."""
    S = y_ln32.shape[1]
    s32, sh32 = scale.float(), shift.float()
    if s32.ndim == 4:
        num_frame = s32.shape[1]
        frame_len = S // num_frame
        out32 = (
            y_ln32.unflatten(dim=1, sizes=(num_frame, frame_len)) * (1 + s32) + sh32
        ).flatten(1, 2)
    else:
        if s32.dim() == 2:
            s32 = rearrange(s32, "b d -> b 1 d")
        if sh32.dim() == 2:
            sh32 = rearrange(sh32, "b d -> b 1 d")
        out32 = y_ln32 * (1 + s32) + sh32
    return out32


@torch.no_grad()
def fused_norm_scale_shift_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_ln32 = _apply_norm(x.float(), norm_type, eps)
    if weight is not None:
        y_ln32 = y_ln32 * weight.float()
        if norm_type == "layer" and bias is not None:
            y_ln32 = y_ln32 + bias.float()
    return _apply_scale_shift(y_ln32, scale, shift).to(dtype)


@torch.no_grad()
def fused_scale_residual_norm_scale_shift_ref(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float,
    dtype: torch.dtype,
):
    r32, x32 = residual.float(), x.float()
    g32 = gate.float() if gate is not None else 1
    # Compute residual + gate * x
    if gate is None:
        out32 = r32 + 1 * x32
    else:
        if gate.ndim == 4:
            num_frame = gate.shape[1]
            frame_len = x.shape[1] // num_frame
            out32 = residual + (
                x.unflatten(dim=1, sizes=(num_frame, frame_len)) * gate
            ).flatten(1, 2)
        else:
            if g32.dim() == 2:
                g32 = rearrange(g32, "b d -> b 1 d")
            out32 = r32 + g32 * x32
    # Apply norm
    y_ln32 = _apply_norm(out32, norm_type, eps)
    if weight is not None:
        y_ln32 = y_ln32 * weight.float()
        if norm_type == "layer" and bias is not None:
            y_ln32 = y_ln32 + bias.float()
    y_ref = _apply_scale_shift(y_ln32, scale, shift)
    return y_ref.to(dtype), out32.to(dtype)


# --- Test runner helpers ---


def _make_tensors(B, S, D, dtype, device="cuda", with_affine=True):
    """Create common test tensors."""
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype) if with_affine else None
    bias = torch.randn(D, device=device, dtype=dtype) if with_affine else None
    return x, weight, bias


@torch.no_grad()
def run_fused_norm_test(
    dtype,
    B,
    S,
    D,
    norm_type,
    eps=1e-5,
    with_affine=True,
    scale_shape=None,
    shift_shape=None,
):
    """Run fused_norm_scale_shift test with given configuration."""
    device = "cuda"
    x, weight, bias = _make_tensors(B, S, D, dtype, device, with_affine)

    if scale_shape:
        scale = torch.randn(*scale_shape, device=device, dtype=dtype)
        shift = torch.randn(*(shift_shape or scale_shape), device=device, dtype=dtype)
    else:
        scale = torch.randn(B, S, D, device=device, dtype=dtype)
        shift = torch.randn(B, S, D, device=device, dtype=dtype)

    y_dev = fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps)
    y_ref = fused_norm_scale_shift_ref(
        x, weight, bias, scale, shift, norm_type, eps, dtype
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@torch.no_grad()
def run_residual_test(
    dtype,
    B,
    S,
    D,
    norm_type,
    eps=1e-5,
    with_affine=True,
    gate_shape=None,
    scale_shape=None,
    shift_shape=None,
):
    """Run fused_scale_residual_norm_scale_shift test with given configuration."""
    device = "cuda"
    x, weight, bias = _make_tensors(B, S, D, dtype, device, with_affine)
    residual = torch.randn(B, S, D, device=device, dtype=dtype)

    gate = torch.randn(*gate_shape, device=device, dtype=dtype) if gate_shape else None

    if scale_shape:
        scale = torch.randn(*scale_shape, device=device, dtype=dtype)
        shift = torch.randn(*(shift_shape or scale_shape), device=device, dtype=dtype)
    else:
        scale = torch.randn(B, S, D, device=device, dtype=dtype)
        shift = torch.randn(B, S, D, device=device, dtype=dtype)

    y_dev, res_dev = fused_scale_residual_norm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, norm_type, eps
    )
    y_ref, res_ref = fused_scale_residual_norm_scale_shift_ref(
        residual, x, gate, weight, bias, scale, shift, norm_type, eps, dtype
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(res_dev, res_ref, atol=_tol(dtype), rtol=_tol(dtype))


# --- Parameterized Tests ---

CASES_3D = [
    (1, 20, 3072),
    (1, 128, 3072),
    (2, 128, 3072),
    (2, 256, 3072),
    (4, 256, 3072),
    (2, 1000, 3072),
    (2, 1024, 3072),
    (1, 115200, 3072),  # Hunyuan
    (1, 5, 3072),
    (1, 32760, 1536),  # Wan
    (1, 2025, 3072),
    (1, 9, 3072),
    (1, 6, 3072),  # Qwen
]

# CASES_4D = [(2, 12, 4, 1024), (12, 24, 1, 2048)]
CASES_4D = [(2, 12, 4, 1024)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,D", CASES_3D)
@pytest.mark.parametrize("norm_type", ["layer", "rms"])
class TestFusedNorm2D:
    def test_with_affine(self, dtype, B, S, D, norm_type):
        if D % 4 != 0:
            pytest.skip("Vectorized kernel requires D % 4 == 0")
        run_fused_norm_test(dtype, B, S, D, norm_type, with_affine=True)

    def test_no_affine(self, dtype, B, S, D, norm_type):
        if D % 4 != 0:
            pytest.skip("Vectorized kernel requires D % 4 == 0")
        run_fused_norm_test(dtype, B, S, D, norm_type, with_affine=False)


@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,F,D", CASES_4D)
# @pytest.mark.parametrize("norm_type", ["layer", "rms"])
@pytest.mark.parametrize("norm_type", ["rms"])
def test_fused_norm_4d(dtype, B, S, F, D, norm_type):
    run_fused_norm_test(dtype, B, S, D, norm_type, scale_shape=(B, F, 1, D))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_type", ["layer", "rms"])
class TestResidualGate:
    @pytest.mark.parametrize("B,S,D", [(2, 5, 1024), (16, 32, 4096), (1, 32760, 1536)])
    def test_no_gate(self, dtype, B, S, D, norm_type):
        run_residual_test(dtype, B, S, D, norm_type)

    @pytest.mark.parametrize("B,S,D", [(2, 5, 1024), (12, 24, 2048)])
    def test_gate_3d(self, dtype, B, S, D, norm_type):
        run_residual_test(dtype, B, S, D, norm_type, gate_shape=(B, 1, D))

    @pytest.mark.parametrize("B,S,F,D", CASES_4D)
    def test_gate_4d(self, dtype, B, S, F, D, norm_type):
        run_residual_test(
            dtype, B, S, D, norm_type, gate_shape=(B, F, 1, D), scale_shape=(B, F, 1, D)
        )

    @pytest.mark.parametrize("B,S,D", [(1, 115200, 3072), (1, 5, 3072)])
    def test_fully_expanded(self, dtype, B, S, D, norm_type):
        run_residual_test(dtype, B, S, D, norm_type, gate_shape=(B, S, D))

    @pytest.mark.parametrize("B,S,D", [(1, 32760, 1536)])
    def test_gate_3d_scalar_scale(self, dtype, B, S, D, norm_type):
        run_residual_test(
            dtype, B, S, D, norm_type, gate_shape=(B, 1, D), scale_shape=(1,)
        )

    @pytest.mark.parametrize("B,S,D", [(1, 32760, 1536)])
    def test_gate_3d_scalar_scale_no_affine(self, dtype, B, S, D, norm_type):
        run_residual_test(
            dtype,
            B,
            S,
            D,
            norm_type,
            with_affine=False,
            gate_shape=(B, 1, D),
            scale_shape=(1,),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_type", ["layer", "rms"])
@pytest.mark.parametrize(
    "broadcast_dims",
    [
        ((1, 1024), None),  # scale/shift 1xN
        ((1, 1, 1024), None),  # scale/shift 1x1xN
        ((1, 1024), (1, 1024)),  # all 1xN with gate
        ((1, 1, 1024), (1, 1, 1024)),  # all 1x1xN with gate
        ((128, 1024), (1, 1024)),  # gate broadcast only
        ((1, 1024), (128, 1024)),  # scale/shift broadcast only
        ((1, 1, 1024), (128, 1024)),  # scale/shift 3d broadcast
    ],
)
def test_broadcast(dtype, norm_type, broadcast_dims):
    B, S, D = 128, 64, 1024
    scale_shape, gate_shape = broadcast_dims

    # Test fused_norm_scale_shift (no gate)
    if gate_shape is None:
        run_fused_norm_test(dtype, B, S, D, norm_type, scale_shape=scale_shape)
        run_fused_norm_test(
            dtype, B, S, D, norm_type, with_affine=False, scale_shape=scale_shape
        )

    # Test residual variant
    run_residual_test(
        dtype, B, S, D, norm_type, gate_shape=gate_shape, scale_shape=scale_shape
    )


if __name__ == "__main__":
    pytest.main([__file__])
