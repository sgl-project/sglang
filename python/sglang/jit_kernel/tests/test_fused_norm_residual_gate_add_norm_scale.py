from typing import Optional, Tuple

import pytest
import torch
from einops import rearrange
from torch import Tensor

from sglang.jit_kernel.diffusion.cutedsl.norm_residual_gate_add_norm_scale import (
    fused_norm_residual_gate_add_norm_scale,
)

DEVICE = "cuda"
SHAPE_MAP = {
    "1": lambda B, S, F, D: (1,),
    "D": lambda B, S, F, D: (D,),
    "1D": lambda B, S, F, D: (1, D),
    "BD": lambda B, S, F, D: (B, D),
    "1BD": lambda B, S, F, D: (1, B, D),
    "11D": lambda B, S, F, D: (1, 1, D),
    "B1D": lambda B, S, F, D: (B, 1, D),
    "1SD": lambda B, S, F, D: (1, S, D),
    "BSD": lambda B, S, F, D: (B, S, D),
    "BF1D": lambda B, S, F, D: (B, F, 1, D),
}
SHAPES = [
    (1, 4160, 1, 3840),  # Z-Image
    (1, 1024, 8, 3072),
    (4, 512, 16, 3072),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
NORM_TYPES = ["layer", "rms"]
AFFINE_MODES = ["D", "NAT"]
INDEX_MODES = ["BSD", "1", "1SD", "BD", "B1D", "D", "1D", "11D", "BF1D"]


def _tol(dtype: torch.dtype):
    return 1e-5 if dtype == torch.float32 else 7e-2


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(42)


def _apply_scale(y: Tensor, scale: Tensor) -> Tensor:
    if scale.ndim == 4:
        num_frame = scale.shape[1]
        return rearrange(
            rearrange(y, "b (f l) d -> b f l d", f=num_frame) * (1 + scale),
            "b f l d -> b (f l) d",
        )
    scale = rearrange(scale, "b d -> b 1 d") if scale.ndim == 2 else scale
    return y * (1 + scale)


def fused_norm_residual_gate_add_norm_scale_ref(
    residual: Tensor,
    x: Tensor,
    gate: Optional[Tensor] | int,
    weight1: Optional[Tensor],
    bias1: Optional[Tensor],
    weight2: Optional[Tensor],
    bias2: Optional[Tensor],
    scale: Tensor,
    norm_type: str,
    eps: float,
):
    original_dtype = x.dtype
    residual, x, gate, weight1, bias1, weight2, bias2, scale = (
        v.float() if isinstance(v, Tensor) else v
        for v in [residual, x, gate, weight1, bias1, weight2, bias2, scale]
    )
    if norm_type == "layer":
        norm1 = torch.layer_norm(
            residual, residual.shape[-1:], eps=eps, weight=weight1, bias=bias1
        )
    else:
        norm1 = torch.rms_norm(residual, residual.shape[-1:], eps=eps, weight=weight1)
    if gate is None:
        residual_out = x + norm1
    elif gate.ndim == 4:
        num_frame = gate.shape[1]
        norm1_fld = rearrange(norm1, "b (f l) d -> b f l d", f=num_frame)
        residual_out = x + rearrange(norm1_fld * gate, "b f l d -> b (f l) d")
    else:
        gate = rearrange(gate, "b d -> b 1 d") if gate.ndim == 2 else gate
        residual_out = x + gate * norm1
    if norm_type == "layer":
        norm2 = torch.layer_norm(
            residual_out,
            residual_out.shape[-1:],
            eps=eps,
            weight=weight2,
            bias=bias2,
        )
    else:
        norm2 = torch.rms_norm(
            residual_out, residual_out.shape[-1:], eps=eps, weight=weight2
        )
    y_ref = _apply_scale(norm2, scale)
    return y_ref.to(original_dtype), residual_out.to(original_dtype)


def _make_tensor(index_mode: str, shape: Tuple, dtype: torch.dtype):
    if index_mode == "NAT":
        return None
    return torch.randn(*SHAPE_MAP[index_mode](*shape), device=DEVICE, dtype=dtype)


@torch.no_grad()
def run_norm_residual_gate_add_norm_scale(
    shape=SHAPES[0],
    dtype=DTYPES[0],
    affine_dtype=DTYPES[0],
    scale_dtype=DTYPES[0],
    norm_type=NORM_TYPES[0],
    affine_mode=AFFINE_MODES[0],
    gate_mode="B1D",
    scale_mode="BSD",
    eps=1e-5,
):
    residual = _make_tensor("BSD", shape, dtype)
    x = _make_tensor("BSD", shape, dtype)
    gate = _make_tensor(gate_mode, shape, dtype)
    weight1 = _make_tensor(affine_mode, shape, affine_dtype)
    bias1 = _make_tensor(affine_mode, shape, affine_dtype)
    weight2 = _make_tensor(affine_mode, shape, affine_dtype)
    bias2 = _make_tensor(affine_mode, shape, affine_dtype)
    scale = _make_tensor(scale_mode, shape, scale_dtype)
    y_dev, res_dev = fused_norm_residual_gate_add_norm_scale(
        x,
        residual,
        gate,
        weight1,
        bias1,
        weight2,
        bias2,
        scale,
        norm_type,
        eps,
    )
    y_ref, res_ref = fused_norm_residual_gate_add_norm_scale_ref(
        residual,
        x,
        gate,
        weight1,
        bias1,
        weight2,
        bias2,
        scale,
        norm_type,
        eps,
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(res_dev, res_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("norm_type", NORM_TYPES)
class TestFusedNormResidualGateAddNormScale:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shape_dtype(self, shape, dtype, norm_type):
        run_norm_residual_gate_add_norm_scale(
            shape=shape, dtype=dtype, norm_type=norm_type
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_0(self, dtype, norm_type):
        run_norm_residual_gate_add_norm_scale(affine_dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_1(self, dtype, norm_type):
        run_norm_residual_gate_add_norm_scale(scale_dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("affine_mode", AFFINE_MODES)
    def test_normtype_affine(self, affine_mode, norm_type):
        run_norm_residual_gate_add_norm_scale(
            affine_mode=affine_mode, norm_type=norm_type
        )

    @pytest.mark.parametrize("index_mode", INDEX_MODES)
    def test_scale_index_mode(self, index_mode, norm_type):
        run_norm_residual_gate_add_norm_scale(
            scale_mode=index_mode, norm_type=norm_type
        )

    @pytest.mark.parametrize("index_mode", INDEX_MODES)
    def test_gate_index_mode(self, index_mode, norm_type):
        run_norm_residual_gate_add_norm_scale(gate_mode=index_mode, norm_type=norm_type)


if __name__ == "__main__":
    pytest.main([__file__])
