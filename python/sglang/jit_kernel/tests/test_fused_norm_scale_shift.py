from typing import Optional, Tuple

import pytest
import torch
from einops import rearrange
from torch import Tensor

from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
    fused_norm_scale_shift,
    fused_scale_residual_norm_scale_shift,
)

DEVICE = "cuda"
SHAPE_MAP = {
    "1": lambda B, S, F, D: (1,),
    "D": lambda B, S, F, D: (D,),
    "1D": lambda B, S, F, D: (1, D),
    "BD": lambda B, S, F, D: (B, D),
    "11D": lambda B, S, F, D: (1, 1, D),
    "B1D": lambda B, S, F, D: (B, 1, D),
    "1SD": lambda B, S, F, D: (1, S, D),
    "BSD": lambda B, S, F, D: (B, S, D),
    "BF1D": lambda B, S, F, D: (B, F, 1, D),
}
SHAPES = [
    # (B, S, F, D)
    (1, 115200, 1, 3072),  # Hunyuan
    (1, 32760, 1, 1536),  # Wan
    (1, 6, 1, 3072),  # Qwen
    (1, 1024, 8, 3072),
    (4, 512, 16, 3072),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
NORM_TYPES = ["layer", "rms"]
AFFINE_MODES = ["D", "NAT"]
INDEX_MODES = ["BSD", "1", "1SD", "BD", "B1D", "D", "1D", "11D", "BF1D"]


def _tol(dtype: torch.dtype):
    return 1e-5 if dtype == torch.float32 else 5e-2


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _apply_scale_shift(y: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    if scale.ndim == 4:
        num_frame = scale.shape[1]
        return rearrange(
            rearrange(y, "b (f l) d -> b f l d", f=num_frame) * (1 + scale) + shift,
            "b f l d -> b (f l) d",
        )
    else:
        scale = rearrange(scale, "b d -> b 1 d") if scale.ndim == 2 else scale
        shift = rearrange(shift, "b d -> b 1 d") if shift.ndim == 2 else shift
        return y * (1 + scale) + shift


def fused_norm_scale_shift_ref(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    scale: Tensor,
    shift: Tensor,
    norm_type: str,
    eps: float,
) -> Tensor:
    original_dtype = x.dtype
    x, weight, bias, scale, shift = (
        v.float() if v is not None else v for v in [x, weight, bias, scale, shift]
    )
    if norm_type == "layer":
        norm = torch.layer_norm(x, x.shape[-1:], eps=eps, weight=weight, bias=bias)
    else:
        norm = torch.rms_norm(x, x.shape[-1:], eps=eps, weight=weight)
    return _apply_scale_shift(norm, scale, shift).to(original_dtype)


def fused_scale_residual_norm_scale_shift_ref(
    residual: Tensor,
    x: Tensor,
    gate: Optional[Tensor] | int,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    scale: Tensor,
    shift: Tensor,
    norm_type: str,
    eps: float,
):
    original_dtype = x.dtype
    residual, x, gate, weight, bias, scale, shift = (
        v.float() if isinstance(v, Tensor) else v
        for v in [residual, x, gate, weight, bias, scale, shift]
    )
    if isinstance(gate, int):
        x = residual + gate * x
    else:
        if gate.ndim == 4:
            num_frame = gate.shape[1]
            x_fld = rearrange(x, "b (f l) d -> b f l d", f=num_frame)
            x = residual + rearrange(x_fld * gate, "b f l d -> b (f l) d")
        else:
            gate = rearrange(gate, "b d -> b 1 d") if gate.ndim == 2 else gate
            x = residual + gate * x
    if norm_type == "layer":
        norm = torch.layer_norm(x, x.shape[-1:], eps=eps, weight=weight, bias=bias)
    else:
        norm = torch.rms_norm(x, x.shape[-1:], eps=eps, weight=weight)
    y_ref = _apply_scale_shift(norm, scale, shift)
    return y_ref.to(original_dtype), x.to(original_dtype)


def _make_tensor(index_mode: str, shape: Tuple, dtype: torch.dtype):
    if index_mode == "NAT":
        return None
    return torch.randn(*SHAPE_MAP[index_mode](*shape), device=DEVICE, dtype=dtype)


@torch.no_grad()
def run_norm_scale_shift(
    shape=SHAPES[0],
    dtype=DTYPES[0],
    affine_dtype=DTYPES[0],
    scale_dtype=DTYPES[0],
    shift_dtype=DTYPES[0],
    norm_type=NORM_TYPES[0],
    affine_mode=AFFINE_MODES[0],
    scale_mode="BSD",
    shift_mode="BSD",
    eps=1e-5,
):
    x = _make_tensor("BSD", shape, dtype)
    weight = _make_tensor(affine_mode, shape, affine_dtype)
    bias = _make_tensor(affine_mode, shape, affine_dtype)
    scale = _make_tensor(scale_mode, shape, scale_dtype)
    shift = _make_tensor(shift_mode, shape, shift_dtype)
    y_dev = fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps)
    y_ref = fused_norm_scale_shift_ref(x, weight, bias, scale, shift, norm_type, eps)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@torch.no_grad()
def run_scale_resi_norm_scale_shift(
    shape=SHAPES[0],
    dtype=DTYPES[0],
    affine_dtype=DTYPES[0],
    scale_dtype=DTYPES[0],
    shift_dtype=DTYPES[0],
    norm_type=NORM_TYPES[0],
    affine_mode=AFFINE_MODES[0],
    gate_mode="B1D",
    scale_mode="BSD",
    shift_mode="BSD",
    eps=1e-5,
):
    residual = _make_tensor("BSD", shape, dtype)
    x = _make_tensor("BSD", shape, dtype)
    gate = _make_tensor(gate_mode, shape, dtype)
    weight = _make_tensor(affine_mode, shape, affine_dtype)
    bias = _make_tensor(affine_mode, shape, affine_dtype)
    scale = _make_tensor(scale_mode, shape, scale_dtype)
    shift = _make_tensor(shift_mode, shape, shift_dtype)
    y_dev, res_dev = fused_scale_residual_norm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, norm_type, eps
    )
    y_ref, res_ref = fused_scale_residual_norm_scale_shift_ref(
        residual, x, gate, weight, bias, scale, shift, norm_type, eps
    )
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(res_dev, res_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("norm_type", NORM_TYPES)
class TestFusedNormScaleShift:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shape_dtype(self, shape, dtype, norm_type):
        run_norm_scale_shift(shape=shape, dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_0(self, dtype, norm_type):
        run_norm_scale_shift(affine_dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_1(self, dtype, norm_type):
        run_norm_scale_shift(scale_dtype=dtype, shift_dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("affine_mode", AFFINE_MODES)
    def test_normtype_affine(self, affine_mode, norm_type):
        run_norm_scale_shift(affine_mode=affine_mode, norm_type=norm_type)

    @pytest.mark.parametrize("index_mode", INDEX_MODES)
    def test_index_mode(self, index_mode, norm_type):
        run_norm_scale_shift(
            scale_mode=index_mode, shift_mode=index_mode, norm_type=norm_type
        )


@pytest.mark.parametrize("norm_type", NORM_TYPES)
class TestFusedScaleResidualNormScaleShift:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shape_dtype(self, shape, dtype, norm_type):
        run_scale_resi_norm_scale_shift(shape=shape, dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_0(self, dtype, norm_type):
        run_scale_resi_norm_scale_shift(affine_dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_1(self, dtype, norm_type):
        run_scale_resi_norm_scale_shift(
            scale_dtype=dtype, shift_dtype=dtype, norm_type=norm_type
        )

    @pytest.mark.parametrize("affine_mode", AFFINE_MODES)
    def test_normtype_affine(self, affine_mode, norm_type):
        run_scale_resi_norm_scale_shift(affine_mode=affine_mode, norm_type=norm_type)

    @pytest.mark.parametrize("index_mode", INDEX_MODES)
    def test_scale_shift_index_mode(self, index_mode, norm_type):
        run_scale_resi_norm_scale_shift(
            scale_mode=index_mode, shift_mode=index_mode, norm_type=norm_type
        )

    @pytest.mark.parametrize("index_mode", INDEX_MODES)
    def test_gate_index_mode(self, index_mode, norm_type):
        run_scale_resi_norm_scale_shift(gate_mode=index_mode, norm_type=norm_type)


if __name__ == "__main__":
    pytest.main([__file__])
