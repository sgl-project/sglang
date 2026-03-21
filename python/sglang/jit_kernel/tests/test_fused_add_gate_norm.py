from typing import Optional, Tuple

import pytest
import torch
from einops import rearrange
from torch import Tensor

from sglang.jit_kernel.diffusion.cutedsl.add_gate_norm import fused_add_gate_norm

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
    return 1e-5 if dtype == torch.float32 else 5e-2


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(42)


def fused_add_gate_norm_ref(
    x: Tensor,
    residual: Tensor,
    gate: Optional[Tensor] | int,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    norm_type: str,
    eps: float,
) -> Tensor:
    original_dtype = x.dtype
    residual, x, gate, weight, bias = (
        v.float() if isinstance(v, Tensor) else v
        for v in [residual, x, gate, weight, bias]
    )
    if norm_type == "layer":
        norm = torch.layer_norm(
            residual, residual.shape[-1:], eps=eps, weight=weight, bias=bias
        )
    else:
        norm = torch.rms_norm(residual, residual.shape[-1:], eps=eps, weight=weight)
    if gate is None:
        out = x + norm
    elif gate.ndim == 4:
        num_frame = gate.shape[1]
        norm_fld = rearrange(norm, "b (f l) d -> b f l d", f=num_frame)
        out = x + rearrange(norm_fld * gate, "b f l d -> b (f l) d")
    else:
        gate = rearrange(gate, "b d -> b 1 d") if gate.ndim == 2 else gate
        out = x + gate * norm
    return out.to(original_dtype)


def _make_tensor(index_mode: str, shape: Tuple, dtype: torch.dtype):
    if index_mode == "NAT":
        return None
    return torch.randn(*SHAPE_MAP[index_mode](*shape), device=DEVICE, dtype=dtype)


@torch.no_grad()
def run_add_gate_norm(
    shape=SHAPES[0],
    dtype=DTYPES[0],
    affine_dtype=DTYPES[0],
    norm_type=NORM_TYPES[0],
    affine_mode=AFFINE_MODES[0],
    gate_mode="B1D",
    eps=1e-5,
):
    residual = _make_tensor("BSD", shape, dtype)
    x = _make_tensor("BSD", shape, dtype)
    gate = _make_tensor(gate_mode, shape, dtype)
    weight = _make_tensor(affine_mode, shape, affine_dtype)
    bias = _make_tensor(affine_mode, shape, affine_dtype)
    y_dev = fused_add_gate_norm(x, residual, gate, weight, bias, norm_type, eps)
    y_ref = fused_add_gate_norm_ref(x, residual, gate, weight, bias, norm_type, eps)
    torch.testing.assert_close(y_dev, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("norm_type", NORM_TYPES)
class TestFusedAddGateNorm:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shape_dtype(self, shape, dtype, norm_type):
        run_add_gate_norm(shape=shape, dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_0(self, dtype, norm_type):
        run_add_gate_norm(affine_dtype=dtype, norm_type=norm_type)

    @pytest.mark.parametrize("affine_mode", AFFINE_MODES)
    def test_normtype_affine(self, affine_mode, norm_type):
        run_add_gate_norm(affine_mode=affine_mode, norm_type=norm_type)


if __name__ == "__main__":
    pytest.main([__file__])
