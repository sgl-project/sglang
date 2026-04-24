"""Route aten::_scaled_mm on CUDA through CUTLASS fp8_scaled_mm via torch.library.

Option 2 (vs the previous extern_kernels monkey-patch): register a CUDA-backend
override for aten::_scaled_mm.default and .out using the official dispatch API.
Inductor's FX graph still captures aten._scaled_mm.default, so prologue fusion
(RMSNorm + per-tensor quant + GEMM) is preserved; only the runtime kernel changes.

Both overloads must be registered — inductor's generated code emits calls to the
.out form (extern_kernels._scaled_mm(..., out=buf)), while eager Python code hits
.default.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_installed = False
_lib: Optional[torch.library.Library] = None


def _cutlass_eligible(
    mat_a, mat_b, scale_a, scale_b, scale_result, use_fast_accum, out_dtype
) -> bool:
    # sgl_kernel.fp8_scaled_mm implements C = scale_a*scale_b*(A@B) + bias with
    # scalar or per-channel scales; it doesn't support output rescaling or
    # use_fast_accum. mat_b element counts must match CUTLASS SM90 FP8 tiling.
    return (
        mat_a.dtype in _FP8_DTYPES
        and mat_b.dtype in _FP8_DTYPES
        and mat_b.shape[0] % 16 == 0
        and mat_b.shape[1] % 16 == 0
        and scale_a is not None
        and scale_b is not None
        and out_dtype is not None
        and scale_result is None
        and not use_fast_accum
    )


def install_cutlass_scaled_mm() -> None:
    """Install CUTLASS override for aten::_scaled_mm on CUDA.

    Safe to call multiple times; only the first call registers. Registered
    kernels are process-global so this is effectively a singleton.
    """
    global _installed, _lib
    if _installed:
        return

    from sgl_kernel import fp8_scaled_mm

    _lib = torch.library.Library("aten", "IMPL")

    def _impl_default(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale_result: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        use_fast_accum: bool = False,
    ) -> torch.Tensor:
        # aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b,
        #   Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None,
        #   bool use_fast_accum=False) -> Tensor
        if _cutlass_eligible(
            mat_a, mat_b, scale_a, scale_b, scale_result, use_fast_accum, out_dtype
        ):
            return fp8_scaled_mm(
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                out_dtype=out_dtype,
                bias=bias,
            )
        # Non-CUTLASS shapes/features reach here. The old monkey-patch could call
        # the original extern_kernels._scaled_mm; at the dispatcher layer there's
        # no clean way to re-dispatch to the pre-override CUDA kernel, so surface
        # clearly. If this raises in practice, add a recomposition fallback.
        raise NotImplementedError(
            "aten::_scaled_mm called with arguments CUTLASS override doesn't "
            f"handle: a.dtype={mat_a.dtype}, b.dtype={mat_b.dtype}, "
            f"b.shape={tuple(mat_b.shape)}, scale_a={None if scale_a is None else scale_a.shape}, "
            f"scale_b={None if scale_b is None else scale_b.shape}, "
            f"scale_result={scale_result}, use_fast_accum={use_fast_accum}, "
            f"out_dtype={out_dtype}. "
            "Add a recomposition fallback in replace_scaled_mm._impl_default."
        )

    def _impl_out(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale_result: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        use_fast_accum: bool = False,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        # aten::_scaled_mm.out(..., *, Tensor(a!) out) -> Tensor(a!)
        # This is the overload inductor emits via extern_kernels._scaled_mm(..., out=buf).
        if _cutlass_eligible(
            mat_a, mat_b, scale_a, scale_b, scale_result, use_fast_accum, out_dtype
        ):
            return fp8_scaled_mm(
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                out_dtype=out_dtype,
                bias=bias,
                out=out,
            )
        raise NotImplementedError(
            "aten::_scaled_mm.out called with arguments CUTLASS override doesn't "
            f"handle: a.dtype={mat_a.dtype}, b.dtype={mat_b.dtype}, "
            f"b.shape={tuple(mat_b.shape)}, out.shape={tuple(out.shape)}, "
            f"scale_result={scale_result}, use_fast_accum={use_fast_accum}. "
            "Add a recomposition fallback in replace_scaled_mm._impl_out."
        )

    _lib.impl("_scaled_mm", _impl_default, "CUDA")
    _lib.impl("_scaled_mm.out", _impl_out, "CUDA")
    _installed = True
    logger.info(
        "Registered CUTLASS override for aten::_scaled_mm (.default + .out) on CUDA"
    )
