"""Replace extern_kernels._scaled_mm with CUTLASS at runtime."""

import logging

import torch

logger = logging.getLogger(__name__)
_installed = False


def install_cutlass_scaled_mm():
    global _installed
    if _installed:
        return

    from sgl_kernel import fp8_scaled_mm
    from torch._inductor.select_algorithm import extern_kernels

    original_fn = extern_kernels._scaled_mm

    def _cutlass_wrapper(*args, **kwargs):
        mat_a = args[0]
        mat_b = args[1]
        is_fp8 = mat_a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        aligned = mat_b.shape[0] % 16 == 0 and mat_b.shape[1] % 16 == 0

        if is_fp8 and aligned:
            scale_a = kwargs.get("scale_a", args[2] if len(args) > 2 else None)
            scale_b = kwargs.get("scale_b", args[3] if len(args) > 3 else None)
            # aten._scaled_mm positional: (mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum)
            bias = kwargs.get("bias", args[4] if len(args) > 4 else None)
            out_dtype = kwargs.get("out_dtype", args[6] if len(args) > 6 else None)
            out = kwargs.get("out", None)

            if out_dtype is not None and scale_a is not None and scale_b is not None:
                return fp8_scaled_mm(
                    mat_a,
                    mat_b,
                    scale_a,
                    scale_b,
                    out_dtype=out_dtype,
                    bias=bias,
                    out=out,
                )

        return original_fn(*args, **kwargs)

    extern_kernels._scaled_mm = _cutlass_wrapper
    _installed = True
    logger.info("Replaced extern_kernels._scaled_mm with CUTLASS")
