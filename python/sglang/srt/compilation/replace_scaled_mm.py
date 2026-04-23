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
    _FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

    def _cutlass_wrapper(*args, **kwargs):
        # aten._scaled_mm.default positional:
        #   (self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum)
        mat_a = args[0]
        mat_b = args[1]
        scale_a = kwargs.get("scale_a", args[2] if len(args) > 2 else None)
        scale_b = kwargs.get("scale_b", args[3] if len(args) > 3 else None)
        bias = kwargs.get("bias", args[4] if len(args) > 4 else None)
        scale_result = kwargs.get("scale_result", args[5] if len(args) > 5 else None)
        out_dtype = kwargs.get("out_dtype", args[6] if len(args) > 6 else None)
        use_fast_accum = kwargs.get(
            "use_fast_accum", args[7] if len(args) > 7 else False
        )
        out = kwargs.get("out", None)

        # sgl_kernel.fp8_scaled_mm implements C = scale_a*scale_b*(A@B) + bias.
        # scale_result (output rescaling) and use_fast_accum aren't supported,
        # so fall through rather than silently drop them.
        # mat_b shape %16 matches CUTLASS SM90 FP8 tile alignment (element count).
        if (
            mat_a.dtype in _FP8
            and mat_b.dtype in _FP8
            and mat_b.shape[0] % 16 == 0
            and mat_b.shape[1] % 16 == 0
            and scale_a is not None
            and scale_b is not None
            and out_dtype is not None
            and scale_result is None
            and not use_fast_accum
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

        return original_fn(*args, **kwargs)

    extern_kernels._scaled_mm = _cutlass_wrapper
    _installed = True
    logger.info("Replaced extern_kernels._scaled_mm with CUTLASS")
