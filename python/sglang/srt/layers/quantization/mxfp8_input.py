"""Explicit input layout for a linear consuming prequantized MXFP8 activations."""

from typing import NamedTuple

import torch


class Mxfp8SwizzledInput(NamedTuple):
    """E4M3 activations and UE8M0 scales in FlashInfer's 128x4 layout.

    A plain FP8 tuple may contain block-FP8 scales with a different layout.
    This marker lets a converted block-FP8 linear distinguish the two.
    """

    data: torch.Tensor
    scales: torch.Tensor


def accepts_mxfp8_swizzled_input(linear) -> bool:
    """Whether a linear can consume the explicit 128x4 producer contract."""
    qm = getattr(linear, "quant_method", None)
    backend = getattr(qm, "mxfp8_dense_backend", None)
    return bool(
        backend is not None
        and not getattr(qm, "use_marlin", False)
        and (
            getattr(qm, "use_mxfp8", False)
            or (
                getattr(qm, "block_fp8_as_mxfp8", False)
                and getattr(linear, "block_fp8_mxfp8_ready", False)
            )
        )
        and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
    )


def accepts_mxfp8_deepgemm_input(linear) -> bool:
    """Whether a linear takes a ``(q, scale)`` pair with DeepGEMM's packed UE8M0 scales."""
    from sglang.srt.layers import deep_gemm_wrapper

    qm = getattr(linear, "quant_method", None)
    backend = getattr(qm, "mxfp8_dense_backend", None)
    weight = getattr(linear, "weight", None)
    return bool(
        backend is not None
        and backend.is_deep_gemm()
        and getattr(qm, "use_mxfp8", False)
        and not getattr(qm, "use_marlin", False)
        and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        and weight is not None
        and weight.shape[0] % 64 == 0
        and weight.shape[1] % 128 == 0
    )
