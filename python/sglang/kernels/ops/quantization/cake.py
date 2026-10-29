"""Cake (FlashInfer) backends for the ``quantization`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels.quantization`, which import
FlashInfer only when a kernel is actually called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

# Exact sm_100a / sm_103a (compute capability 10.0 / 10.3); the adapters'
# ``supports_*`` reject 10.x parts the generated programs were not built for.
_BLACKWELL_DC = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})

register_kernel(
    KernelSpec(
        op="quantization.nvfp4_quantize_per_token",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.quantization:nvfp4_quantize_per_token",
        capabilities=_BLACKWELL_DC,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16", "float16"),
            description=(
                "BF16/FP16 x [M, K] (K % 16 == 0) + global scale inverse -> "
                "(fp4 uint8 [M, K/2], sf uint8 swizzled 128x4 "
                "[round_up(M,128), round_up(K/16,4)], per_token_scale f32 [M])"
            ),
        ),
        description=(
            "Cake NVFP4 per-token activation quantization distributed by "
            "FlashInfer (nvfp4_quantize backend='cake', per_token_activation=True)."
        ),
    )
)

register_kernel(
    KernelSpec(
        op="quantization.mxfp8_grouped_quantize",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.quantization:mxfp8_grouped_quantize",
        capabilities=_BLACKWELL_DC,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16", "float16"),
            description=(
                "BF16/FP16 a [B, M, K] (K % 32 == 0) + int32 mask [B] -> "
                "(x_q e4m3 logical [M, padded_K, B], sf uint8 logical "
                "[32, 4, padded_M//128, 4, padded_K//128, B]); rows >= mask[i] "
                "unspecified"
            ),
        ),
        description=(
            "Cake grouped MXFP8 (UE8M0 block-32 scales) quantization distributed "
            "by FlashInfer (mxfp8_grouped_quantize backend='cake')."
        ),
    )
)

register_kernel(
    KernelSpec(
        op="quantization.sage_fp8_quantize",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.quantization:sage_fp8_quantize",
        capabilities=_BLACKWELL_DC,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description=(
                "BF16 BSHD q [B,Sq,H,128], k/v [B,Sk,Hkv,128] -> e4m3 (q8, k8, v8) + "
                "f32 q_scale [B,H,Sq], k_scale [B,Hkv,ceil(Sk/16)], v_scale [B,Hkv,128]"
            ),
        ),
        description=(
            "Cake Sage-FP8 Q/K/V quantization for SM100 block-sparse Sage "
            "attention distributed by FlashInfer."
        ),
    )
)


def cake_nvfp4_quantize_per_token(
    x: torch.Tensor,
    global_scale_inv,
    *,
    out_scale: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return get_kernel(
        "quantization.nvfp4_quantize_per_token", KernelBackend.FLASHINFER
    )(x, global_scale_inv, out_scale=out_scale, enable_pdl=enable_pdl)


def cake_mxfp8_grouped_quantize(
    a: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return get_kernel("quantization.mxfp8_grouped_quantize", KernelBackend.FLASHINFER)(
        a, mask
    )


def cake_sage_fp8_quantize(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return get_kernel("quantization.sage_fp8_quantize", KernelBackend.FLASHINFER)(
        q, k, v
    )


__all__ = [
    "cake_mxfp8_grouped_quantize",
    "cake_nvfp4_quantize_per_token",
    "cake_sage_fp8_quantize",
]
