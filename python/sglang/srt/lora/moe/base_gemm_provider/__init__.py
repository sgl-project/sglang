"""Lazy selection of base-MoE providers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

logger = logging.getLogger(__name__)

# The first vendor is the default for each weight family.
VENDORS = {
    "bf16": ("cutedsl", "triton"),
    "fp8": ("cutedsl", "triton"),
    "nvfp4": ("marlin",),
}


def select_provider_cls(
    base_gemm_rows: str, family: str, vendor: str | None = None
) -> type[MoeBaseProvider]:
    """Use the family default when the requested vendor cannot serve it.

    Triton and Marlin use route-major rows for either requested row order.
    """
    if base_gemm_rows not in ("expert_major", "route_major"):
        raise ValueError(
            f"unknown MoE LoRA base-GEMM row order {base_gemm_rows!r}; "
            "expected 'expert_major' or 'route_major'"
        )
    vendors = VENDORS[family]
    if vendor not in vendors:
        if vendor is not None:
            logger.info(
                "MoE LoRA base-GEMM vendor %r serves no %s layers; this layer "
                "family uses its default vendor %r",
                vendor,
                family,
                vendors[0],
            )
        vendor = vendors[0]
    expert_major = base_gemm_rows == "expert_major"
    match vendor, family:
        case "cutedsl", "bf16":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
                CuteDslBf16ContiguousProvider,
                CuteDslBf16MaskedProvider,
            )

            return (
                CuteDslBf16MaskedProvider
                if expert_major
                else CuteDslBf16ContiguousProvider
            )
        case "cutedsl", "fp8":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_fp8 import (
                CuteDslFp8ContiguousProvider,
                CuteDslFp8MaskedProvider,
            )

            return (
                CuteDslFp8MaskedProvider
                if expert_major
                else CuteDslFp8ContiguousProvider
            )
        case "triton", "bf16":
            from sglang.srt.lora.moe.base_gemm_provider.triton_bf16 import (
                TritonBf16ContiguousProvider,
            )

            return TritonBf16ContiguousProvider
        case "triton", "fp8":
            from sglang.srt.lora.moe.base_gemm_provider.triton_fp8 import (
                TritonFp8ContiguousProvider,
            )

            return TritonFp8ContiguousProvider
        case "marlin", "nvfp4":
            from sglang.srt.lora.moe.base_gemm_provider.marlin_nvfp4 import (
                MarlinNvFp4ContiguousProvider,
            )

            return MarlinNvFp4ContiguousProvider
