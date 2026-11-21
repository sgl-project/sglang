from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider


def select_provider_cls(
    base_gemm_rows: str, family: str, vendor: str
) -> type[MoeBaseProvider]:
    """Triton and Marlin keep raw pair rows for either plan layout."""
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
            from sglang.srt.lora.moe.base_gemm_provider.triton import (
                TritonBf16PairProvider,
            )

            return TritonBf16PairProvider
        case "triton", "fp8":
            from sglang.srt.lora.moe.base_gemm_provider.triton import (
                TritonFp8PairProvider,
            )

            return TritonFp8PairProvider
        case "marlin", "nvfp4":
            from sglang.srt.lora.moe.base_gemm_provider.marlin import (
                MarlinNvFp4PairProvider,
            )

            return MarlinNvFp4PairProvider
        case "marlin", "mxfp4":
            from sglang.srt.lora.moe.base_gemm_provider.marlin import (
                MarlinMxfp4PairProvider,
            )

            return MarlinMxfp4PairProvider
        case _:
            raise ValueError(
                f"no {vendor} MoE LoRA base-GEMM provider for {family} layers"
            )
