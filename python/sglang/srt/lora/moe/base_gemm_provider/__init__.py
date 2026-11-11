"""Lazy selection of base-MoE providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider


def select_provider_cls(
    base_gemm_rows: str, family: str, vendor: str
) -> type[MoeBaseProvider]:
    """Triton and Marlin use route-major rows for either requested row order."""
    if base_gemm_rows not in ("expert_major", "route_major"):
        raise ValueError(
            f"unknown MoE LoRA base-GEMM row order {base_gemm_rows!r}; "
            "expected 'expert_major' or 'route_major'"
        )
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
        case "triton", "fp8_channel":
            from sglang.srt.lora.moe.base_gemm_provider.triton_fp8 import (
                TritonFp8ChannelContiguousProvider,
            )

            return TritonFp8ChannelContiguousProvider
        case "triton", "fp8_tensor":
            from sglang.srt.lora.moe.base_gemm_provider.triton_fp8 import (
                TritonFp8TensorContiguousProvider,
            )

            return TritonFp8TensorContiguousProvider
        case "cutedsl", "fp8_channel":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_fp8_channel import (
                CuteDslFp8ChannelContiguousProvider,
                CuteDslFp8ChannelMaskedProvider,
            )

            return (
                CuteDslFp8ChannelMaskedProvider
                if expert_major
                else CuteDslFp8ChannelContiguousProvider
            )
        case "marlin", "nvfp4":
            from sglang.srt.lora.moe.base_gemm_provider.marlin_nvfp4 import (
                MarlinNvFp4ContiguousProvider,
            )

            return MarlinNvFp4ContiguousProvider
        case "marlin", "mxfp4":
            from sglang.srt.lora.moe.base_gemm_provider.marlin_mxfp4 import (
                MarlinMxfp4ContiguousProvider,
            )

            return MarlinMxfp4ContiguousProvider
        case "marlin", ("fp8" | "fp8_channel" | "fp8_tensor" | "mxfp8"):
            # One W8A16 provider: the repack folds every FP8 scale layout into
            # Marlin groups, so only the group size differs per family.
            from sglang.srt.lora.moe.base_gemm_provider.marlin_fp8 import (
                MarlinFp8ContiguousProvider,
            )

            return MarlinFp8ContiguousProvider
        case "cutedsl", "mxfp4":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_blockscaled import (
                CuteDslMxfp4ContiguousProvider,
                CuteDslMxfp4MaskedProvider,
            )

            return (
                CuteDslMxfp4MaskedProvider
                if expert_major
                else CuteDslMxfp4ContiguousProvider
            )
        case "cutedsl", "nvfp4":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_blockscaled import (
                CuteDslNvFp4ContiguousProvider,
                CuteDslNvFp4MaskedProvider,
            )

            return (
                CuteDslNvFp4MaskedProvider
                if expert_major
                else CuteDslNvFp4ContiguousProvider
            )
        case "cutedsl", "mxfp8":
            from sglang.srt.lora.moe.base_gemm_provider.cutedsl_blockscaled import (
                CuteDslMxfp8ContiguousProvider,
                CuteDslMxfp8MaskedProvider,
            )

            return (
                CuteDslMxfp8MaskedProvider
                if expert_major
                else CuteDslMxfp8ContiguousProvider
            )
        case _:
            raise ValueError(
                f"no {vendor} MoE LoRA base-GEMM provider for {family} layers"
            )
