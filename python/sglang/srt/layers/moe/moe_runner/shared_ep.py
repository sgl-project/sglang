"""Runner-neutral quantization metadata for SharedEP composite execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend


class SharedEpQuantCapability(str, Enum):
    CANONICAL_BLOCK_FP8 = "canonical_block_fp8"
    CANONICAL_MXFP4 = "canonical_mxfp4"


class SharedEpQuantization(str, Enum):
    BLOCK_FP8 = "block_fp8"
    MXFP4 = "mxfp4"


class SharedEpWeightLayout(str, Enum):
    CANONICAL = "canonical"
    AITER_SHUFFLED = "aiter_shuffled"


class SharedEpScaleLayout(str, Enum):
    CANONICAL = "canonical"
    AITER_SHUFFLED = "aiter_shuffled"


@dataclass
class SharedEpQuantInfo(MoeQuantInfo):
    """Decode metadata plus backend-specific prefill metadata.

    SharedEP decode consumes canonical tensors. The nested fallback payload may
    use a backend-private layout such as AITER's shuffled MXFP4 weights/scales
    and is exposed only to the matching prefill runner. When
    ``fallback_uses_duplicate_tensors`` is true, that payload is the temporary
    MoRI+AITER prefill duplicate; it is not the production no-dup layout.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: Optional[torch.Tensor]
    w2_scale: Optional[torch.Tensor]
    block_shape: tuple[int, int]
    fallback_quant_info: MoeQuantInfo
    fallback_backend: MoeRunnerBackend
    quantization: SharedEpQuantization = SharedEpQuantization.BLOCK_FP8
    weight_layout: SharedEpWeightLayout = SharedEpWeightLayout.CANONICAL
    scale_layout: SharedEpScaleLayout = SharedEpScaleLayout.CANONICAL
    weight_group_size: Optional[int] = None
    scale_format: Optional[str] = None
    capabilities: frozenset[SharedEpQuantCapability] = frozenset(
        {SharedEpQuantCapability.CANONICAL_BLOCK_FP8}
    )
    fallback_weight_layout: SharedEpWeightLayout = SharedEpWeightLayout.CANONICAL
    fallback_scale_layout: SharedEpScaleLayout = SharedEpScaleLayout.CANONICAL
    fallback_uses_duplicate_tensors: bool = False

    def require_decode_capability(
        self,
        capability: SharedEpQuantCapability,
    ) -> None:
        if capability not in self.capabilities:
            raise TypeError(
                f"SharedEP quant metadata does not provide {capability.value}"
            )
        expected_quantization = {
            SharedEpQuantCapability.CANONICAL_BLOCK_FP8: SharedEpQuantization.BLOCK_FP8,
            SharedEpQuantCapability.CANONICAL_MXFP4: SharedEpQuantization.MXFP4,
        }[capability]
        if self.quantization is not expected_quantization:
            raise TypeError(
                f"SharedEP {capability.value} metadata cannot describe "
                f"{self.quantization.value} tensors"
            )
        if self.weight_layout is not SharedEpWeightLayout.CANONICAL:
            raise ValueError(
                "SharedEP decode requires canonical weights, got "
                f"{self.weight_layout.value}"
            )
        if self.scale_layout is not SharedEpScaleLayout.CANONICAL:
            raise ValueError(
                "SharedEP decode requires canonical scales, got "
                f"{self.scale_layout.value}"
            )
        if capability is SharedEpQuantCapability.CANONICAL_MXFP4:
            if self.weight_group_size != 32:
                raise ValueError(
                    "SharedEP MXFP4 requires E8M0 weight group size 32, got "
                    f"{self.weight_group_size}"
                )
            if self.scale_format != "e8m0":
                raise ValueError(
                    "SharedEP MXFP4 requires canonical E8M0 scales, got "
                    f"{self.scale_format!r}"
                )
        shuffled = [
            name
            for name, tensor in (
                ("w13_weight", self.w13_weight),
                ("w2_weight", self.w2_weight),
                ("w13_scale", self.w13_scale),
                ("w2_scale", self.w2_scale),
            )
            if tensor is not None and getattr(tensor, "is_shuffled", False)
        ]
        if shuffled:
            raise ValueError(
                "SharedEP decode cannot consume AITER-pre-shuffled tensors: "
                + ", ".join(shuffled)
            )

    def fallback_for(self, backend: MoeRunnerBackend) -> MoeQuantInfo:
        if backend is not self.fallback_backend:
            raise TypeError(
                "SharedEP prefill quant metadata targets "
                f"{self.fallback_backend.value}, not {backend.value}"
            )
        return self.fallback_quant_info
