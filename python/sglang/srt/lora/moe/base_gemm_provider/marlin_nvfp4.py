"""Marlin NVFP4: group-16 E4M3 scales under per-expert global scales."""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.marlin_common import (
    MarlinContiguousProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo


class MarlinNvFp4ContiguousProvider(MarlinContiguousProvider):
    contract = MoeBaseProviderContract(
        key="marlin_nvfp4_contiguous",
        quant_info_cls=MoeLoraNvFp4MarlinQuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )
    _use_atomic_add = True

    def __init__(self, quant_info: MoeLoraNvFp4MarlinQuantInfo):
        super().__init__(
            quant_info,
            global_scales=(quant_info.w13_global_scale, quant_info.w2_global_scale),
        )
