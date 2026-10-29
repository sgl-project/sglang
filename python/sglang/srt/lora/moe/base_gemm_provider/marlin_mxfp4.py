"""Marlin MXFP4: group-32 E8M0 scales without a global scale."""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.marlin_common import (
    MarlinContiguousProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo


class MarlinMxfp4ContiguousProvider(MarlinContiguousProvider):
    contract = MoeBaseProviderContract(
        key="marlin_mxfp4_contiguous",
        quant_info_cls=MoeLoraMxfp4MarlinQuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    def __init__(self, quant_info: MoeLoraMxfp4MarlinQuantInfo):
        super().__init__(quant_info, global_scales=(None, None))
