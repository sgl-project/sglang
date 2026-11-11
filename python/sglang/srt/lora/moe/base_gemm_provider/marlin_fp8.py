"""Marlin FP8 W8A16: E4M3 weights under channel-wise, 128-block or MXFP8 scales."""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.marlin_common import (
    MarlinContiguousProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo


class MarlinFp8ContiguousProvider(MarlinContiguousProvider):
    contract = MoeBaseProviderContract(
        key="marlin_fp8_contiguous",
        quant_info_cls=MoeLoraFp8MarlinQuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )
    _weight_bits = 8
    _is_fp8 = True
    # Activation-dtype scales: these kernels reduce split-K through atomic adds.
    _use_atomic_add = True

    def __init__(self, quant_info: MoeLoraFp8MarlinQuantInfo):
        super().__init__(quant_info, global_scales=(None, None))
