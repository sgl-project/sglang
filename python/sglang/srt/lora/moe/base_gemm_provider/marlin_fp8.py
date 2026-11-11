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
    # 16-row blocks from 6.5 pairs per expert on: past that enough experts
    # overflow a block of 8 that the E4M3 GEMMs are faster in blocks of 16.
    # Measured for 128-wide scale groups at the 2048 x 512 routed-expert shape
    # on compute capability 9.0; other layouts, shapes and architectures keep
    # the shared rule.
    _early_block_16_fill = ((8, 0.8125), (16, 0.9), (32, 0.9), (48, 0.9))
    _early_block_16_shape = (128, 2048, 512)  # group size, hidden, intermediate
    _early_block_16_capability = (9, 0)

    def __init__(self, quant_info: MoeLoraFp8MarlinQuantInfo):
        super().__init__(quant_info, global_scales=(None, None))
        device = quant_info.w13_qweight.device
        # The case _block_size_m was measured for: SM103 devices with 152 SMs,
        # 32-wide K groups, this layer shape and expert count.
        self._blocks_of_48 = (
            quant_info.group_size == 32
            and (quant_info.hidden_size, quant_info.intermediate_size) == (2048, 512)
            and quant_info.num_local_experts == 256
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == (10, 3)
            and torch.cuda.get_device_properties(device).multi_processor_count == 152
        )

    @classmethod
    def _block_fill_for(cls, quant_info) -> tuple[tuple[int, float], ...]:
        shape = (
            quant_info.group_size,
            quant_info.hidden_size,
            quant_info.intermediate_size,
        )
        if shape == cls._early_block_16_shape and (
            torch.cuda.get_device_capability(quant_info.w13_qweight.device)
            == cls._early_block_16_capability
        ):
            return cls._early_block_16_fill
        return cls._block_fill

    def _block_size_m(self, num_tokens: int, top_k: int, num_experts: int) -> int:
        block_size_m = super()._block_size_m(num_tokens, top_k, num_experts)
        # Measured on the device and layer that _blocks_of_48 names, from 128
        # to 512 pairs per expert: prepare and the two GEMMs together take
        # less time in 48-row blocks than in the shared rule's 64-row blocks.
        if (
            block_size_m == 64
            and self._blocks_of_48
            and 128 <= num_tokens * top_k / num_experts <= 512
        ):
            return 48
        return block_size_m
