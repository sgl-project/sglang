"""Experimental TRT-LLM LoRA kernel variants (gated by ``SGLANG_EXPERIMENTAL_LORA_OPTI`` / ``lora_envs``).

Migrated from ``sglang.srt.lora.trtllm_lora_temp.triton_ops`` (RFC #29630)."""

# --- merged from sglang.kernels.ops.moe.trtllm_lora_temp (RFC #29630 Phase 4) ---
from sglang.kernels.ops.moe.trtllm_lora_temp.core import (
    trtllm_bf16_routed_moe_lora,
    trtllm_fp4_block_scale_moe_lora_finalize,
    trtllm_fp4_block_scale_routed_moe_lora,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_moe_lora_finalize,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fp8_block_scale_routed_moe_lora,
)

__all__ = [
    "trtllm_bf16_routed_moe_lora",
    "trtllm_fp4_block_scale_moe_lora_finalize",
    "trtllm_fp4_block_scale_routed_moe_lora",
    "trtllm_fp8_block_scale_moe_lora_finalize",
    "trtllm_fp8_block_scale_moe",
    "trtllm_fp8_block_scale_routed_moe",
    "trtllm_fp8_block_scale_routed_moe_lora",
]
