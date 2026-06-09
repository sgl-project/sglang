"""Experimental TRT-LLM LoRA kernels (copies of the upstream triton_ops kernels
with the SGLANG_EXPERIMENTAL_LORA_OPTI optimizations).

These are forked from ``sglang.srt.lora.triton_ops`` so the upstream kernels stay
byte-pristine; only the experimental forwards / dispatch in this package import
from here. The opt branches inside each kernel are still gated by ``lora_envs``
(master-gated by SGLANG_EXPERIMENTAL_LORA_OPTI).
"""

from .gate_up_lora_b import gate_up_lora_b_fwd
from .kv_b_lora_absorbed import (
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)
from .qkv_lora_b import qkv_lora_b_fwd
from .sgemm_lora_a import sgemm_lora_a_fwd
from .sgemm_lora_b import sgemm_lora_b_fwd
from .virtual_experts import merged_experts_fused_moe_lora_add

__all__ = [
    "gate_up_lora_b_fwd",
    "qkv_lora_b_fwd",
    "sgemm_lora_a_fwd",
    "sgemm_lora_b_fwd",
    "merged_experts_fused_moe_lora_add",
    "step_a_q_fwd",
    "step_a_v_fwd",
    "step_b_q_fwd",
    "step_b_v_fwd",
]
