"""LoRA adapter kernels that only the LoRA runtime calls: ``dense/`` SGMV GEMMs,
``moe/`` fused MoE-LoRA, and the experimental TRT-LLM LoRA path."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# Triton kernels registered for inventory. Import them from their modules.
_TRITON_KERNELS = [
    ("dense.chunked_embedding_lora_a", "chunked_embedding_lora_a_forward"),
    ("dense.chunked_sgmv_expand", "chunked_sgmv_lora_expand_forward"),
    ("dense.chunked_sgmv_shrink", "chunked_sgmv_lora_shrink_forward"),
    ("dense.embedding_lora_a", "embedding_lora_a_fwd"),
    ("dense.gate_up_lora_b", "gate_up_lora_b_fwd"),
    ("dense.qkv_lora_b", "qkv_lora_b_fwd"),
    ("dense.sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("dense.sgemm_lora_b", "sgemm_lora_b_fwd"),
    ("dense.kv_b_lora_absorbed", "step_a_q_fwd"),
    ("dense.kv_b_lora_absorbed", "step_b_q_fwd"),
    ("dense.kv_b_lora_absorbed", "step_a_v_fwd"),
    ("dense.kv_b_lora_absorbed", "step_b_v_fwd"),
    ("moe.fused_moe_lora_kernel", "fused_moe_lora"),
    ("moe.virtual_experts", "merged_experts_fused_moe_lora_add"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"lora.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.lora.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__: list[str] = []
