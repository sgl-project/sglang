"""LoRA adapter kernels: the adapter-side GEMMs, fused MoE-LoRA kernels, and the
experimental TRT-LLM LoRA path.

Layout (implementation modules; this group import stays metadata-only):

- ``dense/``: Triton SGMV / chunked-SGMV LoRA GEMMs and the absorbed-MLA
  ``kv_b_proj`` correction, plus the CSGMV tuning-config loader and its
  ``csgmv_configs/`` data.
- ``moe/``: fused MoE-LoRA Triton kernels, LoRA-aware block alignment (JIT),
  and the merged virtual-expert path.
- ``dense/trtllm_lora_temp/`` and ``moe/trtllm_lora_temp/``: the experimental
  TRT-LLM LoRA path, moved whole from ``ops/gemm`` and ``ops/moe`` and gated by
  ``SGLANG_EXPERIMENTAL_LORA_OPTI`` / ``lora_envs``. The MoE one also carries
  the FlashInfer TRT-LLM fused-MoE overlay (``core.py`` / ``jit.py`` /
  ``data/``), the merged-alignment kernels, and the fused top-k softmax +
  routed pack and Kimi-K2 fused gate that ``srt/layers/moe/topk.py`` reaches
  under those same experimental flags.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# Op ids keep their original ``gemm.`` / ``moe.`` prefixes (the computation
# class); only the implementation modules moved under this group.
_DENSE_TRITON_KERNELS = [
    ("chunked_embedding_lora_a", "chunked_embedding_lora_a_forward"),
    ("chunked_sgmv_expand", "chunked_sgmv_lora_expand_forward"),
    ("chunked_sgmv_shrink", "chunked_sgmv_lora_shrink_forward"),
    ("embedding_lora_a", "embedding_lora_a_fwd"),
    ("gate_up_lora_b", "gate_up_lora_b_fwd"),
    ("qkv_lora_b", "qkv_lora_b_fwd"),
    ("sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("sgemm_lora_b", "sgemm_lora_b_fwd"),
    ("kv_b_lora_absorbed", "step_a_q_fwd"),
    ("kv_b_lora_absorbed", "step_b_q_fwd"),
    ("kv_b_lora_absorbed", "step_a_v_fwd"),
    ("kv_b_lora_absorbed", "step_b_v_fwd"),
]
for _mod, _fn in _DENSE_TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"gemm.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.lora.dense.{_mod}:{_fn}",
        )
    )

_MOE_TRITON_KERNELS = [
    ("fused_moe_lora_kernel", "fused_moe_lora"),
    ("virtual_experts", "merged_experts_fused_moe_lora_add"),
]
for _mod, _fn in _MOE_TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"moe.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.lora.moe.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__: list[str] = []
