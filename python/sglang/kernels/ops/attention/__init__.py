"""Attention compute kernels (Triton): decode / extend / prefill / metadata.

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.attention.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
KV-cache index/write kernels went to the ``kvcache`` group instead.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# (module, public_fn) migrated from layers/attention/triton_ops + model_executor.
_TRITON_KERNELS = [
    ("decode_attention", "decode_attention_fwd"),
    ("extend_attention", "extend_attention_fwd"),
    ("extend_attention", "build_unified_kv_indices"),
    ("prefill_attention", "context_attention_fwd"),
    ("merge_state", "merge_state_triton"),
    ("metadata", "get_num_kv_splits_triton"),
    ("metadata", "prepare_swa_spec_page_table_triton"),
    ("metadata", "normal_decode_set_metadata"),
    ("dsa_metadata", "fused_dsa_decode_metadata"),
    ("dsa_metadata", "fused_dsa_target_verify_metadata"),
    ("dsa_metadata", "fused_dsa_draft_extend_metadata"),
    ("rocm_mla_decode_rope", "decode_attention_fwd_grouped_rope"),
    ("verify_splitkv", "verify_splitkv_fwd"),
    ("pad", "pad_sequence_with_mask"),
    ("pad", "pad_draft_extend_query"),
    ("pad", "unpad_draft_extend_output"),
    ("pad", "seqlens_expand_triton"),
    ("position", "compute_position_triton"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__ = []


# Vendored linear-attention (flash-linear-attention port) kernels relocated
# in Phase 2.5 (RFC #29630); representative entry points for inventory.
for _mod, _fn in [
    ("fla.chunk", "chunk_gated_delta_rule"),
    ("fla.fused_recurrent", "fused_recurrent_gated_delta_rule"),
    ("fla.kda", "fused_recurrent_kda_fwd"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# RoPE / QK-norm fusion kernels migrated from srt/layers top-level strays
# (RFC #29630, Phase 2.5); registered for inventory.
for _mod, _fn in [
    ("deepseek_v4_rope", "precompute_freqs_cis"),
    ("fused_qk_norm_rope_store", "fused_qk_norm_rope_swa_store"),
    ("fused_qk_rmsnorm_rope_gate", "fused_qk_gemma_rmsnorm_rope_gate"),
    ("fused_qk_norm", "fused_qk_norm"),
    ("rotary_triton", "triton_mrope_fused"),
    ("rotary_triton", "triton_ernie45_rope_fused_inplace"),
    ("mrope", "apply_interleaved_rope_triton"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn
