"""LoRA kernels: shared routing/GEMMs, legacy dense adapters and MoE stages.

Runtime plans/providers select implementations; this package records lazy metadata.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, KernelSpec

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


# Host launch APIs; importing this inventory does not import their implementations.
_ENGINE_TRITON_KERNELS = [
    ("common.routing", "build_route", "build_route"),
    ("common.lora_a", "grouped_lora_a", "grouped_lora_a"),
    ("common.lora_a", "per_row_lora_a", "per_row_lora_a"),
    ("common.lora_b", "grouped_lora_b", "grouped_lora_b"),
    ("common.lora_b", "per_row_lora_b", "per_row_lora_b"),
    (
        "dense.embedding_lora_a",
        "embedding_lora_a_tokens_fwd",
        "embedding_lora_a_tokens_fwd",
    ),
    ("moe.activation_delta", "act_delta_masked", "act_delta_masked"),
    ("moe.activation_delta", "act_delta_contiguous", "act_delta_contiguous"),
    ("moe.align_rows", "pair_to_row_map", "pair_to_row_map"),
    ("moe.align_rows", "moe_align_single_token", "moe_align_single_token"),
    (
        "moe.dispatch_contiguous",
        "dispatch_layout_contiguous",
        "dispatch_layout_contiguous",
    ),
    (
        "moe.dispatch_contiguous",
        "dispatch_fill_rows_contiguous_bf16",
        "dispatch_fill_rows_contiguous_bf16",
    ),
    (
        "moe.dispatch_contiguous",
        "dispatch_fill_rows_contiguous_fp8",
        "dispatch_fill_rows_contiguous_fp8",
    ),
    ("moe.dispatch_masked", "dispatch_fill_masked_bf16", "dispatch_fill_masked_bf16"),
    ("moe.dispatch_masked", "dispatch_fill_masked_fp8", "dispatch_fill_masked_fp8"),
    ("moe.dispatch_masked_small", "small_masked_prepare", "small_masked_prepare"),
    (
        "moe.finalize",
        "invoke_shared_token_delta_reduce",
        "invoke_shared_token_delta_reduce",
    ),
    (
        "moe.finalize",
        "invoke_shared_token_delta_tail",
        "invoke_shared_token_delta_tail",
    ),
    ("moe.finalize", "invoke_shared_one_pass", "invoke_shared_one_pass"),
    ("moe.finalize", "invoke_small_finalize", "invoke_small_finalize"),
    ("moe.fused_act", "fused_b_act_masked", "fused_b_act_masked"),
    ("moe.fused_act", "fused_b_act_contiguous", "fused_b_act_contiguous"),
    ("moe.lora_b", "grouped_lora_b", "moe_grouped_lora_b"),
    ("moe.lora_b", "invoke_down_b_into_base", "invoke_down_b_into_base"),
    (
        "moe.cutedsl.schedule_builder",
        "build_dual_stage_schedules_masked",
        "build_dual_stage_schedules_masked",
    ),
    (
        "moe.cutedsl.schedule_builder",
        "build_dual_stage_schedules_contiguous",
        "build_dual_stage_schedules_contiguous",
    ),
]
for _mod, _fn, _op in _ENGINE_TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"lora.{_op}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.lora.{_mod}:{_fn}",
            capabilities=frozenset({CapabilityRequirement.CUDA}),
        )
    )
del _mod, _fn, _op

# CuTeDSL grouped GEMMs: each compiles once and returns the launchable kernel.
# Their kernel classes exist for SM90 (WGMMA) and SM100 (tcgen05) only.
_ENGINE_CUTE_DSL_KERNELS = (
    "prepare_masked_bf16",
    "prepare_contiguous_bf16",
)
for _fn in _ENGINE_CUTE_DSL_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"lora.{_fn}",
            backend=KernelBackend.CUTE_DSL,
            target=f"sglang.kernels.ops.lora.moe.cutedsl.api:{_fn}",
            capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
        )
    )
del _fn

__all__: list[str] = []
