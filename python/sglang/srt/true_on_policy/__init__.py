"""True-on-policy runtime contract helpers."""

from .config import (
    ROW_LINEAR_INV_BLOCK_K,
    get_on_policy_rms_norm_kwargs,
    get_rl_on_policy_target,
    is_tp_invariant_target,
    is_true_on_policy_enabled,
    patch_prefill_only_deterministic_inference_for_cuda_graph,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
    should_use_tp_invariant_row_linear,
    should_use_tp_invariant_tree_all_reduce,
)

__all__ = [
    "ROW_LINEAR_INV_BLOCK_K",
    "get_on_policy_rms_norm_kwargs",
    "get_rl_on_policy_target",
    "is_tp_invariant_target",
    "is_true_on_policy_enabled",
    "patch_prefill_only_deterministic_inference_for_cuda_graph",
    "should_disable_flashinfer_allreduce_fusion",
    "should_disable_mlp_allreduce_fusion_for_on_policy",
    "should_disable_reduce_scatter_for_on_policy",
    "should_force_bfloat16_dense_tensor_math",
    "should_force_bfloat16_lm_head",
    "should_use_tp_invariant_row_linear",
    "should_use_tp_invariant_tree_all_reduce",
]
