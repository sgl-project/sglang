"""True-on-policy runtime contract helpers."""

from .config import (
    ROW_LINEAR_INV_BLOCK_K,
    get_on_policy_rms_norm_kwargs,
    get_rl_on_policy_target,
    is_tp_invariant_target,
    is_true_on_policy_enabled,
    patch_prefill_only_deterministic_inference_for_cuda_graph,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_fused_qk_norm_mrope,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
    should_use_tp_invariant_row_linear,
    should_use_tp_invariant_tree_all_reduce,
)
from .contracts import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    SGLangTrueOnPolicyContract,
    SGLangTrueOnPolicyRuntimePolicy,
    get_true_on_policy_contract,
    resolve_true_on_policy_runtime_policy,
    validate_true_on_policy_contract,
)

__all__ = [
    "QWEN3_DENSE_TRUE_ON_POLICY_V1",
    "ROW_LINEAR_INV_BLOCK_K",
    "SGLangTrueOnPolicyContract",
    "SGLangTrueOnPolicyRuntimePolicy",
    "get_true_on_policy_contract",
    "get_on_policy_rms_norm_kwargs",
    "get_rl_on_policy_target",
    "is_tp_invariant_target",
    "is_true_on_policy_enabled",
    "patch_prefill_only_deterministic_inference_for_cuda_graph",
    "resolve_true_on_policy_runtime_policy",
    "validate_true_on_policy_contract",
    "should_disable_flashinfer_allreduce_fusion",
    "should_disable_fused_qk_norm_mrope",
    "should_disable_mlp_allreduce_fusion_for_on_policy",
    "should_disable_reduce_scatter_for_on_policy",
    "should_force_bfloat16_dense_tensor_math",
    "should_force_bfloat16_lm_head",
    "should_use_tp_invariant_row_linear",
    "should_use_tp_invariant_tree_all_reduce",
]
