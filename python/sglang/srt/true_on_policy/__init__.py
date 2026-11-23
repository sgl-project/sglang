"""True-on-policy runtime contract helpers."""

from .config import (
    get_on_policy_rms_norm_kwargs,
    get_rl_on_policy_target,
    is_true_on_policy_enabled,
    patch_prefill_only_deterministic_inference_for_cuda_graph,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_fused_qk_norm_mrope,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
)
from .contracts import (
    TRUE_ON_POLICY_V1,
    SGLangTrueOnPolicyContract,
    SGLangTrueOnPolicyRuntimePolicy,
    get_true_on_policy_contract,
    resolve_true_on_policy_runtime_policy,
    validate_true_on_policy_contract,
)

__all__ = [
    "TRUE_ON_POLICY_V1",
    "SGLangTrueOnPolicyContract",
    "SGLangTrueOnPolicyRuntimePolicy",
    "get_true_on_policy_contract",
    "get_on_policy_rms_norm_kwargs",
    "get_rl_on_policy_target",
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
]
