from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


QWEN3_DENSE_TRUE_ON_POLICY_V1 = "qwen3_dense_true_on_policy_v1"


@dataclass(frozen=True)
class SGLangTrueOnPolicyRuntimePolicy:
    """SGLang-local behavior implied by a true-on-policy parity contract."""

    contract_name: Optional[str]
    enabled: bool
    force_bfloat16_dense_tensor_math: bool
    force_bfloat16_lm_head: bool
    disable_reduce_scatter: bool
    disable_mlp_allreduce_fusion: bool
    disable_flashinfer_allreduce_fusion: bool
    tp_invariant_row_linear: bool
    deterministic_tree_all_reduce: bool


DEFAULT_RUNTIME_POLICY = SGLangTrueOnPolicyRuntimePolicy(
    contract_name=None,
    enabled=False,
    force_bfloat16_dense_tensor_math=False,
    force_bfloat16_lm_head=False,
    disable_reduce_scatter=False,
    disable_mlp_allreduce_fusion=False,
    disable_flashinfer_allreduce_fusion=False,
    tp_invariant_row_linear=False,
    deterministic_tree_all_reduce=False,
)


def _contract_name_for(server_args: Any) -> Optional[str]:
    target = getattr(server_args, "rl_on_policy_target", None)
    contract_name = getattr(server_args, "true_on_policy_contract", None)
    if contract_name is not None:
        return contract_name
    if target is not None:
        return QWEN3_DENSE_TRUE_ON_POLICY_V1
    return None


def validate_true_on_policy_contract(server_args: Any) -> None:
    contract_name = getattr(server_args, "true_on_policy_contract", None)
    target = getattr(server_args, "rl_on_policy_target", None)
    if contract_name is None:
        return
    if target is None:
        raise ValueError(
            "--true-on-policy-contract requires --rl-on-policy-target so the "
            "runtime policy cannot silently become a no-op."
        )
    if contract_name != QWEN3_DENSE_TRUE_ON_POLICY_V1:
        raise ValueError(f"Unsupported SGLang true-on-policy contract: {contract_name!r}")


def resolve_true_on_policy_runtime_policy(
    server_args: Any,
) -> SGLangTrueOnPolicyRuntimePolicy:
    target = getattr(server_args, "rl_on_policy_target", None)
    contract_name = _contract_name_for(server_args)
    if contract_name is None:
        return DEFAULT_RUNTIME_POLICY

    validate_true_on_policy_contract(server_args)

    uses_tp_invariant_rollout = target == "fsdp_tp"
    return SGLangTrueOnPolicyRuntimePolicy(
        contract_name=contract_name,
        enabled=True,
        force_bfloat16_dense_tensor_math=True,
        force_bfloat16_lm_head=True,
        disable_reduce_scatter=True,
        disable_mlp_allreduce_fusion=True,
        disable_flashinfer_allreduce_fusion=uses_tp_invariant_rollout,
        tp_invariant_row_linear=uses_tp_invariant_rollout,
        deterministic_tree_all_reduce=uses_tp_invariant_rollout,
    )
