from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.true_on_policy.schema import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
    TrueOnPolicyContractName,
    TrueOnPolicyContractSchema,
)

QWEN3_DENSE_TRUE_ON_POLICY_V1 = QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA.name


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
    disable_fused_qk_norm_mrope: bool


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
    disable_fused_qk_norm_mrope=False,
)


@dataclass(frozen=True)
class SGLangTrueOnPolicyContract:
    """SGLang-local adapter from a shared contract schema to runtime policy."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        return self.schema.name

    def policy_for(self, server_args: Any) -> SGLangTrueOnPolicyRuntimePolicy:
        uses_tp_invariant_rollout = getattr(server_args, "tp_size", 1) > 1
        return SGLangTrueOnPolicyRuntimePolicy(
            contract_name=self.name,
            enabled=True,
            force_bfloat16_dense_tensor_math=True,
            force_bfloat16_lm_head=True,
            disable_reduce_scatter=True,
            disable_mlp_allreduce_fusion=True,
            disable_flashinfer_allreduce_fusion=uses_tp_invariant_rollout,
            tp_invariant_row_linear=uses_tp_invariant_rollout,
            deterministic_tree_all_reduce=uses_tp_invariant_rollout,
            disable_fused_qk_norm_mrope=True,
        )


QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT = SGLangTrueOnPolicyContract(
    schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT.name: QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT,
}


def get_true_on_policy_contract(contract_name: str) -> SGLangTrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[contract_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(
            f"Unsupported SGLang true-on-policy contract {contract_name!r}. "
            f"Supported contracts: {supported}"
        ) from exc


def _contract_name_for(server_args: Any) -> Optional[str]:
    return getattr(server_args, "true_on_policy_contract", None)


def validate_true_on_policy_contract(server_args: Any) -> None:
    contract_name = getattr(server_args, "true_on_policy_contract", None)
    if contract_name is None:
        return
    get_true_on_policy_contract(contract_name)


def resolve_true_on_policy_runtime_policy(
    server_args: Any,
) -> SGLangTrueOnPolicyRuntimePolicy:
    contract_name = _contract_name_for(server_args)
    if contract_name is None:
        return DEFAULT_RUNTIME_POLICY

    validate_true_on_policy_contract(server_args)
    return get_true_on_policy_contract(contract_name).policy_for(server_args)
