from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrueOnPolicyContractName = Literal[
    "qwen3_dense_true_on_policy_v1",
    "qwen3_moe_true_on_policy_v1",
]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
KernelContract = Literal["qwen3_dense_sglang_math", "qwen3_moe_sglang_math"]
LogprobContract = Literal["sglang_prefill"]


@dataclass(frozen=True)
class TrueOnPolicyContractSchema:
    """Declarative cross-repo identity for a true-on-policy parity contract."""

    name: TrueOnPolicyContractName
    model_family: ModelFamily
    required_kernel_contracts: tuple[KernelContract, ...]
    logprob_contract: LogprobContract
    sglang_attention_backend: str
    fsdp_attention_implementation: str
    disable_megatron_sequence_parallel: bool


QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)

QWEN3_MOE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_moe_true_on_policy_v1",
    model_family="qwen3_moe",
    required_kernel_contracts=("qwen3_dense_sglang_math", "qwen3_moe_sglang_math"),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)
