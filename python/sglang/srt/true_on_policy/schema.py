from __future__ import annotations

from typing import Literal

import msgspec

TrueOnPolicyContractName = Literal["true_on_policy_v1"]


class TrueOnPolicyContractSchema(msgspec.Struct, frozen=True, kw_only=True):
    """Cross-repo identity of a true-on-policy program version, including its attention backend."""

    name: TrueOnPolicyContractName
    disable_megatron_sequence_parallel: bool
    attention_backend: str


TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="true_on_policy_v1",
    disable_megatron_sequence_parallel=True,
    attention_backend="fa4",
)
