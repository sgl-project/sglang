from __future__ import annotations

import logging
from typing import Any, Optional

import msgspec

from sglang.srt.true_on_policy.schema import (
    TRUE_ON_POLICY_V1_SCHEMA,
    TrueOnPolicyContractName,
    TrueOnPolicyContractSchema,
)

logger = logging.getLogger(__name__)

TRUE_ON_POLICY_V1 = TRUE_ON_POLICY_V1_SCHEMA.name


class SGLangTrueOnPolicyRuntimePolicy(msgspec.Struct, frozen=True, kw_only=True):
    """SGLang-local behavior implied by a true-on-policy parity contract."""

    contract_name: Optional[str]
    enabled: bool
    force_bfloat16_dense_tensor_math: bool
    force_bfloat16_lm_head: bool
    disable_reduce_scatter: bool
    disable_mlp_allreduce_fusion: bool
    disable_flashinfer_allreduce_fusion: bool
    matched_tp1_row_linear: bool
    disable_fused_qk_norm_mrope: bool


DEFAULT_RUNTIME_POLICY = SGLangTrueOnPolicyRuntimePolicy(
    contract_name=None,
    enabled=False,
    force_bfloat16_dense_tensor_math=False,
    force_bfloat16_lm_head=False,
    disable_reduce_scatter=False,
    disable_mlp_allreduce_fusion=False,
    disable_flashinfer_allreduce_fusion=False,
    matched_tp1_row_linear=False,
    disable_fused_qk_norm_mrope=False,
)


class SGLangTrueOnPolicyContract(msgspec.Struct, frozen=True):
    """SGLang-local adapter from a shared contract schema to runtime policy."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        return self.schema.name

    def policy(self) -> SGLangTrueOnPolicyRuntimePolicy:
        # Declared, never derived from topology: every projection runs full-K at TP1.
        return SGLangTrueOnPolicyRuntimePolicy(
            contract_name=self.name,
            enabled=True,
            force_bfloat16_dense_tensor_math=True,
            force_bfloat16_lm_head=True,
            disable_reduce_scatter=True,
            # The fused all-reduce + residual add + RMSNorm kernels take only some forward modes and
            # batch sizes, so a row's norm would depend on its batch.
            disable_mlp_allreduce_fusion=True,
            disable_flashinfer_allreduce_fusion=True,
            matched_tp1_row_linear=True,
            # The trainer runs the Q/K norm and RoPE as separate kernels (true_on_policy.qk_norm, the rope op).
            disable_fused_qk_norm_mrope=True,
        )


TRUE_ON_POLICY_CONTRACT = SGLangTrueOnPolicyContract(
    schema=TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    TRUE_ON_POLICY_CONTRACT.name: TRUE_ON_POLICY_CONTRACT,
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


def data_parallel_layout(server_args: Any) -> tuple[int, int]:
    """(attention-DP width, replica count).

    Argument resolution turns the deprecated ``--dp-size N --enable-dp-attention``
    into ``attn_dp_size=N, dp_size=1``; arguments read before that still carry
    the deprecated pair.
    """
    dp = server_args.dp_size or 1
    attn_dp = server_args.attn_dp_size or 1
    if attn_dp > 1:
        return attn_dp, dp
    if server_args.enable_dp_attention:
        return dp, 1
    return 1, dp


def validate_true_on_policy_contract(server_args: Any) -> None:
    contract_name = server_args.true_on_policy_contract
    if contract_name is None:
        return
    get_true_on_policy_contract(contract_name)
    if not _no_split_reductions(server_args):
        raise ValueError(
            "[top] every projection must see its full K: engine TP>1 only as attention DP with dense, "
            "expert (EP = engine TP) and LM-head TP1, and no serving context parallelism"
        )
    refused = _unqualified_features(server_args)
    if refused:
        raise ValueError(
            f"[top] no TOP program covers {', '.join(refused)}; remove it under the contract"
        )


def _unqualified_features(server_args: Any) -> list[str]:
    """Serving features that change the program and that TOP does not verify."""
    lora = server_args.enable_lora or (
        bool(server_args.lora_paths) and server_args.enable_lora is None
    )
    features = (
        ("LoRA (--enable-lora / --lora-paths)", lora),
        # Inductor may lower GEMMs past the batch-invariant aten overrides.
        ("--enable-torch-compile", server_args.enable_torch_compile),
        # Batches decode rows into the prefill's varlen attention call.
        ("--enable-mixed-chunk", server_args.enable_mixed_chunk),
    )
    return [name for name, enabled in features if enabled]


def _no_split_reductions(server_args: Any) -> bool:
    """No layer splits a contraction across ranks, so full-K kernels match the TP1 trainer."""
    tp = server_args.tp_size
    attention_tp1 = tp == 1 or data_parallel_layout(server_args)[0] == tp
    dense_tp1 = tp == 1 or server_args.moe_dense_tp_size == 1
    expert_tp1 = server_args.ep_size == tp
    lm_head_tp1 = tp == 1 or server_args.enable_dp_lm_head
    no_serving_cp = server_args.attn_cp_size == 1 and server_args.dcp_size == 1
    return attention_tp1 and dense_tp1 and expert_tp1 and lm_head_tp1 and no_serving_cp


# Families whose TOP program runs a model-specific attention backend instead of the contract's.
MODEL_ATTENTION_BACKENDS = {
    "glm_moe_dsa": "dsa",
    "glm5_next": "dsa",
    "deepseek_v4": "dsv4",
    "deepseek_v41": "dsv4",
}
# Attention backends whose page-aligned radix hits equal a cold prefill byte for byte.
RADIX_EXACT_BACKENDS = frozenset({"dsv4"})
# Families whose TOP attention resumes a radix hit as a chunked-prefill segment from the same offset, on a backend
# other families share: GLM-5.3-Flash's DSA (group-aligned extends) and KDA (64-token track grid).
RADIX_EXACT_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})


def _model_type(server_args: Any) -> Optional[str]:
    """The checkpoint's model type, guarded as handle_deterministic_inference reads it: an instance connector has no
    config to read, and a config that cannot load yet leaves the contract's own backend."""
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.connector import ConnectorType
    from sglang.srt.utils.common import parse_connector_type

    if parse_connector_type(server_args.model_path) == ConnectorType.INSTANCE:
        return None
    try:
        return model_config_of(server_args).hf_config.model_type
    except Exception as e:
        logger.warning(
            f"[top] cannot read the model type for the attention backend: {e}"
        )
        return None


def get_true_on_policy_attention_overrides(server_args: Any) -> dict:
    """Select the contract kernel before hardware defaults or split backends."""
    contract_name = server_args.true_on_policy_contract
    if contract_name is None:
        return {}
    backend = get_true_on_policy_contract(contract_name).schema.attention_backend
    backend = MODEL_ATTENTION_BACKENDS.get(_model_type(server_args), backend)
    fields = (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    )
    from sglang.srt.arg_groups.arg_utils import is_record
    from sglang.srt.arg_groups.model_override_base import record_of

    # The contract wins over model/hardware defaults. Reject only conflicting
    # user inputs, not a backend selected earlier by the resolution pipeline.
    requested_args = server_args if is_record(server_args) else record_of(server_args)
    for field in fields:
        requested = getattr(requested_args, field)
        if requested not in (None, backend):
            raise ValueError(f"TOP requires {field}={backend!r}; got {requested!r}.")
    return dict.fromkeys(fields, backend)


def resolve_true_on_policy_runtime_policy(
    server_args: Any,
) -> SGLangTrueOnPolicyRuntimePolicy:
    contract_name = server_args.true_on_policy_contract
    if contract_name is None:
        return DEFAULT_RUNTIME_POLICY
    return get_true_on_policy_contract(contract_name).policy()


def top_keeps_radix_cache(server_args: Any, attention_backend: str) -> bool:
    """TOP DeepSeek-V4, V4.1 and GLM-5.3-Flash reuse radix prefixes: a page-aligned hit equals a cold prefill byte
    for byte.

    The c4 compressor's prefill plan writes overlap state for every SWA page's last window, so a hit
    restores what the cold path computes; the request-ring layout, which skips those writes, is refused
    in ``validate_dsv4``. V4.1's pages hold whole ratio-2 pairs, so a hit leaves no pending pair; the
    prefix sources a hit could come from other than this engine's device tree are refused in
    ``validate_dsv41_radix``.
    """
    return server_args.true_on_policy_contract is not None and (
        attention_backend in RADIX_EXACT_BACKENDS
        or _model_type(server_args) in RADIX_EXACT_MODEL_TYPES
    )
