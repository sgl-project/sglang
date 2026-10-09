"""Versioned export contract for the KV-input DSpark architecture."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import msgspec
from sglang.srt.training_capture.kv_codec import StandardRopeConfig
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    KVSpec,
    Nonnegative,
    Positive,
    StrictStruct,
    TeacherIdentity,
    Text,
    canonical_bytes,
    digest_bytes,
)


class KVEncoderConfig(StrictStruct):
    hidden_size: Positive
    rms_norm_eps: Annotated[float, msgspec.Meta(gt=0)]
    bias: bool = False
    norm_semantics: Literal["fp32_variance_and_weight_then_cast"] = (
        "fp32_variance_and_weight_then_cast"
    )


class KVSequenceContract(StrictStruct):
    prediction_count: Annotated[int, msgspec.Meta(ge=1, le=64)]
    mask_token_id: Nonnegative
    input_length: Positive
    context_boundary: Literal["strictly_before_anchor"] = "strictly_before_anchor"
    backbone_input: Literal["anchor_then_masks"] = "anchor_then_masks"
    label_shift: Literal[1] = 1
    markov_previous: Literal["anchor_then_previous_labels"] = (
        "anchor_then_previous_labels"
    )
    position_semantics: Literal["actual_target_positions"] = "actual_target_positions"


class KVTrainingContract(StrictStruct):
    lambda_tv: Annotated[float, msgspec.Meta(ge=0)]
    objective: Literal["ce_tv128_full_vocabulary_lse_t1_v1"] = (
        "ce_tv128_full_vocabulary_lse_t1_v1"
    )
    tv_tail_policy: Literal["omit_tail_without_renormalization"] = (
        "omit_tail_without_renormalization"
    )
    confidence_policy: Literal["disabled"] = "disabled"
    shared_modules: Literal["frozen_target_reference"] = "frozen_target_reference"
    shared_head_transform_placement: Literal["before_markov"] = "before_markov"
    base_logits_dtype: Literal["float32"] = "float32"


class SharedHeadTransform(StrictStruct):
    logit_scale: float | None = None
    final_logit_softcapping: Annotated[float, msgspec.Meta(ge=0)] | None = None

    @classmethod
    def decode(cls, value: str):
        transform = (
            cls() if value == "identity" else msgspec.json.decode(value, type=cls)
        )
        if any(
            value is not None and not math.isfinite(value)
            for value in (transform.logit_scale, transform.final_logit_softcapping)
        ):
            raise ContractError("shared-head transform must be finite")
        return transform

    def apply(self, logits):
        logits = logits.float()
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        if self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = cap * (logits / cap).tanh()
        return logits


class KVCompatibility(StrictStruct):
    sglang_revision: Text
    specforge_revision: Text
    serving_contract: Literal["sglang_dspark_target_kv_v1"] = (
        "sglang_dspark_target_kv_v1"
    )


class KVValidation(StrictStruct):
    golden_fixture_sha256: Digest
    parity_rtol: Annotated[float, msgspec.Meta(ge=0)]
    parity_atol: Annotated[float, msgspec.Meta(ge=0)]
    acceptance_report_sha256: Digest | None = None


class TargetKVDraftContract(StrictStruct):
    teacher: TeacherIdentity
    kv: KVSpec
    encoder: KVEncoderConfig
    sequence: KVSequenceContract
    training: KVTrainingContract
    compatibility: KVCompatibility
    validation: KVValidation
    architecture_revision: Literal["dspark_target_kv_v1"] = "dspark_target_kv_v1"
    input_mode: Literal["target_kv"] = "target_kv"
    schema_version: Literal[1] = 1
    feature_order: Literal["layer_then_k_v_head_dim"] = "layer_then_k_v_head_dim"
    feature_k_stage: Literal["pre_rope", "post_rope"] = "pre_rope"

    @classmethod
    def decode(cls, value):
        data = canonical_bytes(value)
        if len(data) > 1 << 20:
            raise ContractError("target KV draft contract is too large")
        contract = msgspec.json.decode(data, type=cls)
        StandardRopeConfig.from_kv(contract.kv)
        SharedHeadTransform.decode(contract.teacher.output_transform)
        if contract.encoder.bias:
            raise ContractError("KV encoder v1 requires a bias-free projection")
        if contract.sequence.input_length != contract.sequence.prediction_count:
            raise ContractError("DSpark input length must equal prediction_count")
        if contract.sequence.mask_token_id >= contract.teacher.vocab_size:
            raise ContractError("draft mask token is outside the bound vocabulary")
        if contract.teacher.adapter_revision is not None:
            raise ContractError("target KV draft v1 does not support target adapters")
        if (
            contract.feature_k_stage == "post_rope"
            and contract.kv.source_k_stage != "post_rope"
        ):
            raise ContractError("post-RoPE features require a post-RoPE source")
        if not all(
            math.isfinite(value)
            for value in (
                contract.encoder.rms_norm_eps,
                contract.training.lambda_tv,
                contract.validation.parity_rtol,
                contract.validation.parity_atol,
            )
        ):
            raise ContractError("draft numerical contract values must be finite")
        return contract

    @property
    def feature_size(self):
        return sum(
            layer.num_kv_heads * (layer.key_head_dim + layer.value_head_dim)
            for layer in self.kv.layers
        )

    @property
    def fingerprint(self):
        return digest_bytes(canonical_bytes(self))


def read_target_kv_draft_contract(config) -> TargetKVDraftContract | None:
    raw = config if isinstance(config, dict) else config.to_dict()
    mode = raw.get("input_mode", "target_hidden")
    if mode == "target_hidden":
        if "target_kv_contract" in raw:
            raise ContractError("hidden-input checkpoint carries a target KV contract")
        return None
    if mode != "target_kv":
        raise ContractError(f"unknown DSpark input_mode: {mode!r}")
    if raw.get("architectures") != ["DSparkTargetKVDraftModel"]:
        raise ContractError("target KV input requires DSparkTargetKVDraftModel")
    if "target_kv_contract" not in raw:
        raise ContractError("target KV checkpoint is missing its contract")
    contract = TargetKVDraftContract.decode(raw["target_kv_contract"])
    if raw.get("hidden_size") != contract.encoder.hidden_size:
        raise ContractError("encoder output and draft hidden size disagree")
    if raw.get("enable_confidence_head") is not False:
        raise ContractError(
            "target KV v1 requires an explicitly disabled confidence head"
        )
    return contract


def validate_target_kv_draft_contract(
    *,
    target: TeacherIdentity,
    draft: TargetKVDraftContract,
    pool_codec: KVSpec,
    target_hidden_size: int,
    prediction_count: int,
    mask_token_id: int,
) -> None:
    if target != draft.teacher:
        raise ContractError("loaded target identity differs from the draft checkpoint")
    StandardRopeConfig.from_kv(pool_codec)
    for field in (
        "codec",
        "dtype",
        "selected_layer_ids",
        "layers",
        "source_k_stage",
        "source_k_norm",
        "rope_config",
        "rope_config_sha256",
        "layout",
        "layer_numbering",
        "validity_policy",
    ):
        # These are the declared wire-schema fields, not optional runtime fields.
        if getattr(pool_codec, field) != getattr(draft.kv, field):
            raise ContractError(
                f"loaded target KV differs from draft contract: {field}"
            )
    if target_hidden_size != draft.encoder.hidden_size:
        raise ContractError(
            "draft hidden size cannot use the bound shared embedding/head"
        )
    if prediction_count != draft.sequence.prediction_count:
        raise ContractError(
            "runtime prediction_count differs from trained DSpark window"
        )
    if mask_token_id != draft.sequence.mask_token_id:
        raise ContractError("runtime mask token differs from trained DSpark window")
