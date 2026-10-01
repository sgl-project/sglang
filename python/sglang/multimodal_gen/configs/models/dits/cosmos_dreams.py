# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams schema-v1 causal manifest and conditioning contracts.

A Cosmos-Dreams (Cosmos3-Interactive) checkpoint exports a ``cosmos3_nano_sim_bimanual``
block inside ``transformer/config.json``. It fixes the causal rollout
geometry (chunk size, K/V window, mRoPE constants), the distilled fixed-step
sampler, and one conditioning contract selected by ``conditioning.mode``:
``action`` (per-embodiment raw action width, domain id, affine normalizer) or
``control_video`` (Cosmos-Dreams-Transfer: edge/blur/depth/seg control clips).
Only fields that change model output are interpreted; exporter provenance
blocks are accepted and ignored.
"""

import hashlib
import json
import math
import os
import struct
from collections.abc import Sequence
from typing import Any

import msgspec

COSMOS_DREAMS_SCHEMA_VERSION = 1
ACTION_CONTRACT_SCHEMA_VERSION = 3
# Schema 5: every embodiment shares the 59-slot unified_v1 row, normalized by the
# client (imaginaire4 "normalized_unified_v1" sidecars); domains may vary per row.
UNIFIED_ACTION_CONTRACT_SCHEMA_VERSION = 5
UNIFIED_ACTION_SPACE = "normalized_unified_v1"
UNIFIED_ACTION_DIM = 59
UNIFIED_LAYOUT_ID = "unified_v1"
UNIFIED_POSE_CONVENTION = "backward_chunk_anchored_16f"
UNIFIED_NORMALIZER_METHOD = "global_asinh_unified_v1"
UNIFIED_DOMAIN_ROUTING = "scalar_or_per_action_row"
UNIFIED_VALIDITY = "masked_to_zero_after_normalization_by_source"
PADDING_STAGE_AFTER_NORMALIZATION_AND_MASK = "after_normalization_and_validity_mask"
CAMERA_EMBODIMENT = "camera_pose"
# Slots of the unified row a camera-only row may use (ego pose); the rest must be zero.
UNIFIED_CAMERA_SLOTS = 9
HISTORY_MODE_FULL = "full"
HISTORY_MODE_SLIDING = "sliding"
HISTORY_MODES = (HISTORY_MODE_FULL, HISTORY_MODE_SLIDING)
# Transfer only: slide when the export's training config recorded a KV window.
HISTORY_MODE_AUTO = "auto"
ACTION_CONDITIONING_MODE = "action"
CONTROL_VIDEO_CONDITIONING_MODE = "control_video"
TRANSFER_HINTS = ("edge", "blur", "depth", "seg")
TRANSFER_CONTROL_ATTENTION_MODE = "causal_control_with_rgb_history"
TRANSFER_SYSTEM_PROMPT_ID = "cosmos3_transfer_v1"
# Training truncated the chat-formatted caption ids at this many tokens before the
# packer appended EOS and vision-start (imaginaire4 text_tokenizer._MAX_NUM_TOKENS).
TEXT_TOKENS_TRAINING_MAX = 4096
ATTENTION_MODE_THREE_WAY = "three_way"
SAMPLE_TYPE_SDE = "sde"
AFFINE_TRANSFORM_TYPE = "affine"
PADDING_STAGE_AFTER_NORMALIZATION = "after_normalization"
POSE_SCALE_METHOD = "pose_scale"
_SHA256_HEX_LEN = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


def float32_value(value: float) -> float:
    """Round a scalar to float32, the precision the action normalizer runs at."""
    result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    if not math.isfinite(result):
        raise ValueError(
            f"Cosmos-Dreams action-contract value must be finite, got {value!r}."
        )
    return 0.0 if result == 0.0 else result


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return float32_value(value)
    return value


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Hash semantic JSON with the exporter's float32 canonicalization."""
    encoded = json.dumps(
        _canonicalize(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _drop_none(value: Any) -> Any:
    """Mirror the exporter's ``exclude_none`` dump of the layout block."""
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _is_sha256_hex(value: str) -> bool:
    return len(value) == _SHA256_HEX_LEN and set(value) <= _HEX_DIGITS


class AffineTransform(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    type: str
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    forward_clamp: bool


class ActionNormalizerContract(msgspec.Struct, frozen=True):
    """Normalizer applied to one embodiment's raw action rows.

    ``source`` / ``training_config`` provenance blocks are not declared so an
    extended exporter keeps loading; they never enter a hashed payload.
    """

    schema_version: int
    method: str
    transform: AffineTransform
    derivation: dict[str, Any]
    transform_sha256: str

    def behavioral_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "transform": msgspec.to_builtins(self.transform),
            "derivation": self.derivation,
        }


class EmbodimentContract(msgspec.Struct, frozen=True):
    domain_id: int
    raw_action_dim: int
    layout: dict[str, Any]
    normalizer: ActionNormalizerContract


class ActionPadding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    stage: str
    value: float


class CosmosDreamsActionContract(
    msgspec.Struct, frozen=True, tag_field="mode", tag=ACTION_CONDITIONING_MODE
):
    """Per-embodiment action contract of an action-conditioned checkpoint."""

    schema_version: int
    action_tokens_per_frame: int
    model_action_dim: int
    num_embodiment_domains: int
    default_embodiment: str
    embodiments: dict[str, EmbodimentContract]
    padding: ActionPadding
    contract_sha256: str

    def behavioral_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_tokens_per_frame": self.action_tokens_per_frame,
            "model_action_dim": self.model_action_dim,
            "num_embodiment_domains": self.num_embodiment_domains,
            "default_embodiment": self.default_embodiment,
            "embodiments": {
                name: {
                    "domain_id": contract.domain_id,
                    "raw_action_dim": contract.raw_action_dim,
                    "layout": _drop_none(contract.layout),
                    "normalizer_sha256": contract.normalizer.transform_sha256,
                }
                for name, contract in sorted(self.embodiments.items())
            },
            "padding": msgspec.to_builtins(self.padding),
        }

    def resolve_embodiment(self, name: str | None, domain_id: int | None) -> str:
        """Resolve the embodiment named by ``name`` and/or ``domain_id``.

        Without a name, ``domain_id`` selects the embodiment when it is
        unambiguous; the default embodiment wins when it uses that domain.
        """
        return _resolve_embodiment(
            embodiments=self.embodiment_to_domain,
            default_embodiment=self.default_embodiment,
            name=name,
            domain_id=domain_id,
        )

    @property
    def embodiment_to_domain(self) -> dict[str, int]:
        return {name: contract.domain_id for name, contract in self.embodiments.items()}

    @property
    def mode(self) -> str:
        return ACTION_CONDITIONING_MODE


def _resolve_embodiment(
    *,
    embodiments: dict[str, int],
    default_embodiment: str,
    name: str | None,
    domain_id: int | None,
) -> str:
    """Shared name/domain resolution of both action contract schemas."""
    if name is None or not str(name).strip():
        candidates = [
            embodiment
            for embodiment, domain in embodiments.items()
            if domain_id is not None and domain == int(domain_id)
        ]
        if domain_id is None or default_embodiment in candidates:
            embodiment = default_embodiment
        elif len(candidates) == 1:
            embodiment = candidates[0]
        elif not candidates:
            raise ValueError(
                f"No Cosmos-Dreams embodiment uses domain_id={domain_id}; "
                f"known domains: {embodiments}."
            )
        else:
            raise ValueError(
                f"Cosmos-Dreams domain_id={domain_id} is ambiguous across "
                f"{sorted(candidates)}; supply domain_name."
            )
    else:
        embodiment = str(name).strip().lower()
    if embodiment not in embodiments:
        raise ValueError(
            f"Unknown Cosmos-Dreams embodiment {name!r}; expected one of "
            f"{sorted(embodiments)}."
        )
    expected_domain = embodiments[embodiment]
    if domain_id is not None and int(domain_id) != expected_domain:
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} requires "
            f"domain_id={expected_domain}, got {domain_id}."
        )
    return embodiment


class UnifiedEmbodimentContract(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    domain_id: int
    input_action_dim: int


class UnifiedInputContract(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    action_space: str
    action_dim: int
    domain_routing: str
    validity: str
    runtime_normalization: bool
    model_mode: str


class UnifiedLayoutField(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    offset: int
    size: int


class UnifiedLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    id: str
    pose_convention: str
    rotation_representation: str
    fields: tuple[UnifiedLayoutField, ...]


class UnifiedNormalizerReference(msgspec.Struct, frozen=True):
    """Provenance of the client-side normalizer; ``source`` names the stats file
    under ``cosmos3_nano_sim_bimanual_action_sources`` and its sha256."""

    method: str
    runtime_application: bool
    source: dict[str, Any]


class CosmosDreamsUnifiedActionContract(
    msgspec.Struct, frozen=True, tag_field="mode", tag=ACTION_CONDITIONING_MODE
):
    """Schema-5 action contract: one normalized 59-D row layout for every embodiment."""

    schema_version: int
    action_tokens_per_frame: int
    model_action_dim: int
    num_embodiment_domains: int
    default_embodiment: str
    embodiments: dict[str, UnifiedEmbodimentContract]
    padding: ActionPadding
    input_contract: UnifiedInputContract
    layout: UnifiedLayout
    normalizer: UnifiedNormalizerReference
    training_config_excerpt: dict[str, Any]
    contract_sha256: str

    def resolve_embodiment(self, name: str | None, domain_id: int | None) -> str:
        return _resolve_embodiment(
            embodiments=self.embodiment_to_domain,
            default_embodiment=self.default_embodiment,
            name=name,
            domain_id=domain_id,
        )

    @property
    def embodiment_to_domain(self) -> dict[str, int]:
        return {name: contract.domain_id for name, contract in self.embodiments.items()}

    @property
    def camera_domain_id(self) -> int | None:
        """Domain whose rows may only use the ego-pose slots, if declared."""
        contract = self.embodiments.get(CAMERA_EMBODIMENT)
        return None if contract is None else contract.domain_id

    @property
    def training_prompt_as_json(self) -> bool:
        """Whether the training loader wrapped captions in the JSON action prompt."""
        loader = self.training_config_excerpt.get("loader")
        if isinstance(loader, dict) and "format_prompt_as_json" in loader:
            return bool(loader["format_prompt_as_json"])
        return True

    @property
    def mode(self) -> str:
        return ACTION_CONDITIONING_MODE


class CosmosDreamsControlVideoContract(
    msgspec.Struct, frozen=True, tag_field="mode", tag=CONTROL_VIDEO_CONDITIONING_MODE
):
    """Control-video (transfer) conditioning of a Cosmos-Dreams-Transfer checkpoint.

    The control clip is encoded by the video VAE and packed as clean vision
    tokens that share the target frames' temporal positions; ``no_eviction``
    keeps every committed control/RGB pair for the whole rollout.
    """

    hints: tuple[str, ...]
    transfer_control_attention_mode: str
    share_vision_temporal_positions: bool
    system_prompt_id: str
    emphasize_control_in_prompt: bool
    no_eviction: bool

    @property
    def mode(self) -> str:
        return CONTROL_VIDEO_CONDITIONING_MODE


CosmosDreamsActionConditioning = (
    CosmosDreamsActionContract | CosmosDreamsUnifiedActionContract
)
CosmosDreamsConditioning = (
    CosmosDreamsActionContract
    | CosmosDreamsUnifiedActionContract
    | CosmosDreamsControlVideoContract
)


class FixedStepSamplerConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    sample_type: str
    t_list: tuple[float, ...]
    num_train_timesteps: int


class CosmosDreamsManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The complete schema-v1 artifact; every field is required."""

    schema_version: int
    checkpoint_id: str
    checkpoint_iteration: int
    checkpoint_hash: str
    chunk_size: int
    # None: the checkpoint trained with an unbounded K/V history.
    window_frames: int | None
    sink_frames: int
    text_cache_max_len: int
    attention_mode: str
    video_temporal_causal: bool
    latent_patch_size: int
    vae_spatial_compression_factor: int
    temporal_compression_factor: int
    fixed_step_sampler_config: FixedStepSamplerConfig
    conditioning: CosmosDreamsConditioning
    temporal_modality_margin: int
    unified_3d_mrope_reset_spatial_ids: bool
    base_fps: float
    enable_fps_modulation: bool

    @property
    def t_list(self) -> tuple[float, ...]:
        return self.fixed_step_sampler_config.t_list

    @property
    def num_train_timesteps(self) -> int:
        return self.fixed_step_sampler_config.num_train_timesteps

    @property
    def conditioning_mode(self) -> str:
        return self.conditioning.mode

    @property
    def action_contract(self) -> CosmosDreamsActionConditioning:
        if not isinstance(
            self.conditioning,
            (CosmosDreamsActionContract, CosmosDreamsUnifiedActionContract),
        ):
            raise ValueError(
                f"Cosmos-Dreams checkpoint {self.checkpoint_id} is conditioned on "
                f"{self.conditioning_mode!r}, not on actions."
            )
        return self.conditioning

    @property
    def unified_actions(self) -> bool:
        """Schema-5 checkpoints take client-normalized 59-D unified rows."""
        return isinstance(self.conditioning, CosmosDreamsUnifiedActionContract)

    def action_input_dim(self, embodiment: str) -> int:
        """Width of the action rows a request supplies for ``embodiment``."""
        contract = self.action_contract.embodiments[embodiment]
        if isinstance(contract, UnifiedEmbodimentContract):
            return contract.input_action_dim
        return contract.raw_action_dim

    @property
    def training_prompt_as_json(self) -> bool:
        """Whether captions were wrapped in the JSON action prompt during training."""
        if isinstance(self.conditioning, CosmosDreamsUnifiedActionContract):
            return self.conditioning.training_prompt_as_json
        return True

    @property
    def control_contract(self) -> CosmosDreamsControlVideoContract:
        if not isinstance(self.conditioning, CosmosDreamsControlVideoContract):
            raise ValueError(
                f"Cosmos-Dreams checkpoint {self.checkpoint_id} is conditioned on "
                f"{self.conditioning_mode!r}, not on control video."
            )
        return self.conditioning

    @property
    def action_tokens_per_frame(self) -> int:
        """Action tokens interleaved before each frame's patches; 0 without actions."""
        if isinstance(
            self.conditioning,
            (CosmosDreamsActionContract, CosmosDreamsUnifiedActionContract),
        ):
            return self.conditioning.action_tokens_per_frame
        return 0

    @property
    def max_action_dim(self) -> int:
        return self.action_contract.model_action_dim


class _ManifestBody(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """``CosmosDreamsManifest`` without ``conditioning``, which is converted separately."""

    schema_version: int
    checkpoint_id: str
    checkpoint_iteration: int
    checkpoint_hash: str
    chunk_size: int
    window_frames: int | None
    sink_frames: int
    text_cache_max_len: int
    attention_mode: str
    video_temporal_causal: bool
    latent_patch_size: int
    vae_spatial_compression_factor: int
    temporal_compression_factor: int
    fixed_step_sampler_config: FixedStepSamplerConfig
    temporal_modality_margin: int
    unified_3d_mrope_reset_spatial_ids: bool
    base_fps: float
    enable_fps_modulation: bool


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Cosmos-Dreams manifest {name} must be a positive integer, got {value!r}."
        )


def _validate_sampler(sampler: FixedStepSamplerConfig) -> None:
    if sampler.sample_type != SAMPLE_TYPE_SDE:
        raise ValueError(
            f"Cosmos-Dreams sample_type must be {SAMPLE_TYPE_SDE!r}, "
            f"got {sampler.sample_type!r}."
        )
    _require_positive("num_train_timesteps", sampler.num_train_timesteps)
    t_list = sampler.t_list
    if not t_list:
        raise ValueError("Cosmos-Dreams t_list must not be empty.")
    if abs(t_list[0] - 1.0) > 1e-6:
        raise ValueError(f"Cosmos-Dreams t_list must start at 1.0, got {t_list[0]}.")
    if any(not math.isfinite(t) or t <= 0.0 or t > 1.0 for t in t_list):
        raise ValueError(
            f"Cosmos-Dreams t_list entries must be in (0, 1], got {t_list}."
        )
    if any(left <= right for left, right in zip(t_list, t_list[1:])):
        raise ValueError(
            f"Cosmos-Dreams t_list must be strictly descending, got {t_list}."
        )


def _validate_normalizer(
    embodiment: str, contract: EmbodimentContract, normalizer: ActionNormalizerContract
) -> None:
    transform = normalizer.transform
    if transform.type != AFFINE_TRANSFORM_TYPE or transform.forward_clamp:
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} needs an unclamped affine "
            "normalizer transform."
        )
    if len(transform.offset) != contract.raw_action_dim or len(transform.scale) != len(
        transform.offset
    ):
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} normalizer width must equal "
            f"raw_action_dim={contract.raw_action_dim}, got offset={len(transform.offset)} "
            f"scale={len(transform.scale)}."
        )
    if any(value <= 0.0 for value in transform.scale):
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} normalizer scales must be positive."
        )
    canonical_offset = tuple(float32_value(value) for value in transform.offset)
    canonical_scale = tuple(float32_value(value) for value in transform.scale)
    if canonical_offset != transform.offset or canonical_scale != transform.scale:
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} normalizer offset/scale must be "
            "encoded at float32 precision."
        )
    if normalizer.method == POSE_SCALE_METHOD:
        translation_scale = float(normalizer.derivation["translation_scale"])
        rotation_scale = float(normalizer.derivation["rotation_scale"])
        expected_scale = (float32_value(1.0 / translation_scale),) * 3 + (
            float32_value(1.0 / rotation_scale),
        ) * (contract.raw_action_dim - 3)
        if transform.offset != (0.0,) * contract.raw_action_dim or (
            transform.scale != expected_scale
        ):
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} pose_scale transform does not "
                "match its translation_scale/rotation_scale derivation."
            )
    if not _is_sha256_hex(normalizer.transform_sha256):
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} transform_sha256 is not a SHA-256 hex digest."
        )
    expected_hash = canonical_sha256(normalizer.behavioral_payload())
    if normalizer.transform_sha256 != expected_hash:
        raise ValueError(
            f"Cosmos-Dreams embodiment {embodiment!r} transform_sha256 does not match its "
            f"behavioral payload: expected {expected_hash}, got {normalizer.transform_sha256}."
        )


def _validate_action_contract(contract: CosmosDreamsActionContract) -> None:
    if contract.schema_version != ACTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Cosmos-Dreams action contract schema_version="
            f"{contract.schema_version}; expected {ACTION_CONTRACT_SCHEMA_VERSION}."
        )
    for name in (
        "action_tokens_per_frame",
        "model_action_dim",
        "num_embodiment_domains",
    ):
        _require_positive(name, getattr(contract, name))
    if not contract.embodiments:
        raise ValueError(
            "Cosmos-Dreams action contract must declare at least one embodiment."
        )
    if contract.default_embodiment not in contract.embodiments:
        raise ValueError(
            f"Cosmos-Dreams default_embodiment {contract.default_embodiment!r} is not declared."
        )
    if (
        contract.padding.stage != PADDING_STAGE_AFTER_NORMALIZATION
        or contract.padding.value != 0.0
    ):
        raise ValueError(
            "Cosmos-Dreams action padding must be zeros applied after normalization."
        )
    for embodiment, embodiment_contract in contract.embodiments.items():
        if not 0 <= embodiment_contract.domain_id < contract.num_embodiment_domains:
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} domain_id="
                f"{embodiment_contract.domain_id} is outside "
                f"[0, {contract.num_embodiment_domains})."
            )
        if not 0 < embodiment_contract.raw_action_dim <= contract.model_action_dim:
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} raw_action_dim="
                f"{embodiment_contract.raw_action_dim} must be in "
                f"[1, {contract.model_action_dim}]."
            )
        _validate_normalizer(
            embodiment, embodiment_contract, embodiment_contract.normalizer
        )
    if not _is_sha256_hex(contract.contract_sha256):
        raise ValueError("Cosmos-Dreams contract_sha256 is not a SHA-256 hex digest.")
    expected_hash = canonical_sha256(contract.behavioral_payload())
    if contract.contract_sha256 != expected_hash:
        raise ValueError(
            "Cosmos-Dreams action contract_sha256 does not match its behavioral payload: "
            f"expected {expected_hash}, got {contract.contract_sha256}."
        )


def _validate_unified_action_contract(
    contract: CosmosDreamsUnifiedActionContract, raw: dict[str, Any]
) -> None:
    if contract.schema_version != UNIFIED_ACTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Cosmos-Dreams unified action contract schema_version="
            f"{contract.schema_version}; expected {UNIFIED_ACTION_CONTRACT_SCHEMA_VERSION}."
        )
    for name in (
        "action_tokens_per_frame",
        "model_action_dim",
        "num_embodiment_domains",
    ):
        _require_positive(name, getattr(contract, name))
    expected_input = UnifiedInputContract(
        action_space=UNIFIED_ACTION_SPACE,
        action_dim=UNIFIED_ACTION_DIM,
        domain_routing=UNIFIED_DOMAIN_ROUTING,
        validity=UNIFIED_VALIDITY,
        runtime_normalization=False,
        model_mode="forward_dynamics",
    )
    if contract.input_contract != expected_input:
        raise ValueError(
            "Cosmos-Dreams unified input_contract is not the supported one: "
            f"got {msgspec.to_builtins(contract.input_contract)}, "
            f"expected {msgspec.to_builtins(expected_input)}."
        )
    layout = contract.layout
    if (
        layout.id != UNIFIED_LAYOUT_ID
        or layout.pose_convention != UNIFIED_POSE_CONVENTION
        or layout.rotation_representation != "rot6d_columns"
    ):
        raise ValueError(
            "Cosmos-Dreams unified layout must be unified_v1 with "
            f"{UNIFIED_POSE_CONVENTION} rot6d columns, got {msgspec.to_builtins(layout)}."
        )
    cursor = 0
    for field in layout.fields:
        if field.offset != cursor or field.size <= 0:
            raise ValueError(
                "Cosmos-Dreams unified layout fields must tile the row contiguously; "
                f"field {field.name!r} starts at {field.offset}, expected {cursor}."
            )
        cursor += field.size
    if cursor != UNIFIED_ACTION_DIM:
        raise ValueError(
            f"Cosmos-Dreams unified layout covers {cursor} slots, expected {UNIFIED_ACTION_DIM}."
        )
    if (
        contract.normalizer.method != UNIFIED_NORMALIZER_METHOD
        or contract.normalizer.runtime_application
    ):
        raise ValueError(
            "Cosmos-Dreams unified rows must arrive normalized with "
            f"{UNIFIED_NORMALIZER_METHOD}; got method={contract.normalizer.method!r}, "
            f"runtime_application={contract.normalizer.runtime_application}."
        )
    if (
        contract.padding.stage != PADDING_STAGE_AFTER_NORMALIZATION_AND_MASK
        or contract.padding.value != 0.0
    ):
        raise ValueError(
            "Cosmos-Dreams unified action padding must be zeros applied after "
            "normalization and validity masking."
        )
    if not contract.embodiments:
        raise ValueError(
            "Cosmos-Dreams unified action contract must declare at least one embodiment."
        )
    if contract.default_embodiment not in contract.embodiments:
        raise ValueError(
            f"Cosmos-Dreams default_embodiment {contract.default_embodiment!r} is not declared."
        )
    if contract.model_action_dim < UNIFIED_ACTION_DIM:
        raise ValueError(
            f"Cosmos-Dreams model_action_dim={contract.model_action_dim} cannot hold "
            f"{UNIFIED_ACTION_DIM}-D unified rows."
        )
    for embodiment, embodiment_contract in contract.embodiments.items():
        if not 0 <= embodiment_contract.domain_id < contract.num_embodiment_domains:
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} domain_id="
                f"{embodiment_contract.domain_id} is outside "
                f"[0, {contract.num_embodiment_domains})."
            )
        if embodiment_contract.input_action_dim != UNIFIED_ACTION_DIM:
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} input_action_dim="
                f"{embodiment_contract.input_action_dim} must be {UNIFIED_ACTION_DIM}."
            )
    if not _is_sha256_hex(contract.contract_sha256):
        raise ValueError("Cosmos-Dreams contract_sha256 is not a SHA-256 hex digest.")
    # The exporter hashes the whole block except the tag and the digest itself.
    payload = {
        key: value
        for key, value in raw.items()
        if key not in ("mode", "contract_sha256")
    }
    expected_hash = canonical_sha256(payload)
    if contract.contract_sha256 != expected_hash:
        raise ValueError(
            "Cosmos-Dreams unified contract_sha256 does not match its payload: "
            f"expected {expected_hash}, got {contract.contract_sha256}."
        )


def _validate_control_video_contract(
    contract: CosmosDreamsControlVideoContract,
) -> None:
    if not contract.hints or len(set(contract.hints)) != len(contract.hints):
        raise ValueError(
            "Cosmos-Dreams control_video contract must list distinct hints, got "
            f"{contract.hints!r}."
        )
    unknown = sorted(set(contract.hints) - set(TRANSFER_HINTS))
    if unknown:
        raise ValueError(
            f"Cosmos-Dreams control_video hints {unknown} are not supported; "
            f"expected a subset of {list(TRANSFER_HINTS)}."
        )
    if contract.transfer_control_attention_mode != TRANSFER_CONTROL_ATTENTION_MODE:
        raise ValueError(
            "Cosmos-Dreams control_video requires transfer_control_attention_mode="
            f"{TRANSFER_CONTROL_ATTENTION_MODE!r}, got "
            f"{contract.transfer_control_attention_mode!r}."
        )
    if not contract.share_vision_temporal_positions:
        raise ValueError(
            "Cosmos-Dreams control_video requires share_vision_temporal_positions=True."
        )
    if contract.system_prompt_id != TRANSFER_SYSTEM_PROMPT_ID:
        raise ValueError(
            f"Cosmos-Dreams control_video requires system_prompt_id={TRANSFER_SYSTEM_PROMPT_ID!r}, "
            f"got {contract.system_prompt_id!r}."
        )
    if not contract.no_eviction:
        raise ValueError(
            "Cosmos-Dreams control_video with a finite K/V window is not supported; "
            "the artifact must set no_eviction=true."
        )


def validate_cosmos_dreams_manifest(
    manifest: CosmosDreamsManifest, raw_conditioning: dict[str, Any] | None = None
) -> None:
    """Reject artifacts the causal runtime cannot honor exactly.

    ``raw_conditioning`` is the artifact's conditioning block as exported; the
    schema-5 digest covers it verbatim.
    """
    if manifest.schema_version != COSMOS_DREAMS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Cosmos-Dreams artifact schema_version={manifest.schema_version}; "
            f"expected {COSMOS_DREAMS_SCHEMA_VERSION}."
        )
    for name in (
        "chunk_size",
        "text_cache_max_len",
        "latent_patch_size",
        "vae_spatial_compression_factor",
        "temporal_compression_factor",
        "temporal_modality_margin",
        "checkpoint_iteration",
    ):
        _require_positive(name, getattr(manifest, name))
    if manifest.window_frames is not None:
        _require_positive("window_frames", manifest.window_frames)
    if isinstance(manifest.sink_frames, bool) or manifest.sink_frames < 0:
        raise ValueError(
            f"Cosmos-Dreams manifest sink_frames must be non-negative, got {manifest.sink_frames!r}."
        )
    if manifest.window_frames is None and manifest.sink_frames:
        raise ValueError(
            "Cosmos-Dreams manifest pins sink_frames without a K/V window "
            f"(window_frames=None, sink_frames={manifest.sink_frames})."
        )
    if manifest.attention_mode != ATTENTION_MODE_THREE_WAY:
        raise ValueError(
            f"Cosmos-Dreams requires attention_mode={ATTENTION_MODE_THREE_WAY!r}, "
            f"got {manifest.attention_mode!r}."
        )
    if not manifest.video_temporal_causal:
        raise ValueError("Cosmos-Dreams requires video_temporal_causal=True.")
    if not manifest.unified_3d_mrope_reset_spatial_ids:
        raise ValueError(
            "Cosmos-Dreams requires unified_3d_mrope_reset_spatial_ids=True."
        )
    if not manifest.enable_fps_modulation:
        raise ValueError("Cosmos-Dreams AR inference requires FPS modulation.")
    if not math.isfinite(manifest.base_fps) or manifest.base_fps <= 0:
        raise ValueError(
            f"Cosmos-Dreams base_fps must be positive, got {manifest.base_fps}."
        )
    if not manifest.checkpoint_id or manifest.checkpoint_id == "unknown":
        raise ValueError("Cosmos-Dreams requires an exported checkpoint_id.")
    if not _is_sha256_hex(manifest.checkpoint_hash) or set(
        manifest.checkpoint_hash
    ) == {"0"}:
        raise ValueError(
            "Cosmos-Dreams checkpoint_hash must be a real 64-character SHA-256 digest, "
            f"got {manifest.checkpoint_hash!r}."
        )
    _validate_sampler(manifest.fixed_step_sampler_config)
    conditioning = manifest.conditioning
    if isinstance(
        conditioning, (CosmosDreamsActionContract, CosmosDreamsUnifiedActionContract)
    ):
        if isinstance(conditioning, CosmosDreamsActionContract):
            _validate_action_contract(conditioning)
        else:
            _validate_unified_action_contract(conditioning, raw_conditioning or {})
        if conditioning.action_tokens_per_frame != manifest.temporal_compression_factor:
            raise ValueError(
                "Cosmos-Dreams action_tokens_per_frame must equal temporal_compression_factor; "
                f"got {conditioning.action_tokens_per_frame} and "
                f"{manifest.temporal_compression_factor}."
            )
    else:
        _validate_control_video_contract(conditioning)


def _conditioning_type(raw: Any) -> type:
    """Pick the contract struct from the block's mode and schema version."""
    if not isinstance(raw, dict):
        raise ValueError("Cosmos-Dreams artifact conditioning must be an object.")
    mode = raw.get("mode")
    if mode == CONTROL_VIDEO_CONDITIONING_MODE:
        return CosmosDreamsControlVideoContract
    if mode != ACTION_CONDITIONING_MODE:
        raise ValueError(
            f"Cosmos-Dreams artifact conditioning mode must be "
            f"{ACTION_CONDITIONING_MODE!r} or {CONTROL_VIDEO_CONDITIONING_MODE!r}, got {mode!r}."
        )
    schema = raw.get("schema_version")
    if schema == UNIFIED_ACTION_CONTRACT_SCHEMA_VERSION:
        return CosmosDreamsUnifiedActionContract
    if schema == ACTION_CONTRACT_SCHEMA_VERSION:
        return CosmosDreamsActionContract
    raise ValueError(
        "Unsupported Cosmos-Dreams action contract schema_version="
        f"{schema!r}; expected {ACTION_CONTRACT_SCHEMA_VERSION} or "
        f"{UNIFIED_ACTION_CONTRACT_SCHEMA_VERSION}."
    )


def parse_cosmos_dreams_manifest(artifact: dict[str, Any]) -> CosmosDreamsManifest:
    """Parse and validate the exporter's transformer-config artifact block."""
    if not isinstance(artifact, dict) or "conditioning" not in artifact:
        raise ValueError(
            "Cosmos-Dreams schema-v1 artifact is invalid: missing `conditioning`."
        )
    raw_conditioning = artifact["conditioning"]
    body = {key: value for key, value in artifact.items() if key != "conditioning"}
    try:
        conditioning = msgspec.convert(
            raw_conditioning, type=_conditioning_type(raw_conditioning)
        )
        fields = msgspec.structs.asdict(msgspec.convert(body, type=_ManifestBody))
        manifest = CosmosDreamsManifest(conditioning=conditioning, **fields)
    except msgspec.ValidationError as exc:
        raise ValueError(f"Cosmos-Dreams schema-v1 artifact is invalid: {exc}") from exc
    validate_cosmos_dreams_manifest(
        manifest, raw_conditioning if isinstance(raw_conditioning, dict) else None
    )
    return manifest


# The imaginaire4 exporter writes one envelope for both Sim-Bimanual and
# Sim-Depth; the pipeline variant is selected by model_index.json's _class_name.
COSMOS_DREAMS_ARTIFACT_KEY = "cosmos3_nano_sim_bimanual"
_LEGACY_ARTIFACT_KEY = "cosmos_dreams"


def load_cosmos_dreams_manifest(
    transformer_config: dict[str, Any],
) -> CosmosDreamsManifest:
    """Read the artifact from a diffusers transformer ``config.json`` dict."""
    artifact = transformer_config.get(COSMOS_DREAMS_ARTIFACT_KEY)
    if not isinstance(artifact, dict) or not artifact:
        if _LEGACY_ARTIFACT_KEY in transformer_config:
            raise ValueError(
                "Legacy Cosmos-Dreams exports (transformer/config.json"
                f"[{_LEGACY_ARTIFACT_KEY!r}]) are no longer supported; re-export the "
                "checkpoint with the imaginaire4 --cosmos3-nano-sim-bimanual flow."
            )
        raise ValueError(
            "Cosmos-Dreams requires a schema-v1 artifact under "
            f"transformer/config.json[{COSMOS_DREAMS_ARTIFACT_KEY!r}]; this checkpoint "
            f"has keys {sorted(transformer_config)}."
        )
    return parse_cosmos_dreams_manifest(artifact)


class CosmosDreamsInferenceProfile(msgspec.Struct, frozen=True):
    """Rollout settings resolved from the artifact plus deployment overrides.

    The exporter fills ``window_frames`` with a default when training set no KV
    window and never records a per-frame step budget, so both are deployment settings.
    """

    # Sigma schedule of the chunk whose first latent frame is the index; the
    # last entry repeats for later chunks.
    frame_sigma_schedules: tuple[tuple[float, ...], ...]
    # Latest latent frames of committed K/V kept after the sink frames.
    window_frames: int
    sink_frames: int
    history_mode: str

    def sigmas_for_frame(self, frame_idx: int) -> tuple[float, ...]:
        if frame_idx < 0:
            raise ValueError(f"frame_idx must be non-negative, got {frame_idx}.")
        schedules = self.frame_sigma_schedules
        return schedules[min(frame_idx, len(schedules) - 1)]

    @property
    def max_steps(self) -> int:
        return max(len(schedule) for schedule in self.frame_sigma_schedules)


def default_frame_sigma_schedules(
    t_list: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """The self-forcing "step42" budget the Sim-Bimanual checkpoints were distilled with:
    every ``t_list`` entry for a chunk starting at latent frame 0, then two steps
    (``t_list[0]`` and ``t_list[2]``) for every later chunk."""
    sigmas = tuple(float(value) for value in t_list)
    if len(sigmas) < 3:
        return (sigmas,)
    return (sigmas, (sigmas[0], sigmas[2]))


def _validate_sigma_schedule(
    schedule: Any, index: int, t_list: Sequence[float]
) -> tuple[float, ...]:
    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise ValueError(
            f"Cosmos-Dreams frame_sigma_schedules[{index}] must be a non-empty list of sigmas."
        )
    sigmas: list[float] = []
    for value in schedule:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Cosmos-Dreams frame_sigma_schedules[{index}] must contain numbers, got {value!r}."
            )
        sigma = float(value)
        if not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
            raise ValueError(
                f"Cosmos-Dreams frame_sigma_schedules[{index}] entries must lie in (0, 1], got {sigma}."
            )
        # A distilled fixed-step model only saw the artifact's timesteps.
        if not any(abs(sigma - trained) <= 1e-6 for trained in t_list):
            raise ValueError(
                f"Cosmos-Dreams frame_sigma_schedules[{index}] sigma {sigma} is not in the "
                f"artifact's distilled t_list {list(t_list)}."
            )
        sigmas.append(sigma)
    if abs(sigmas[0] - 1.0) > 1e-6:
        raise ValueError(
            f"Cosmos-Dreams frame_sigma_schedules[{index}] must start at 1.0, got {sigmas[0]}."
        )
    if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError(
            f"Cosmos-Dreams frame_sigma_schedules[{index}] must be strictly descending, got {sigmas}."
        )
    return tuple(sigmas)


def resolve_inference_profile(
    manifest: CosmosDreamsManifest,
    *,
    frame_sigma_schedules: Any = None,
    history_mode: str = HISTORY_MODE_FULL,
    history_max_frames: int,
) -> CosmosDreamsInferenceProfile:
    """Resolve ``frame_sigma_schedules`` (None: the step42 budget derived from the artifact's
    t_list) and the history length (``full``: the first ``history_max_frames`` pixel frames;
    ``sliding``: the artifact's ``window_frames``) into one validated profile."""
    if frame_sigma_schedules is None:
        schedules = default_frame_sigma_schedules(manifest.t_list)
    else:
        if (
            not isinstance(frame_sigma_schedules, (list, tuple))
            or not frame_sigma_schedules
        ):
            raise ValueError(
                "Cosmos-Dreams frame_sigma_schedules must be a non-empty list of sigma lists."
            )
        schedules = tuple(
            _validate_sigma_schedule(schedule, index, manifest.t_list)
            for index, schedule in enumerate(frame_sigma_schedules)
        )
    if history_mode not in HISTORY_MODES:
        raise ValueError(
            f"Cosmos-Dreams history_mode must be one of {HISTORY_MODES}, got {history_mode!r}."
        )
    if history_mode == HISTORY_MODE_FULL:
        factor = manifest.temporal_compression_factor
        if (
            isinstance(history_max_frames, bool)
            or not isinstance(history_max_frames, int)
            or history_max_frames <= 1
            or (history_max_frames - 1) % factor
        ):
            raise ValueError(
                "Cosmos-Dreams history_max_frames must be 1 + "
                f"{factor} * N pixel frames, got {history_max_frames!r}."
            )
        window_frames = (history_max_frames - 1) // factor + 1
    else:
        if manifest.window_frames is None:
            raise ValueError(
                "Cosmos-Dreams history_mode='sliding' needs a K/V window, but the "
                f"artifact {manifest.checkpoint_id} declares window_frames=None."
            )
        window_frames = manifest.window_frames
    return CosmosDreamsInferenceProfile(
        frame_sigma_schedules=schedules,
        window_frames=window_frames,
        sink_frames=manifest.sink_frames,
        history_mode=history_mode,
    )


class CosmosDreamsTransferHistoryProfile(msgspec.Struct, frozen=True):
    """Committed K/V kept by a control-video rollout, in cache entries.

    A logical frame is two entries, its control latent and its clean RGB latent.
    The reference exposes ``window_frames`` logical frames to the frame being
    generated: ``sink_frames`` pinned pairs, the newest ``recent`` pairs, and the
    frame's own control; ``recent = window_frames - sink_frames - 1``.
    """

    history_mode: str
    window_frames: int | None
    sink_frames: int

    @property
    def sink_entries(self) -> int:
        return 0 if self.window_frames is None else 2 * self.sink_frames

    @property
    def recent_pairs(self) -> int | None:
        if self.window_frames is None:
            return None
        return self.window_frames - self.sink_frames - 1

    def entries_after_control_commit(self) -> int | None:
        """History the RGB target may read: recent pairs plus its own control."""
        return None if self.recent_pairs is None else 2 * self.recent_pairs + 1

    def entries_after_rgb_commit(self) -> int | None:
        """History the next control seed may read: whole recent pairs only."""
        return None if self.recent_pairs is None else 2 * self.recent_pairs


def resolve_transfer_history_profile(
    manifest: CosmosDreamsManifest, *, history_mode: str = HISTORY_MODE_FULL
) -> CosmosDreamsTransferHistoryProfile:
    """``full`` keeps every committed entry; ``sliding`` applies the artifact's
    ``window_frames``/``sink_frames`` like the reference's finite-window path."""
    if history_mode not in HISTORY_MODES:
        raise ValueError(
            f"Cosmos-Dreams history_mode must be one of {HISTORY_MODES}, got {history_mode!r}."
        )
    if history_mode == HISTORY_MODE_FULL:
        return CosmosDreamsTransferHistoryProfile(
            history_mode=history_mode, window_frames=None, sink_frames=0
        )
    if manifest.chunk_size != 1:
        raise ValueError(
            "Cosmos-Dreams control_video sliding history requires chunk_size=1 "
            f"(the reference finite-window path); the artifact has chunk_size={manifest.chunk_size}."
        )
    if manifest.window_frames - manifest.sink_frames < 2:
        raise ValueError(
            "Cosmos-Dreams control_video sliding history needs window_frames - sink_frames >= 2, "
            f"got window_frames={manifest.window_frames}, sink_frames={manifest.sink_frames}."
        )
    return CosmosDreamsTransferHistoryProfile(
        history_mode=history_mode,
        window_frames=manifest.window_frames,
        sink_frames=manifest.sink_frames,
    )


# Rotation identity in rot6d column form: the first two columns of I3.
ROT6D_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class UnifiedNormalizer(msgspec.Struct, frozen=True):
    """Client-side ``global_asinh`` normalizer of unified_v1 rows.

    ``y = asinh((raw - offset) / scale) / asinh(1)`` with ``offset = (q01 + q99) / 2``
    and ``scale = (q99 - q01) / 2`` from the exported statistics (imaginaire4
    ``ActionAsinhNormalization``). Slots a source lacks stay zero after normalization.
    """

    offset: tuple[float, ...]
    scale: tuple[float, ...]

    def normalize(self, raw: list[float]) -> list[float]:
        if len(raw) != len(self.offset):
            raise ValueError(
                f"Unified rows have {len(self.offset)} slots, got {len(raw)}."
            )
        unit = math.asinh(1.0)
        return [
            math.asinh((value - offset) / scale) / unit
            for value, offset, scale in zip(raw, self.offset, self.scale, strict=True)
        ]

    def identity_row(self, layout: UnifiedLayout, *, embodiment: str) -> list[float]:
        """Normalized "no motion" row: identity poses in the slots the embodiment
        fills (ego pose; wrists for non-camera embodiments), zeros elsewhere."""
        raw = [0.0] * len(self.offset)
        pose_fields = ["ego_pose"]
        if embodiment != CAMERA_EMBODIMENT:
            pose_fields += ["right_wrist_pose", "left_wrist_pose"]
        filled: list[int] = []
        for field in layout.fields:
            if field.name in pose_fields and field.size == 9:
                raw[field.offset + 3 : field.offset + 9] = ROT6D_IDENTITY
                filled.extend(range(field.offset, field.offset + 9))
        normalized = self.normalize(raw)
        return [normalized[i] if i in set(filled) else 0.0 for i in range(len(raw))]


def load_unified_normalizer(
    model_path: str, contract: CosmosDreamsUnifiedActionContract
) -> UnifiedNormalizer:
    """Read the statistics file the contract's ``normalizer.source`` names."""
    source = contract.normalizer.source
    artifact_path = source.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError(
            "Cosmos-Dreams unified normalizer source has no artifact_path."
        )
    path = os.path.join(model_path, artifact_path)
    with open(path, "rb") as handle:
        data = handle.read()
    expected = source.get("sha256")
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(
            f"Cosmos-Dreams unified normalizer statistics at {path} do not match "
            f"the contract's sha256 {expected}."
        )
    stats = json.loads(data)
    block = stats.get("global")
    if not isinstance(block, dict) or "q01" not in block or "q99" not in block:
        raise ValueError(
            f"Cosmos-Dreams unified normalizer statistics at {path} lack global q01/q99."
        )
    low, high = block["q01"], block["q99"]
    if len(low) != UNIFIED_ACTION_DIM or len(high) != UNIFIED_ACTION_DIM:
        raise ValueError(
            f"Cosmos-Dreams unified normalizer statistics must have {UNIFIED_ACTION_DIM} "
            f"entries, got {len(low)} and {len(high)}."
        )
    offset = tuple(float((lo + hi) / 2.0) for lo, hi in zip(low, high, strict=True))
    # imaginaire4 clamps the half range at 1e-8 so constant slots stay finite.
    scale = tuple(
        float(max((hi - lo) / 2.0, 1e-8)) for lo, hi in zip(low, high, strict=True)
    )
    return UnifiedNormalizer(offset=offset, scale=scale)
