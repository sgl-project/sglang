# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV pipeline configuration.

``transformer/config.json`` marks the checkpoint with
``backbone_type: cosmos3_multiview`` and a ``multiview`` object that fixes the
camera list and the attention contract the weights were trained with. Unversioned v1 exports
ship the regular Cosmos3 Nano weight layout; schema-2 exports add the
``lidar_proj_in/out`` projections and a ``lidar_vae/`` component for joint
camera/LiDAR generation. This config reuses ``Cosmos3Config`` (Wan VAE, Qwen2
tokenizer, FlowUniPC) and swaps in the transformer subclass that installs the
maskless cross-camera attention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import msgspec

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    _transformer_config,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_maskless import (
    maskless_unavailable_reason,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COSMOS3_MULTIVIEW_BACKBONE_TYPE = "cosmos3_multiview"
COSMOS3_MULTIVIEW_ATTENTION_SCOPES = ("all_views", "same_view", "decomposed")
# ``triton``/``fa4`` name the masked attention family: one visibility mask that
# counts every permitted key once (the Sep-14 and the Oct-1 2026 exports; the
# latter trained its folds exact-count with a 0.4 s past window, which the mask
# expresses directly). ``maskless`` is the three-fold attention of the Sep-18/22
# exports, which double-counts the query's own cell. A checkpoint is served
# faithfully only by its own family; within the masked family the kernel is a
# speed knob.
COSMOS3_MULTIVIEW_ATTENTION_BACKENDS = ("triton", "fa4", "maskless")
COSMOS3_MULTIVIEW_MASKED_BACKENDS = ("triton", "fa4")
# Masked-family kernel left to the device: FA4 block-sparse on Blackwell,
# FlexAttention's Triton kernel elsewhere.
COSMOS3_MULTIVIEW_AUTO_BACKEND = "auto"
MULTIVIEW_BACKEND_ENV_VAR = "SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND"
# Versioned exports this build reads. Version 3 (Oct 1 2026) adds the rig view
# embedding and a LiDAR patch that may differ from the camera patch.
COSMOS3_MULTIVIEW_SCHEMA_VERSIONS = (2, 3)
# Every top-level field the imaginaire4 exporter writes for a servable artifact.
# A versioned contract carrying anything else fails loudly instead of being
# silently ignored.
COSMOS3_MULTIVIEW_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "causal_training_strategy",
        "attention_scope",
        "backend",
        "decomposed_temporal_window_seconds",
        "control_attends_sensor",
        "lidar_attends_captions",
        "align_temporal_positions_across_views",
        "share_vision_temporal_positions",
        "cameras",
        "max_views",
        "per_view_captions",
        "variable_view_count",
        "inference_defaults",
        "lidar",
        "lidar_patch_spatial_hw",
        "rig_view_embedding",
        # Written by 2026-09 exporters before per_view_captions replaced it.
        "separate_view_text_tokenization",
        # Optional: an export may pin which system prompt wording it trained under.
        "system_prompt_variant",
    }
)
# AV system prompt wordings: "wsm_controls" is the Sep-15-2026-onward training text
# (names WSM, adds a control-adherence paragraph) every versioned export trained
# under; "provided_controls" is the earlier wording of the unversioned Sep-14
# export. ``system_prompt_variant`` in the export overrides the vintage rule.
COSMOS3_SYSTEM_PROMPT_VARIANTS = ("provided_controls", "wsm_controls")

# The fixed 11-camera MADS rig order the v1 checkpoint was exported with.
COSMOS3_MADS_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)


# Stable camera attributes the per-camera caption headers are rendered from
# (the MADS rig description the separate-view-caption checkpoints trained on).
COSMOS3_MADS_CAMERA_ATTRIBUTES: dict[str, dict[str, str | int]] = {
    "camera_front_wide_120fov": {
        "camera_role": "front",
        "camera_type": "wide",
        "facing": "forward",
        "fov_degrees": 120,
    },
    "camera_cross_right_120fov": {
        "camera_role": "right_side",
        "camera_type": "wide",
        "facing": "right",
        "fov_degrees": 120,
    },
    "camera_rear_right_70fov": {
        "camera_role": "rear_right",
        "camera_type": "standard",
        "facing": "rear_right",
        "fov_degrees": 70,
    },
    "camera_rear_tele_30fov": {
        "camera_role": "rear",
        "camera_type": "telephoto",
        "facing": "backward",
        "fov_degrees": 30,
    },
    "camera_rear_left_70fov": {
        "camera_role": "rear_left",
        "camera_type": "standard",
        "facing": "rear_left",
        "fov_degrees": 70,
    },
    "camera_cross_left_120fov": {
        "camera_role": "left_side",
        "camera_type": "wide",
        "facing": "left",
        "fov_degrees": 120,
    },
    "camera_front_tele_30fov": {
        "camera_role": "front",
        "camera_type": "telephoto",
        "facing": "forward",
        "fov_degrees": 30,
    },
    "camera_front_fisheye_200fov": {
        "camera_role": "front",
        "camera_type": "fisheye",
        "facing": "forward",
        "fov_degrees": 200,
    },
    "camera_left_fisheye_200fov": {
        "camera_role": "left_side",
        "camera_type": "fisheye",
        "facing": "left",
        "fov_degrees": 200,
    },
    "camera_right_fisheye_200fov": {
        "camera_role": "right_side",
        "camera_type": "fisheye",
        "facing": "right",
        "fov_degrees": 200,
    },
    "camera_rear_fisheye_200fov": {
        "camera_role": "rear",
        "camera_type": "fisheye",
        "facing": "backward",
        "fov_degrees": 200,
    },
}

# Schema-2 exports carry these request defaults; the sampling params fall
# back to them for fields the request leaves unset.
COSMOS3_MULTIVIEW_INFERENCE_DEFAULT_KEYS = frozenset(
    {
        "resolution",
        "fps",
        "num_steps",
        "guidance",
        "shift",
        "control_guidance",
        "emphasize_control_in_prompt",
        "guidance_interval",
        "control_guidance_interval",
        "sigma_max",
        "normalize_cfg",
        "negative_metadata_mode",
    }
)
COSMOS3_MULTIVIEW_RESOLUTIONS = ("480", "720")

# The V1.2 LiDAR tokenizer contract the joint checkpoints were exported with.
COSMOS3_LIDAR_TOKENIZER_VERSION = "1.2"
COSMOS3_LIDAR_RANGE_PROJECTION = {
    "semantic_width": 1800,
    "model_width": 1808,
    "native_height": 128,
    "model_width_transform": "circular_pad",
    "intensity_encoding": "unit",
}


class Cosmos3MultiviewDeploymentConfig(msgspec.Struct, frozen=True):
    """The validated ``multiview`` block of ``transformer/config.json``.

    Unversioned exports (``schema_version`` None) are the v1 WSM artifacts: the
    fixed MADS camera order, one prompt for the whole rig, no LiDAR. Schema-2
    exports add per-camera captions, camera subsets, request defaults, and
    optionally the joint camera/LiDAR contract.
    """

    cameras: tuple[str, ...]
    attention_scope: str
    decomposed_temporal_window_seconds: float | None
    control_attends_sensor: bool
    align_temporal_positions_across_views: bool
    share_vision_temporal_positions: bool
    backend: str
    schema_version: int | None = None
    #: One caption per camera (the export's ``per_view_captions``; older exports
    #: spelled it ``separate_view_text_tokenization``).
    per_view_captions: bool = False
    variable_view_count: bool = False
    inference_defaults: dict[str, Any] | None = None
    lidar: dict[str, Any] | None = None
    lidar_attends_captions: bool = True
    #: Which AV system prompt wording the captions were trained under: the
    #: export vintage decides unless the export records ``system_prompt_variant``.
    system_prompt_variant: str = "wsm_controls"
    #: LiDAR token patch ``(height, width)`` in latent cells; None follows the
    #: camera patch (schema-2 exports). The Oct-1 2026 joint export uses (1, 1).
    lidar_patch_spatial_hw: tuple[int, int] | None = None
    #: Physical rig identity table of schema-3 exports: ``camera_ids`` maps every
    #: exported camera to its trained embedding row, ``lidar_id`` is the final row.
    rig_view_embedding: dict[str, Any] | None = None

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def uses_masked_attention(self) -> bool:
        return self.backend in COSMOS3_MULTIVIEW_MASKED_BACKENDS

    def rig_view_ids(self, camera_keys: Sequence[str]) -> list[int] | None:
        """Trained embedding rows for the request's cameras, in request order.

        Subsets and reordered views keep each camera's physical id. None when
        the export has no rig table.
        """
        if self.rig_view_embedding is None:
            return None
        table = self.rig_view_embedding["camera_ids"]
        unknown = [key for key in camera_keys if key not in table]
        if unknown:
            raise ValueError(
                f"Cosmos3 multiview rig view embedding has no row for cameras {unknown}."
            )
        return [int(table[key]) for key in camera_keys]

    @property
    def is_legacy(self) -> bool:
        return self.schema_version is None

    @property
    def supports_lidar(self) -> bool:
        return self.lidar is not None

    def inference_default(self, name: str, fallback: Any) -> Any:
        """A request default the export declares, else ``fallback``."""
        if self.inference_defaults is None:
            return fallback
        value = self.inference_defaults.get(name)
        return fallback if value is None else value


def _per_view_captions_flag(raw: Mapping[str, Any]) -> bool:
    """The caption layout flag under either of its two names.

    Exports from Sep 22 2026 on (HF 38f182c) write ``per_view_captions``, the key
    imaginaire4's exporter and vLLM-Omni use; earlier schema-2 exports wrote
    ``separate_view_text_tokenization``. Both are accepted; they must agree.
    """
    values = {
        name: raw[name]
        for name in ("per_view_captions", "separate_view_text_tokenization")
        if name in raw
    }
    if not values:
        raise ValueError(
            "Cosmos3 multiview transformer config requires field 'per_view_captions' "
            "(or the older 'separate_view_text_tokenization')."
        )
    for name, value in values.items():
        if not isinstance(value, bool):
            raise TypeError(f"Cosmos3 multiview {name} must be boolean.")
    if len(set(values.values())) > 1:
        raise ValueError(
            f"Cosmos3 multiview per_view_captions flags disagree: {values}."
        )
    return next(iter(values.values()))


def _required_field(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(
            f"Cosmos3 multiview transformer config requires field {name!r}."
        )
    return config[name]


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
    )


def validate_lidar_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exported V1.2 LiDAR tokenizer contract."""
    required = {
        "version",
        "fps",
        "latent_channels",
        "temporal_compression_factor",
        "spatial_compression",
        "network_config",
        "range_projection",
        "streaming_chunk_frames",
        "streaming_context_frames",
    }
    if missing := required - set(config):
        raise ValueError(
            f"Incomplete Cosmos3 LiDAR metadata: missing {sorted(missing)}."
        )
    if str(config["version"]) != COSMOS3_LIDAR_TOKENIZER_VERSION:
        raise ValueError(
            f"Only the V{COSMOS3_LIDAR_TOKENIZER_VERSION} LiDAR tokenizer is supported, "
            f"got {config['version']!r}."
        )
    projection = config["range_projection"]
    if not isinstance(projection, Mapping):
        raise TypeError("Cosmos3 LiDAR range_projection must be an object.")
    mismatched = {
        key: projection.get(key)
        for key, value in COSMOS3_LIDAR_RANGE_PROJECTION.items()
        if projection.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "Cosmos3 LiDAR V1.2 requires the 128x1800 metric grid circularly padded "
            f"to 1808 with unit intensity; got {mismatched}."
        )
    minimum, maximum = projection.get("min_range_m"), projection.get("max_range_m")
    if not (_is_number(minimum) and _is_number(maximum)) or maximum <= minimum:
        raise ValueError(
            "Cosmos3 LiDAR range normalization requires finite min_range_m < max_range_m."
        )
    network = config["network_config"]
    if not isinstance(network, Mapping):
        raise TypeError("Cosmos3 LiDAR network_config must be an object.")
    spatial = list(config["spatial_compression"])
    if (
        list(network.get("resolution", ())) != [128, 1808]
        or list(network.get("patch_size", ())) != [2, 2]
        or len(network.get("depths", ())) != 4
        or spatial != [16, 16]
        or int(config["temporal_compression_factor"]) != 1
        or network.get("z_dim") != config["latent_channels"]
        or network.get("in_channels") != 3
        or any(network.get("temporal_downsample", [True]))
    ):
        raise ValueError(
            "Cosmos3 LiDAR architecture and V1.2 compression metadata disagree."
        )
    if not _is_number(config["fps"]) or config["fps"] <= 0:
        raise ValueError("Cosmos3 LiDAR fps must be finite and positive.")
    chunk, context = (
        config["streaming_chunk_frames"],
        config["streaming_context_frames"],
    )
    if (
        isinstance(chunk, bool)
        or not isinstance(chunk, int)
        or chunk < 1
        or (
            context is not None
            and (
                isinstance(context, bool)
                or not isinstance(context, int)
                or context < chunk
            )
        )
    ):
        raise ValueError(
            "Cosmos3 LiDAR streaming context must be at least the positive chunk length."
        )
    return dict(config)


def _validate_inference_defaults(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("Cosmos3 multiview inference_defaults must be an object.")
    if missing := COSMOS3_MULTIVIEW_INFERENCE_DEFAULT_KEYS - set(raw):
        raise ValueError(
            f"Incomplete Cosmos3 multiview inference_defaults: missing {sorted(missing)}."
        )
    if str(raw["resolution"]) not in COSMOS3_MULTIVIEW_RESOLUTIONS:
        raise ValueError(
            "Cosmos3 multiview inference_defaults.resolution must be one of "
            f"{list(COSMOS3_MULTIVIEW_RESOLUTIONS)}, got {raw['resolution']!r}."
        )
    for name in (
        "fps",
        "num_steps",
        "guidance",
        "shift",
        "control_guidance",
        "sigma_max",
    ):
        if not _is_number(raw[name]) or raw[name] < 0:
            raise ValueError(
                f"Cosmos3 multiview inference_defaults.{name} must be finite and non-negative."
            )
    if raw["fps"] == 0 or raw["num_steps"] < 1 or raw["shift"] == 0:
        raise ValueError(
            "Cosmos3 multiview inference_defaults requires positive fps, num_steps, and shift."
        )
    defaults = dict(raw)
    defaults["resolution"] = str(defaults["resolution"])
    return defaults


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validated_lidar_patch(
    raw: Mapping[str, Any], schema_version: int | None, camera_patch: int
) -> tuple[int, int] | None:
    """The LiDAR stream's ``(height, width)`` patch, defaulting to the camera patch."""
    patch = raw.get("lidar_patch_spatial_hw")
    if raw.get("lidar") is None:
        if patch is not None:
            raise ValueError(
                "Cosmos3 multiview lidar_patch_spatial_hw requires a lidar block."
            )
        return None
    if patch is None:
        return (camera_patch, camera_patch)
    if (
        not isinstance(patch, list | tuple)
        or len(patch) != 2
        or not all(_positive_int(side) for side in patch)
    ):
        raise ValueError(
            "Cosmos3 multiview lidar_patch_spatial_hw must be two positive integers, "
            f"got {patch!r}."
        )
    patch = (int(patch[0]), int(patch[1]))
    if schema_version == 2 and patch != (camera_patch, camera_patch):
        raise ValueError(
            "Cosmos3 multiview schema_version=2 requires the LiDAR patch to equal the "
            f"camera patch {camera_patch}, got {list(patch)}; a different LiDAR patch "
            "requires schema_version=3."
        )
    return patch


def _validated_rig_view_embedding(
    raw: Mapping[str, Any], schema_version: int | None, cameras: Sequence[str]
) -> dict[str, Any] | None:
    """Validate the physical rig-id table: one row per exported camera plus a final LiDAR row."""
    rig = raw.get("rig_view_embedding")
    if rig is None:
        return None
    if schema_version != 3:
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding requires schema_version=3."
        )
    if not isinstance(rig, Mapping):
        raise TypeError("Cosmos3 multiview rig_view_embedding must be an object.")
    if unknown := set(rig) - {"num_embeddings", "camera_ids", "lidar_id"}:
        raise ValueError(
            f"Unknown Cosmos3 multiview rig_view_embedding fields: {sorted(unknown)}."
        )
    num_embeddings = rig.get("num_embeddings")
    if not _positive_int(num_embeddings) or num_embeddings < 2:
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.num_embeddings must be an integer "
            f">= 2, got {num_embeddings!r}."
        )
    camera_ids = rig.get("camera_ids")
    if not isinstance(camera_ids, Mapping):
        raise TypeError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must be an object."
        )
    if set(camera_ids) != set(cameras):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must name exactly the "
            f"exported cameras: expected={sorted(cameras)}, got={sorted(camera_ids)}."
        )
    for camera, row in camera_ids.items():
        # The final row is reserved for LiDAR.
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row <= num_embeddings - 2
        ):
            raise ValueError(
                f"Cosmos3 multiview rig_view_embedding.camera_ids[{camera!r}] must be an "
                f"integer in [0, {num_embeddings - 2}], got {row!r}."
            )
    if len(set(camera_ids.values())) != len(camera_ids):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must assign distinct rows."
        )
    lidar_id = rig.get("lidar_id")
    if (
        isinstance(lidar_id, bool)
        or not isinstance(lidar_id, int)
        or lidar_id != num_embeddings - 1
    ):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.lidar_id must be the final row "
            f"{num_embeddings - 1}, got {lidar_id!r}."
        )
    return {
        "num_embeddings": int(num_embeddings),
        "camera_ids": {str(camera): int(row) for camera, row in camera_ids.items()},
        "lidar_id": int(lidar_id),
    }


def parse_multiview_deployment_config(
    transformer_config: Mapping[str, Any],
) -> Cosmos3MultiviewDeploymentConfig:
    """Validate the exported multiview contract before any weights are loaded.

    The backbone is inspected before the multiview fields so selecting this
    pipeline for another Cosmos3 variant reports the actual mismatch. Within a
    multiview config the training strategy is inspected first: teacher-forcing
    artifacts need a replay/cached-memory runtime that is not implemented.
    """
    backbone_type = transformer_config.get("backbone_type")
    if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
        raise ValueError(
            "Cosmos3MultiviewPipeline requires transformer/config.json "
            f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
        )
    raw = transformer_config.get("multiview")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Cosmos3 multiview transformer config must contain a 'multiview' object."
        )

    strategy = _required_field(raw, "causal_training_strategy")
    if not isinstance(strategy, str):
        raise TypeError("Cosmos3 multiview causal_training_strategy must be a string.")
    if strategy in {"teacher_forcing", "teacher_forcing_dcm"}:
        raise ValueError(
            f"Cosmos3 multiview {strategy} artifacts require replay/cached-memory "
            "inference, which is not supported. Export a bidirectional "
            "causal_training_strategy='none' checkpoint."
        )
    if strategy != "none":
        raise ValueError(
            "Cosmos3 multiview causal_training_strategy must be 'none', "
            f"got {strategy!r}."
        )

    attention_scope = _required_field(raw, "attention_scope")
    if not isinstance(attention_scope, str):
        raise TypeError("Cosmos3 multiview attention_scope must be a string.")
    if attention_scope not in COSMOS3_MULTIVIEW_ATTENTION_SCOPES:
        raise ValueError(
            "Cosmos3 multiview attention_scope must be one of "
            f"{sorted(COSMOS3_MULTIVIEW_ATTENTION_SCOPES)}; got {attention_scope!r}."
        )

    temporal_window = _required_field(raw, "decomposed_temporal_window_seconds")
    if temporal_window is not None:
        if isinstance(temporal_window, bool) or not isinstance(
            temporal_window, int | float
        ):
            raise TypeError(
                "Cosmos3 multiview decomposed_temporal_window_seconds must be null or a number."
            )
        if not math.isfinite(temporal_window) or temporal_window < 0:
            raise ValueError(
                "Cosmos3 multiview decomposed_temporal_window_seconds must be finite "
                "and non-negative."
            )
        temporal_window = float(temporal_window)

    for field_name in (
        "control_attends_sensor",
        "align_temporal_positions_across_views",
        "share_vision_temporal_positions",
    ):
        if not isinstance(_required_field(raw, field_name), bool):
            raise TypeError(f"Cosmos3 multiview {field_name} must be boolean.")
    if not raw["share_vision_temporal_positions"]:
        raise ValueError(
            "Cosmos3 multiview requires share_vision_temporal_positions=true."
        )

    cameras = _required_field(raw, "cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, str) and camera for camera in cameras)
    ):
        raise TypeError(
            "Cosmos3 multiview cameras must be a non-empty list of strings."
        )
    if len(cameras) != len(set(cameras)):
        raise ValueError("Cosmos3 multiview cameras must be unique.")
    max_views = _required_field(raw, "max_views")
    if isinstance(max_views, bool) or not isinstance(max_views, int):
        raise TypeError("Cosmos3 multiview max_views must be an integer.")
    if max_views != len(cameras):
        raise ValueError(
            "Cosmos3 multiview max_views must equal the exported camera list length: "
            f"max_views={max_views}, cameras={len(cameras)}."
        )

    schema_version = raw.get("schema_version")
    if schema_version is not None and (
        isinstance(schema_version, bool)
        or schema_version not in COSMOS3_MULTIVIEW_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"Unsupported Cosmos3 multiview schema_version={schema_version!r}; expected "
            f"null (v1 WSM artifact) or one of {list(COSMOS3_MULTIVIEW_SCHEMA_VERSIONS)}."
        )
    versioned = schema_version is not None
    if versioned and (unknown := set(raw) - COSMOS3_MULTIVIEW_CONTRACT_FIELDS):
        raise ValueError(
            f"Unknown Cosmos3 multiview contract fields {sorted(unknown)} "
            f"(schema_version={schema_version}); this build cannot honour them."
        )
    if not versioned and tuple(cameras) != COSMOS3_MADS_CAMERAS:
        raise ValueError(
            "Unversioned Cosmos3 multiview artifacts require the fixed 11-camera MADS "
            f"order: expected={list(COSMOS3_MADS_CAMERAS)}, got={cameras}."
        )
    separate_captions = False
    variable_view_count = False
    inference_defaults: dict[str, Any] | None = None
    if versioned:
        if not isinstance(_required_field(raw, "variable_view_count"), bool):
            raise TypeError("Cosmos3 multiview variable_view_count must be boolean.")
        separate_captions = _per_view_captions_flag(raw)
        variable_view_count = bool(raw["variable_view_count"])
        inference_defaults = _validate_inference_defaults(
            _required_field(raw, "inference_defaults")
        )
        if separate_captions:
            unlabeled = [c for c in cameras if c not in COSMOS3_MADS_CAMERA_ATTRIBUTES]
            if unlabeled:
                raise ValueError(
                    "Cosmos3 multiview per-camera captions need rig attributes for every "
                    f"exported camera; none are known for {unlabeled}."
                )
    lidar = raw.get("lidar")
    if lidar is not None:
        if not versioned:
            raise ValueError(
                "Cosmos3 joint camera/LiDAR artifacts require versioned "
                "(schema_version >= 2) multiview metadata."
            )
        lidar = validate_lidar_config(lidar)
    camera_patch = transformer_config.get("latent_patch_size", 2)
    if not _positive_int(camera_patch):
        raise TypeError(
            "Cosmos3 transformer latent_patch_size must be a positive integer."
        )
    lidar_patch = _validated_lidar_patch(raw, schema_version, int(camera_patch))
    rig_view_embedding = _validated_rig_view_embedding(raw, schema_version, cameras)
    if (
        schema_version == 3
        and rig_view_embedding is None
        and lidar_patch in (None, (camera_patch, camera_patch))
    ):
        raise ValueError(
            "Cosmos3 multiview schema_version=3 requires rig_view_embedding or a LiDAR "
            "patch that differs from the camera patch; export version 2 otherwise."
        )

    backend = _required_field(raw, "backend")
    if not isinstance(backend, str):
        raise TypeError("Cosmos3 multiview backend must be a string.")
    if backend not in COSMOS3_MULTIVIEW_ATTENTION_BACKENDS:
        raise ValueError(
            "Cosmos3 multiview backend must be one of "
            f"{list(COSMOS3_MULTIVIEW_ATTENTION_BACKENDS)}, got {backend!r}."
        )
    if backend == "maskless":
        if not versioned:
            raise ValueError(
                "Cosmos3 multiview maskless exports carry versioned metadata; "
                f"got schema_version={schema_version!r}. Re-export the checkpoint."
            )
        reason = maskless_unavailable_reason(
            attention_scope=attention_scope,
            decomposed_temporal_window_seconds=temporal_window,
            control_attends_sensor=bool(raw["control_attends_sensor"]),
        )
        if reason is not None:
            raise ValueError(f"Cosmos3 multiview backend 'maskless': {reason}")
    lidar_attends_captions = raw.get("lidar_attends_captions", True)
    if not isinstance(lidar_attends_captions, bool):
        raise TypeError("Cosmos3 multiview lidar_attends_captions must be boolean.")
    system_prompt_variant = raw.get(
        "system_prompt_variant", "wsm_controls" if versioned else "provided_controls"
    )
    if system_prompt_variant not in COSMOS3_SYSTEM_PROMPT_VARIANTS:
        raise ValueError(
            "Cosmos3 multiview system_prompt_variant must be one of "
            f"{list(COSMOS3_SYSTEM_PROMPT_VARIANTS)}, got {system_prompt_variant!r}."
        )

    return Cosmos3MultiviewDeploymentConfig(
        cameras=tuple(cameras),
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=temporal_window,
        control_attends_sensor=bool(raw["control_attends_sensor"]),
        align_temporal_positions_across_views=bool(
            raw["align_temporal_positions_across_views"]
        ),
        share_vision_temporal_positions=True,
        backend=backend,
        schema_version=schema_version,
        per_view_captions=separate_captions,
        variable_view_count=variable_view_count,
        inference_defaults=inference_defaults,
        lidar=lidar,
        lidar_attends_captions=lidar_attends_captions,
        system_prompt_variant=system_prompt_variant,
        lidar_patch_spatial_hw=lidar_patch,
        rig_view_embedding=rig_view_embedding,
    )


@dataclass
class Cosmos3MultiviewConfig(Cosmos3Config):
    """Cosmos3 Multiview-AV: 11-camera WSM-to-RGB transfer in one denoising pass."""

    # Single-task pipeline; the parent's multi-task declaration (TI2V, T2I, V2V)
    # does not apply, so requests resolve to task_type alone.
    supported_task_types: ClassVar[tuple[ModelTaskType, ...] | None] = None

    # Same weights, different GEN cross-attention: the loader instantiates the
    # multiview transformer subclass instead of the checkpoint's class name.
    transformer_class_override: str | None = "Cosmos3MultiviewTransformer"

    # The multiview tokenization stage owns prompt formatting: transfer system
    # prompt, duration/resolution sentences, and the WSM emphasis sentence.
    use_duration_template: bool = True
    use_system_prompt: bool = True

    # Attention kernel for masked exports: None follows the environment override,
    # then the device ("auto": FA4 block-sparse on Blackwell, FlexAttention
    # Triton elsewhere); "triton" and "fa4" pin one. Swapping triton and fa4 is
    # safe for A/B measurement; the masked family and "maskless" are different
    # attention patterns and cannot be swapped.
    multiview_attention_backend: str | None = None

    # Parsed once from transformer/config.json in update_config_from_dict.
    multiview_deployment: Cosmos3MultiviewDeploymentConfig | None = None

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        if self.model_path:
            self.multiview_deployment = parse_multiview_deployment_config(
                _transformer_config(self.model_path)
            )
            if self.distilled_sigmas is not None:
                raise ValueError(
                    "Cosmos3 multiview expects the FlowUniPC schedule; a distilled "
                    "fixed-step scheduler is not supported."
                )
            # Fail on a bad backend name or family at launch, not on the first request.
            backend, source = self._resolve_multiview_backend()
            logger.info(
                "Cosmos3 multiview attention backend: %s (from %s)", backend, source
            )

    def _resolve_multiview_backend(self) -> tuple[str, str]:
        """Explicit config field, then the env override, then the checkpoint.

        Returns ``"auto"`` for a masked export with no pin; the pipeline resolves
        that against the device it runs on.
        """
        if self.multiview_deployment is None:
            raise ValueError(
                "Cosmos3 multiview deployment config has not been resolved; "
                "set model_path first."
            )
        exported = self.multiview_deployment.backend
        backend = self.multiview_attention_backend
        source = "pipeline config multiview_attention_backend"
        if backend is None:
            backend = envs.SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND
            source = MULTIVIEW_BACKEND_ENV_VAR
        if not backend:
            backend = COSMOS3_MULTIVIEW_AUTO_BACKEND
            source = "transformer/config.json multiview.backend"
        allowed = (*COSMOS3_MULTIVIEW_ATTENTION_BACKENDS, COSMOS3_MULTIVIEW_AUTO_BACKEND)
        if backend not in allowed:
            raise ValueError(
                "Cosmos3 multiview attention backend must be one of "
                f"{list(allowed)}, got {backend!r} (from {source})."
            )
        if backend == COSMOS3_MULTIVIEW_AUTO_BACKEND:
            resolved = (
                "maskless" if exported == "maskless" else COSMOS3_MULTIVIEW_AUTO_BACKEND
            )
            return resolved, source
        if (backend == "maskless") != (exported == "maskless"):
            # triton <-> fa4 is a kernel swap; masked <-> maskless changes the
            # attention the weights were trained with, so it is not an override.
            raise ValueError(
                f"Cosmos3 multiview attention backend {backend!r} (from {source}) is not "
                f"the family the checkpoint was exported for ({exported!r}). The masked "
                "(triton/fa4) and maskless backends are different attention patterns; "
                "pick a backend from the checkpoint's family or re-export the checkpoint."
            )
        return backend, source

    def resolved_multiview_backend(self) -> str:
        """``triton``, ``fa4``, ``maskless``, or ``auto`` (masked family, device decides)."""
        return self._resolve_multiview_backend()[0]

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        parallel_degrees = {
            "tp_size": server_args.tp_size,
            "sp_degree": server_args.sp_degree,
            "ulysses_degree": server_args.ulysses_degree,
            "ring_degree": server_args.ring_degree,
        }
        # The attention plan spans the whole camera-major sequence, so sequence and
        # tensor sharding are not supported. CFG parallel is fine: each rank runs
        # one complete branch with its own plan cache and only the weighted
        # velocities are all-reduced.
        for name, degree in parallel_degrees.items():
            if int(degree or 1) > 1:
                raise ValueError(
                    "Cosmos3 multiview does not support sequence or tensor "
                    f"parallelism; {name} must be 1 (use --enable-cfg-parallel on "
                    "2 GPUs instead)."
                )

    def supports_action_endpoint(self) -> bool:
        return False

    def supports_dynamic_batching(self) -> bool:
        return False


def register():
    from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
        Cosmos3MultiviewSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    # Cosmos3 Multiview-AV: 11-camera WSM transfer with maskless cross-camera
    # attention. The hf path is the pre-release folder name, which must sort
    # before the shorter "nvidia/Cosmos3-Nano" in the partial-path match.
    register_configs(
        sampling_param_cls=Cosmos3MultiviewSamplingParams,
        pipeline_config_cls=Cosmos3MultiviewConfig,
        hf_model_paths=["nvidia/Cosmos3-Nano-Transfer-Auto"],
        # Matches the ``Cosmos3MultiviewPipeline`` ``_class_name`` of the checkpoint.
        model_detectors=[lambda hf_id: "cosmos3multiviewpipeline" in hf_id.lower()],
    )
