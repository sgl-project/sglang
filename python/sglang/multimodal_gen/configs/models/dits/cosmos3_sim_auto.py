# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Auto artifact contract.

The imaginaire4 exporter embeds one ``cosmos3_nano_sim_auto`` object (schema 1)
in ``transformer/config.json``: the sensor clock and chunk partition, the KV
ring, the replay visibility policy, the segmented prompt policy, mRoPE
constants, the fixed-step student schedule, the LiDAR tokenizer contract and
the physical rig table. Validation is exact: a missing or unknown field, or a
policy value the rollout does not implement, fails at launch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import msgspec

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MADS_CAMERA_ATTRIBUTES,
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    Cosmos3MultiviewDeploymentConfig,
    _validated_rig_view_embedding,
    validate_lidar_config,
)

COSMOS3_SIM_AUTO_ARTIFACT_KEY = "cosmos3_nano_sim_auto"
COSMOS3_SIM_AUTO_SCHEMA_VERSION = 1

# Reference system prompt identifiers; the tokenization stage maps them to the
# exact prompt strings of the multiview stages.
COSMOS3_SIM_AUTO_SYSTEM_PROMPT_RGB = "av_multiview_transfer"
COSMOS3_SIM_AUTO_SYSTEM_PROMPT_JOINT = "av_joint_camera_lidar_transfer"
COSMOS3_SIM_AUTO_SYSTEM_PROMPT_IDS = frozenset(
    {COSMOS3_SIM_AUTO_SYSTEM_PROMPT_RGB, COSMOS3_SIM_AUTO_SYSTEM_PROMPT_JOINT}
)

_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "checkpoint_iteration",
        "checkpoint_hash",
        "attention_mode",
        "video_temporal_causal",
        "cameras",
        "rgb",
        "lidar",
        "lidar_latent_patch_size_hw",
        "chunk",
        "ring",
        "replay",
        "prompt",
        "mrope",
        "fixed_step_sampler_config",
        "inference_defaults",
        "rig_view_embedding",
    }
)
_RGB_FIELDS = frozenset(
    {
        "latent_channels",
        "latent_patch_size",
        "vae_spatial_compression_factor",
        "temporal_compression_factor",
    }
)
_CHUNK_FIELDS = frozenset({"frames_per_chunk"})
_RING_FIELDS = frozenset({"cache_chunks", "sink_chunks"})
_REPLAY_FIELDS = frozenset(
    {
        "control_visibility",
        "controls_read_strict_past_clean_rgb",
        "clean_pass_causality",
        "attention_scope",
        "cross_view_past_window_seconds",
        "deduplicate_cross_view",
        "lidar_attends_captions",
    }
)
_PROMPT_FIELDS = frozenset(
    {
        "policy",
        "text_sink_tokens",
        "text_position_mode",
        "context_before",
        "context_after",
        "media_text_extent",
        "media_rope_scale",
        "sink_age_cap_seconds",
        "use_first_chunk_only",
        "system_prompt_rgb",
        "system_prompt_joint",
        "fractional_duration",
        "emphasize_control_in_prompt",
        "text_cache_max_len",
    }
)
_MROPE_FIELDS = frozenset(
    {
        "temporal_modality_margin",
        "reset_spatial_ids",
        "base_fps",
        "enable_fps_modulation",
    }
)
_SAMPLER_FIELDS = frozenset({"t_list", "sample_type", "num_train_timesteps"})
_DEFAULTS_FIELDS = frozenset({"resolution", "fps", "guidance", "control_guidance"})

# The replay and prompt policies the rollout implements; any other value is a
# different model, not a knob.
_EXPECTED_REPLAY = {
    "control_visibility": "causal",
    "controls_read_strict_past_clean_rgb": True,
    "clean_pass_causality": "chunk",
    "attention_scope": "decomposed",
    "deduplicate_cross_view": True,
    "lidar_attends_captions": True,
}
_EXPECTED_PROMPT = {
    "policy": "segmented",
    "text_position_mode": "fixed",
    "context_before": 0,
    "context_after": 0,
    "media_text_extent": None,
    "media_rope_scale": 1.0,
    "sink_age_cap_seconds": None,
    "use_first_chunk_only": True,
}
_EXPECTED_MROPE = {"reset_spatial_ids": True, "enable_fps_modulation": True}


def _fields(payload: Any, allowed: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be an object, got "
            f"{type(payload).__name__}."
        )
    missing = sorted(allowed - payload.keys())
    unknown = sorted(payload.keys() - allowed)
    if missing or unknown:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} field set mismatch: missing={missing}, "
            f"unknown={unknown}. Re-export the checkpoint with the matching imaginaire4 exporter."
        )
    return payload


def _int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be an integer, got {value!r}."
        )
    if value < minimum:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be >= {minimum}, got {value}."
        )
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be a boolean, got {value!r}."
        )
    return value


def _float(
    value: Any, name: str, *, minimum: float | None = None, positive: bool = False
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be a finite number, got {value!r}."
        )
    result = float(value)
    if positive and result <= 0:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be positive, got {value!r}."
        )
    if minimum is not None and result < minimum:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be >= {minimum}, got {value!r}."
        )
    return result


def _str(value: Any, name: str, *, choices: frozenset[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name} must be a non-empty string, got {value!r}."
        )
    if choices is not None and value not in choices:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto artifact {name}={value!r} is not one of {sorted(choices)}."
        )
    return value


def _expect(payload: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto {name}.{key}={payload[key]!r} is unsupported; "
                f"this build implements {value!r}."
            )


def _t_list(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto fixed_step_sampler_config.t_list must be a non-empty list."
        )
    t_list = tuple(
        _float(value, "fixed_step_sampler_config.t_list entry") for value in raw
    )
    if (
        t_list[0] != 1.0
        or t_list[-1] <= 0
        or any(b >= a for a, b in zip(t_list[:-1], t_list[1:]))
    ):
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto t_list must start at 1.0 and be strictly decreasing and positive."
        )
    return t_list


def _lidar_patch(raw: Any) -> tuple[int, int]:
    if (
        not isinstance(raw, list | tuple)
        or len(raw) != 2
        or any(
            isinstance(side, bool) or not isinstance(side, int) or side <= 0
            for side in raw
        )
    ):
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto lidar_latent_patch_size_hw must be two positive integers, got {raw!r}."
        )
    return (int(raw[0]), int(raw[1]))


class Cosmos3SimAutoManifest(msgspec.Struct, frozen=True, kw_only=True):
    """Validated schema-1 artifact of a Cosmos3-Nano-Sim-Auto checkpoint."""

    checkpoint_id: str
    checkpoint_iteration: int
    checkpoint_hash: str
    cameras: tuple[str, ...]
    # RGB tokenizer and transformer geometry.
    latent_channels: int
    latent_patch_size: int
    vae_spatial_compression_factor: int
    temporal_compression_factor: int
    # Sensor clock: RGB latents per causal step; LiDAR sweeps follow the same wall clock.
    frames_per_chunk: int
    # KV ring of committed chunks; the first ``sink_chunks`` slots are pinned.
    cache_chunks: int
    sink_chunks: int
    # Cross-view keys are visible when captured at most this long before the query.
    cross_view_past_window_seconds: float
    # Segmented prompting: a per-camera sink prefix plus the camera's caption.
    text_sink_tokens: int
    system_prompt_rgb: str
    system_prompt_joint: str
    fractional_duration: bool
    emphasize_control_in_prompt: bool
    text_cache_max_len: int
    # mRoPE.
    temporal_modality_margin: int
    base_fps: float
    # Fixed-step student schedule.
    t_list: tuple[float, ...]
    sample_type: str
    num_train_timesteps: int
    # Request defaults.
    default_resolution: str
    default_fps: float
    default_guidance: float
    default_control_guidance: float
    # Physical rig table: one trained row per MADS camera plus the final LiDAR row.
    rig_view_embedding: dict[str, Any]
    # Joint checkpoints: the V1.2 LiDAR tokenizer contract and the LiDAR token patch.
    lidar: dict[str, Any] | None = None
    lidar_latent_patch_size_hw: tuple[int, int] | None = None

    @property
    def joint(self) -> bool:
        return self.lidar is not None

    @property
    def lidar_fps(self) -> float | None:
        return None if self.lidar is None else float(self.lidar["fps"])

    @property
    def lidar_seconds_per_frame(self) -> float | None:
        if self.lidar is None:
            return None
        return float(self.lidar["temporal_compression_factor"]) / float(
            self.lidar["fps"]
        )

    def rgb_seconds_per_frame(self, fps: float) -> float:
        return self.temporal_compression_factor / fps

    def rig_camera_ids(self, cameras: tuple[str, ...] | list[str]) -> list[int]:
        """Rig rows of ``cameras`` in request order."""
        table = self.rig_view_embedding["camera_ids"]
        return [int(table[camera]) for camera in cameras]

    def system_prompt_id(self, *, joint: bool) -> str:
        return self.system_prompt_joint if joint else self.system_prompt_rgb

    def describe(self) -> str:
        return (
            f"{self.checkpoint_id}: {len(self.cameras)} cameras, {self.frames_per_chunk} latents per "
            f"chunk, ring {self.cache_chunks} chunks ({self.sink_chunks} sink), cross-view window "
            f"{self.cross_view_past_window_seconds:g} s, {len(self.t_list)}-step {self.sample_type}, "
            f"lidar={'yes' if self.joint else 'no'}"
        )


def _identity_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    cameras = payload["cameras"]
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(
            not isinstance(c, str) or c not in COSMOS3_MADS_CAMERA_ATTRIBUTES
            for c in cameras
        )
        or len(set(cameras)) != len(cameras)
    ):
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto cameras must be unique MADS camera keys."
        )
    if payload["rig_view_embedding"] is None:
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto requires the rig_view_embedding table; re-export the checkpoint."
        )
    return {
        "checkpoint_id": _str(payload["checkpoint_id"], "checkpoint_id"),
        "checkpoint_iteration": _int(
            payload["checkpoint_iteration"], "checkpoint_iteration", minimum=0
        ),
        "checkpoint_hash": _str(payload["checkpoint_hash"], "checkpoint_hash"),
        "cameras": tuple(cameras),
        "rig_view_embedding": _validated_rig_view_embedding(
            {"rig_view_embedding": payload["rig_view_embedding"]}, cameras
        ),
    }


def _sensor_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    rgb = _fields(payload["rgb"], _RGB_FIELDS, "rgb")
    chunk = _fields(payload["chunk"], _CHUNK_FIELDS, "chunk")
    ring = _fields(payload["ring"], _RING_FIELDS, "ring")
    replay = _fields(payload["replay"], _REPLAY_FIELDS, "replay")
    _expect(replay, _EXPECTED_REPLAY, "replay")
    cache_chunks = _int(ring["cache_chunks"], "ring.cache_chunks", minimum=2)
    sink_chunks = _int(ring["sink_chunks"], "ring.sink_chunks", minimum=1)
    if sink_chunks >= cache_chunks:
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto ring must retain at least one non-sink chunk."
        )
    return {
        "latent_channels": _int(rgb["latent_channels"], "rgb.latent_channels"),
        "latent_patch_size": _int(rgb["latent_patch_size"], "rgb.latent_patch_size"),
        "vae_spatial_compression_factor": _int(
            rgb["vae_spatial_compression_factor"], "rgb.vae_spatial_compression_factor"
        ),
        "temporal_compression_factor": _int(
            rgb["temporal_compression_factor"], "rgb.temporal_compression_factor"
        ),
        "frames_per_chunk": _int(chunk["frames_per_chunk"], "chunk.frames_per_chunk"),
        "cache_chunks": cache_chunks,
        "sink_chunks": sink_chunks,
        "cross_view_past_window_seconds": _float(
            replay["cross_view_past_window_seconds"],
            "replay.cross_view_past_window_seconds",
            minimum=0.0,
        ),
    }


def _prompt_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    prompt = _fields(payload["prompt"], _PROMPT_FIELDS, "prompt")
    mrope = _fields(payload["mrope"], _MROPE_FIELDS, "mrope")
    _expect(prompt, _EXPECTED_PROMPT, "prompt")
    _expect(mrope, _EXPECTED_MROPE, "mrope")
    return {
        "text_sink_tokens": _int(
            prompt["text_sink_tokens"], "prompt.text_sink_tokens", minimum=0
        ),
        "system_prompt_rgb": _str(
            prompt["system_prompt_rgb"],
            "prompt.system_prompt_rgb",
            choices=COSMOS3_SIM_AUTO_SYSTEM_PROMPT_IDS,
        ),
        "system_prompt_joint": _str(
            prompt["system_prompt_joint"],
            "prompt.system_prompt_joint",
            choices=COSMOS3_SIM_AUTO_SYSTEM_PROMPT_IDS,
        ),
        "fractional_duration": _bool(
            prompt["fractional_duration"], "prompt.fractional_duration"
        ),
        "emphasize_control_in_prompt": _bool(
            prompt["emphasize_control_in_prompt"], "prompt.emphasize_control_in_prompt"
        ),
        "text_cache_max_len": _int(
            prompt["text_cache_max_len"], "prompt.text_cache_max_len"
        ),
        "temporal_modality_margin": _int(
            mrope["temporal_modality_margin"],
            "mrope.temporal_modality_margin",
            minimum=0,
        ),
        "base_fps": _float(mrope["base_fps"], "mrope.base_fps", positive=True),
    }


def _sampler_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    sampler = _fields(
        payload["fixed_step_sampler_config"],
        _SAMPLER_FIELDS,
        "fixed_step_sampler_config",
    )
    defaults = _fields(
        payload["inference_defaults"], _DEFAULTS_FIELDS, "inference_defaults"
    )
    return {
        "t_list": _t_list(sampler["t_list"]),
        "sample_type": _str(
            sampler["sample_type"],
            "fixed_step_sampler_config.sample_type",
            choices=frozenset({"sde", "ode"}),
        ),
        "num_train_timesteps": _int(
            sampler["num_train_timesteps"],
            "fixed_step_sampler_config.num_train_timesteps",
        ),
        "default_resolution": _str(
            str(defaults["resolution"]),
            "inference_defaults.resolution",
            choices=frozenset({"480", "720"}),
        ),
        "default_fps": _float(defaults["fps"], "inference_defaults.fps", positive=True),
        "default_guidance": _float(
            defaults["guidance"], "inference_defaults.guidance", minimum=0.0
        ),
        "default_control_guidance": _float(
            defaults["control_guidance"],
            "inference_defaults.control_guidance",
            minimum=0.0,
        ),
    }


def _lidar_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    lidar = payload["lidar"]
    lidar_patch = payload["lidar_latent_patch_size_hw"]
    if lidar is None:
        if lidar_patch is not None:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto lidar_latent_patch_size_hw requires a lidar block."
            )
        return {"lidar": None, "lidar_latent_patch_size_hw": None}
    return {
        "lidar": validate_lidar_config(lidar),
        "lidar_latent_patch_size_hw": _lidar_patch(lidar_patch),
    }


def parse_cosmos3_sim_auto_manifest(artifact: Any) -> Cosmos3SimAutoManifest:
    """Validate the ``cosmos3_nano_sim_auto`` object field for field."""
    payload = _fields(artifact, _ARTIFACT_FIELDS, "root")
    schema = _int(payload["schema_version"], "schema_version")
    if schema != COSMOS3_SIM_AUTO_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Cosmos3-Nano-Sim-Auto artifact schema_version={schema}; "
            f"expected {COSMOS3_SIM_AUTO_SCHEMA_VERSION}."
        )
    _str(payload["attention_mode"], "attention_mode", choices=frozenset({"multiview"}))
    if _bool(payload["video_temporal_causal"], "video_temporal_causal") is not True:
        raise ValueError("Cosmos3-Nano-Sim-Auto requires video_temporal_causal=true.")
    return Cosmos3SimAutoManifest(
        **_identity_fields(payload),
        **_sensor_fields(payload),
        **_prompt_fields(payload),
        **_sampler_fields(payload),
        **_lidar_fields(payload),
    )


def load_cosmos3_sim_auto_manifest(
    transformer_config: Mapping[str, Any],
) -> Cosmos3SimAutoManifest:
    """The artifact of a ``transformer/config.json`` mapping, checking the backbone first."""
    backbone_type = transformer_config.get("backbone_type")
    if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto requires transformer/config.json "
            f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
        )
    artifact = transformer_config.get(COSMOS3_SIM_AUTO_ARTIFACT_KEY)
    if artifact is None:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Auto requires transformer/config.json[{COSMOS3_SIM_AUTO_ARTIFACT_KEY!r}]; "
            "export the checkpoint through both imaginaire4 stages with --cosmos3-nano-sim-auto."
        )
    return parse_cosmos3_sim_auto_manifest(artifact)


def multiview_deployment_from_manifest(
    manifest: Cosmos3SimAutoManifest,
) -> Cosmos3MultiviewDeploymentConfig:
    """The multiview stage contract this artifact implies.

    The Sim-Auto pipeline reuses the Multiview-AV input, tokenization, latent
    and decoding stages, which read the deployment config of the bidirectional
    model. Its request defaults are the student's: one fixed schedule, no
    guidance, no control emphasis in the captions.
    """
    return Cosmos3MultiviewDeploymentConfig(
        cameras=manifest.cameras,
        cross_view_past_window_seconds=manifest.cross_view_past_window_seconds,
        rig_view_embedding=manifest.rig_view_embedding,
        inference_defaults={
            "resolution": manifest.default_resolution,
            "fps": manifest.default_fps,
            "num_steps": len(manifest.t_list),
            "guidance": manifest.default_guidance,
            "shift": 1.0,
            "control_guidance": manifest.default_control_guidance,
            "emphasize_control_in_prompt": manifest.emphasize_control_in_prompt,
            "guidance_interval": None,
            "control_guidance_interval": None,
            "normalize_cfg": False,
        },
        lidar=manifest.lidar,
        lidar_latent_patch_size_hw=manifest.lidar_latent_patch_size_hw,
    )
