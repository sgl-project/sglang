# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV pipeline configuration.

The checkpoint ships the regular Cosmos3 Nano weight layout; no multiview-only
weight keys exist. ``transformer/config.json`` marks it with
``backbone_type: cosmos3_multiview`` and a ``multiview`` object that fixes the
camera list and the sparse-attention visibility rules. Multiview behavior is
entirely request-, VAE-, position-, and attention-mask-side, so this config
reuses ``Cosmos3Config`` (Wan VAE, Qwen2 tokenizer, FlowUniPC) and swaps in the
transformer subclass that installs the sparse cross-camera attention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import msgspec

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    _transformer_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MULTIVIEW_BACKEND_ENV_VAR = "SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND"

COSMOS3_MULTIVIEW_BACKBONE_TYPE = "cosmos3_multiview"
COSMOS3_MULTIVIEW_ATTENTION_SCOPES = ("all_views", "same_view", "decomposed")
COSMOS3_MULTIVIEW_ATTENTION_BACKENDS = ("triton", "fa4")

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


class Cosmos3MultiviewDeploymentConfig(msgspec.Struct, frozen=True):
    """The validated ``multiview`` block of ``transformer/config.json``."""

    cameras: tuple[str, ...]
    attention_scope: str
    decomposed_temporal_window_seconds: float | None
    control_attends_sensor: bool
    align_temporal_positions_across_views: bool
    share_vision_temporal_positions: bool
    backend: str

    @property
    def num_views(self) -> int:
        return len(self.cameras)


def _required_field(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(
            f"Cosmos3 multiview transformer config requires field {name!r}."
        )
    return config[name]


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
    if tuple(cameras) != COSMOS3_MADS_CAMERAS:
        raise ValueError(
            "Cosmos3 Multiview-AV v1 requires the fixed 11-camera MADS order: "
            f"expected={list(COSMOS3_MADS_CAMERAS)}, got={cameras}."
        )
    max_views = _required_field(raw, "max_views")
    if isinstance(max_views, bool) or not isinstance(max_views, int):
        raise TypeError("Cosmos3 multiview max_views must be an integer.")
    if max_views != len(cameras):
        raise ValueError(
            "Cosmos3 multiview max_views must equal the exported camera list length: "
            f"max_views={max_views}, cameras={len(cameras)}."
        )

    backend = _required_field(raw, "backend")
    if not isinstance(backend, str):
        raise TypeError("Cosmos3 multiview backend must be a string.")
    if backend not in COSMOS3_MULTIVIEW_ATTENTION_BACKENDS:
        raise ValueError(
            "Cosmos3 multiview backend must be one of "
            f"{list(COSMOS3_MULTIVIEW_ATTENTION_BACKENDS)}, got {backend!r}."
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
    )


@dataclass
class Cosmos3MultiviewConfig(Cosmos3Config):
    """Cosmos3 Multiview-AV: 11-camera WSM-to-RGB transfer in one denoising pass."""

    # Same weights, different GEN cross-attention: the loader instantiates the
    # multiview transformer subclass instead of the checkpoint's class name.
    transformer_class_override: str | None = "Cosmos3MultiviewTransformer"

    # The multiview tokenization stage owns prompt formatting: transfer system
    # prompt, duration/resolution sentences, and the WSM emphasis sentence.
    use_duration_template: bool = True
    use_system_prompt: bool = True

    # Sparse attention kernel. ``None`` follows the checkpoint's
    # ``multiview.backend``; ``"fa4"`` needs an SM90/SM100 GPU, CUDA 13, and flash-attn-4.
    # Both backends project the same visibility predicate, so overriding is
    # safe for A/B measurement without editing the checkpoint.
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
            # Fail on a bad backend name at launch, not on the first request.
            backend, source = self._resolve_multiview_backend()
            logger.info(
                "Cosmos3 multiview attention backend: %s (from %s)", backend, source
            )

    def _resolve_multiview_backend(self) -> tuple[str, str]:
        """Explicit config field, then the env override, then the checkpoint."""
        backend = self.multiview_attention_backend
        source = "pipeline config multiview_attention_backend"
        if backend is None:
            backend = envs.SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND
            source = MULTIVIEW_BACKEND_ENV_VAR
        if not backend:
            if self.multiview_deployment is None:
                raise ValueError(
                    "Cosmos3 multiview deployment config has not been resolved; "
                    "set model_path first."
                )
            backend = self.multiview_deployment.backend
            source = "transformer/config.json multiview.backend"
        if backend not in COSMOS3_MULTIVIEW_ATTENTION_BACKENDS:
            raise ValueError(
                "Cosmos3 multiview attention backend must be one of "
                f"{list(COSMOS3_MULTIVIEW_ATTENTION_BACKENDS)}, got {backend!r} "
                f"(from {source})."
            )
        return backend, source

    def resolved_multiview_backend(self) -> str:
        return self._resolve_multiview_backend()[0]

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        parallel_degrees = {
            "tp_size": server_args.tp_size,
            "sp_degree": server_args.sp_degree,
            "ulysses_degree": server_args.ulysses_degree,
            "ring_degree": server_args.ring_degree,
        }
        # The sparse mask spans the whole camera-major sequence, so sequence and
        # tensor sharding are not supported. CFG parallel is fine: each rank runs
        # one complete branch with its own mask cache and only the weighted
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
