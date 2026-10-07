# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Transfer-Auto (Multiview-AV) pipeline configuration.

``transformer/config.json`` marks the checkpoint with
``backbone_type: cosmos3_multiview`` and a ``multiview`` object that carries the
deployment contract vLLM-Omni's ``multiview_config.py`` defines (HF da7c96b,
Oct 7 2026): the camera list, the past-only cross-view window, the rig view
embedding table, the request defaults, and for joint checkpoints the LiDAR
tokenizer metadata and the LiDAR latent patch. Everything older exports spelled
out is implied by the architecture: masked block-sparse attention with the
kernel left to the runtime, decomposed scope, controls reading their own view's
targets, LiDAR reading every caption, one caption per camera, camera subsets.
This config reuses ``Cosmos3Config`` (Wan VAE, Qwen2 tokenizer, FlowUniPC) and
swaps in the transformer subclass that installs the multiview attention.
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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COSMOS3_MULTIVIEW_BACKBONE_TYPE = "cosmos3_multiview"
# Kernels behind the one attention pattern this model has: FlexAttention's Triton
# kernels or FlashAttention-4 block-sparse CuTe. Both count every permitted key
# once; the choice is a speed knob.
COSMOS3_MULTIVIEW_ATTENTION_BACKENDS = ("triton", "fa4")
# Left to the device: FA4 where its kernels run (Hopper, Blackwell), Triton elsewhere.
COSMOS3_MULTIVIEW_AUTO_BACKEND = "auto"
MULTIVIEW_BACKEND_ENV_VAR = "SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND"
# The deployment contract, field for field what vLLM-Omni's multiview_config.py
# reads. Anything else fails loudly: a new field means a new architecture.
COSMOS3_MULTIVIEW_CONTRACT_FIELDS = frozenset(
    {
        "cameras",
        "cross_view_past_window_seconds",
        "rig_view_embedding",
        "lidar_latent_patch_size_hw",
        "lidar",
        "inference_defaults",
    }
)

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
        "normalize_cfg",
    }
)
# Written by the Sep/Oct-1 exporters, dropped by the Oct-7 contract (sglang never
# read sigma_max; negative_metadata_mode is a request field).
COSMOS3_MULTIVIEW_OPTIONAL_INFERENCE_DEFAULT_KEYS = frozenset(
    {"sigma_max", "negative_metadata_mode"}
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
    """The validated ``multiview`` block of ``transformer/config.json``."""

    cameras: tuple[str, ...]
    #: A key of another view is visible when captured at most this many seconds
    #: before the query (same instant included); 0 keeps cross-view attention to
    #: the same instant.
    cross_view_past_window_seconds: float
    #: Physical rig identity table: ``camera_ids`` maps every exported camera to
    #: its trained embedding row, ``lidar_id`` is the final row.
    rig_view_embedding: dict[str, Any]
    inference_defaults: dict[str, Any]
    lidar: dict[str, Any] | None = None
    #: LiDAR token patch ``(height, width)`` in latent cells; present with ``lidar``.
    lidar_latent_patch_size_hw: tuple[int, int] | None = None

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    def rig_view_ids(self, camera_keys: Sequence[str]) -> list[int]:
        """Trained embedding rows for the request's cameras, in request order.

        Subsets and reordered views keep each camera's physical id.
        """
        table = self.rig_view_embedding["camera_ids"]
        unknown = [key for key in camera_keys if key not in table]
        if unknown:
            raise ValueError(
                f"Cosmos3 multiview rig view embedding has no row for cameras {unknown}."
            )
        return [int(table[key]) for key in camera_keys]

    @property
    def supports_lidar(self) -> bool:
        return self.lidar is not None

    def inference_default(self, name: str, fallback: Any) -> Any:
        """A request default the export declares, else ``fallback``."""
        value = self.inference_defaults.get(name)
        return fallback if value is None else value


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
    """Validate the exported V1.2 LiDAR tokenizer contract.

    ``version`` was written by the Sep/Oct-1 exporters and dropped by the Oct-7
    contract; when present it must name the V1.2 tokenizer.
    """
    required = {
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
    if (
        "version" in config
        and str(config["version"]) != COSMOS3_LIDAR_TOKENIZER_VERSION
    ):
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
    numeric = ["fps", "num_steps", "guidance", "shift", "control_guidance"]
    if "sigma_max" in raw:
        numeric.append("sigma_max")
    for name in numeric:
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


def _validated_lidar_patch(raw: Mapping[str, Any]) -> tuple[int, int] | None:
    """The LiDAR stream's ``(height, width)`` patch; required with, and only with, ``lidar``."""
    name = "lidar_latent_patch_size_hw"
    if raw.get("lidar") is None:
        if name in raw:
            raise ValueError(f"Cosmos3 multiview {name} requires a lidar block.")
        return None
    patch = _required_field(raw, name)
    if (
        not isinstance(patch, list | tuple)
        or len(patch) != 2
        or not all(_positive_int(side) for side in patch)
    ):
        raise ValueError(
            f"Cosmos3 multiview {name} must be two positive integers, got {patch!r}."
        )
    return (int(patch[0]), int(patch[1]))


def _validated_rig_view_embedding(
    raw: Mapping[str, Any], cameras: Sequence[str]
) -> dict[str, Any]:
    """Validate the physical rig-id table: one row per exported camera plus a final LiDAR row."""
    rig = _required_field(raw, "rig_view_embedding")
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

    The backbone is inspected first so selecting this pipeline for another
    Cosmos3 variant reports the actual mismatch.
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
    if unknown := set(raw) - COSMOS3_MULTIVIEW_CONTRACT_FIELDS:
        raise ValueError(
            f"Unknown Cosmos3 multiview contract fields {sorted(unknown)}; this build "
            "serves the Oct-7 2026 contract only (HF da7c96b). Re-export the checkpoint."
        )

    window = _required_field(raw, "cross_view_past_window_seconds")
    if isinstance(window, bool) or not isinstance(window, int | float):
        raise TypeError(
            "Cosmos3 multiview cross_view_past_window_seconds must be a number."
        )
    if not math.isfinite(window) or window < 0:
        raise ValueError(
            "Cosmos3 multiview cross_view_past_window_seconds must be finite and "
            "non-negative."
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
    unlabeled = [c for c in cameras if c not in COSMOS3_MADS_CAMERA_ATTRIBUTES]
    if unlabeled:
        raise ValueError(
            "Cosmos3 multiview per-camera captions need rig attributes for every "
            f"exported camera; none are known for {unlabeled}."
        )

    rig_view_embedding = _validated_rig_view_embedding(raw, cameras)
    inference_defaults = _validate_inference_defaults(
        _required_field(raw, "inference_defaults")
    )
    lidar = raw.get("lidar")
    if lidar is not None:
        lidar = validate_lidar_config(lidar)
    lidar_patch = _validated_lidar_patch(raw)

    return Cosmos3MultiviewDeploymentConfig(
        cameras=tuple(cameras),
        cross_view_past_window_seconds=float(window),
        rig_view_embedding=rig_view_embedding,
        inference_defaults=inference_defaults,
        lidar=lidar,
        lidar_latent_patch_size_hw=lidar_patch,
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

    # Kernel behind the multiview attention: "triton", "fa4", or None for the
    # env override / device default. Both kernels compute the same attention, so
    # this is safe for A/B measurement.
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
        """Explicit config field, then the env override, then ``auto``.

        ``auto`` is resolved by the pipeline against the device it runs on.
        """
        if self.multiview_deployment is None:
            raise ValueError(
                "Cosmos3 multiview deployment config has not been resolved; "
                "set model_path first."
            )
        backend = self.multiview_attention_backend
        source = "pipeline config multiview_attention_backend"
        if backend is None:
            backend = envs.SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND
            source = MULTIVIEW_BACKEND_ENV_VAR
        if not backend:
            backend = COSMOS3_MULTIVIEW_AUTO_BACKEND
            source = "device default"
        allowed = (
            *COSMOS3_MULTIVIEW_ATTENTION_BACKENDS,
            COSMOS3_MULTIVIEW_AUTO_BACKEND,
        )
        if backend not in allowed:
            raise ValueError(
                "Cosmos3 multiview attention backend must be one of "
                f"{list(allowed)}, got {backend!r} (from {source})."
            )
        return backend, source

    def resolved_multiview_backend(self) -> str:
        """``triton``, ``fa4``, or ``auto`` (the device decides)."""
        return self._resolve_multiview_backend()[0]

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        # Ulysses shards the GEN stream over ranks and trades sequence for heads
        # around every attention call; the mask spans the whole camera-major
        # sequence, so ring attention and tensor parallelism are not supported.
        # CFG parallel composes with it: each branch runs on its own SP group.
        for name in ("tp_size", "ring_degree"):
            if int(getattr(server_args, name) or 1) > 1:
                raise ValueError(
                    f"Cosmos3 multiview does not support {name} > 1; use "
                    "--ulysses-degree for sequence parallelism and "
                    "--enable-cfg-parallel for the two CFG branches."
                )
        ulysses = int(server_args.ulysses_degree or 1)
        sp_degree = int(server_args.sp_degree or 1)
        if sp_degree != ulysses:
            raise ValueError(
                "Cosmos3 multiview sequence parallelism is Ulysses only: sp_degree "
                f"({sp_degree}) must equal ulysses_degree ({ulysses})."
            )
        if ulysses > 1 and ulysses not in (2, 4, 8):
            raise ValueError(
                "Cosmos3 multiview --ulysses-degree must divide the 8 key/value "
                f"heads (2, 4 or 8), got {ulysses}."
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
