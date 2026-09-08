"""Request/response data structures for post-training APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import msgspec
from pydantic import BaseModel


@dataclass
class UpdateWeightFromDiskReqInput:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    flush_cache: bool = True
    target_modules: list[str] | None = None


@dataclass
class UpdateWeightFromTensorReqInput:
    """Request to update model weights from tensor payloads for diffusion models."""

    serialized_named_tensors: list[str | bytes]
    # Physical GPU UUID each payload was exported from, one per payload.
    payload_gpu_uuids: list[str] | None = None
    load_format: str | None = None
    target_modules: list[str] | None = None
    weight_update_mode: str | None = None
    lora_alpha: int | None = None
    lora_rank: int | None = None


@dataclass
class UpdateWeightFromTensorCheckerReqInput:
    """Request to verify live module weights against expected SHA-256 values."""

    target_module: str
    expected_named_tensors_sha256: dict[str, str]


@dataclass
class GetWeightsChecksumReqInput:
    """Compute SHA-256 checksum of loaded module weights for verification."""

    module_names: list[str] | None = None


@dataclass
class ReleaseMemoryOccupationReqInput:
    """Request to release (sleep) GPU memory occupation for the diffusion engine."""

    pass


@dataclass
class ResumeMemoryOccupationReqInput:
    """Request to resume (wake) GPU memory occupation for the diffusion engine."""

    pass


class RolloutRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    generator_device: str = "cuda"

    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    num_outputs_per_prompt: Optional[int] = None

    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = None

    # video-specific (ignored by image pipelines)
    num_frames: Optional[int] = None
    fps: Optional[int] = None

    rollout: bool = True
    # "uint8": quantise the video engine-side; None: ship unchanged
    rollout_video_dtype: Optional[str] = None
    rollout_sde_type: str = "sde"
    rollout_noise_level: float = 0.7
    rollout_log_prob_no_const: bool = False
    rollout_debug_mode: bool = True

    rollout_return_denoising_env: bool = False
    rollout_return_dit_trajectory: bool = False

    # 0-indexed denoising-loop step filters. None = all steps.
    rollout_sde_step_indices: Optional[list[int]] = None
    rollout_return_step_indices: Optional[list[int]] = None

    image_path: Optional[list[str]] = None

    # suppress verbose per-request logging (also gates peak_memory_mb collection)
    suppress_logs: bool = False

    extra_sampling_params: Optional[dict[str, Any]] = None


class RolloutResponse(BaseModel):
    request_id: str
    prompt: str
    seed: int

    generated_output: Any = None

    rollout_log_probs: Optional[dict[str, Any]] = None
    rollout_debug_tensors: Optional[dict[str, Any]] = None
    denoising_env: Optional[dict[str, Any]] = None
    dit_trajectory: Optional[dict[str, Any]] = None

    inference_time_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None


class InitWeightsUpdateGroupReqInput(msgspec.Struct, frozen=True):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str = "nccl"

    def __post_init__(self):
        if not self.master_address or not 0 < self.master_port < 65536:
            raise ValueError("A valid master address and port are required")
        if not self.group_name or not 0 < self.rank_offset < self.world_size:
            raise ValueError("A nonempty group name and valid rank offset are required")


class DestroyWeightsUpdateGroupReqInput(msgspec.Struct, frozen=True):
    group_name: str

    def __post_init__(self):
        if not self.group_name:
            raise ValueError("group_name must be nonempty")


class UpdateWeightsFromDistributedReqInput(msgspec.Struct, frozen=True):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str
    target_modules: list[str]
    weight_update_mode: str | None = None
    lora_alpha: int | None = None
    lora_rank: int | None = None

    def __post_init__(self):
        import torch

        if (
            not self.names
            or len(self.names) != len(self.dtypes)
            or len(self.names) != len(self.shapes)
        ):
            raise ValueError(
                "names, dtypes and shapes must have the same nonzero length"
            )
        if not all(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Tensor names must be nonempty and unique")
        if any(
            not isinstance(torch.__dict__.get(dtype), torch.dtype)
            for dtype in self.dtypes
        ):
            raise ValueError("Unsupported tensor dtype")
        if any(size < 0 for shape in self.shapes for size in shape):
            raise ValueError("Tensor dimensions must be nonnegative")
        if (
            not self.group_name
            or len(self.target_modules) != 1
            or not self.target_modules[0]
        ):
            raise ValueError("A group name and exactly one target module are required")
        if self.weight_update_mode not in (None, "lora_merge"):
            raise ValueError("Unsupported weight update mode")
        if self.lora_rank is not None and self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
