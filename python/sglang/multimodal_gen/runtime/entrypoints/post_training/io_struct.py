"""Request/response data structures for post-training APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel


@dataclass
class UpdateWeightFromDiskReqInput:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    flush_cache: bool = True
    target_modules: list[str] | None = None


@dataclass
class GetWeightsChecksumReqInput:
    """Compute SHA-256 checksum of loaded module weights for verification."""

    module_names: list[str] | None = None


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
