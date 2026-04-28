# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

from __future__ import annotations

import logging
import os
import pprint
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Optional, Union

import PIL.Image
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    _sanitize_for_logging,
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics
from sglang.multimodal_gen.utils import align_to
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext

logger = init_logger(__name__)

SAMPLING_PARAMS_FIELDS = {f.name for f in fields(SamplingParams)}


@dataclass(init=False)
class Req:
    """
    Complete state passed through the pipeline execution.

    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.

    [IMPORTANT] Fields that overlap with SamplingParams are automatically delegated to the
    sampling_params member via __getattr__ and __setattr__.
    """

    sampling_params: SamplingParams | None = None

    generator: torch.Generator | list[torch.Generator] | None = None

    # Image encoder hidden states
    image_embeds: list[torch.Tensor] = field(default_factory=list)

    original_condition_image_size: tuple[int, int] = None
    condition_image: torch.Tensor | PIL.Image.Image | None = None
    vae_image: torch.Tensor | PIL.Image.Image | None = None
    pixel_values: torch.Tensor | PIL.Image.Image | None = None
    preprocessed_image: torch.Tensor | None = None

    output_file_ext: str | None = None
    # Primary encoder embeddings
    prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None
    clip_embedding_pos: list[torch.Tensor] | None = None
    clip_embedding_neg: list[torch.Tensor] | None = None

    pooled_embeds: list[torch.Tensor] = field(default_factory=list)
    neg_pooled_embeds: list[torch.Tensor] = field(default_factory=list)

    # Additional text-related parameters
    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False

    seeds: list[int] | None = None

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Audio Embeddings (LTX-2)
    audio_prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    negative_audio_prompt_embeds: list[torch.Tensor] | torch.Tensor = field(
        default_factory=list
    )

    # Latent tensors
    latents: torch.Tensor | None = None
    y: torch.Tensor | None = None
    # Flux-2
    latent_ids: torch.Tensor | None = None

    # Audio Latents
    audio_latents: torch.Tensor | None = None
    audio_noise: torch.Tensor | None = None
    raw_audio_latent_shape: tuple[int, ...] | None = None
    did_sp_shard_audio_latents: bool = False
    sp_audio_start_frame: int = 0
    sp_audio_orig_num_frames: int = 0

    # Audio Parameters
    generate_audio: bool = True

    raw_latent_shape: torch.Tensor | None = None
    did_sp_shard_latents: bool = False
    sp_video_start_frame: int = 0
    noise_pred: torch.Tensor | None = None
    # vae-encoded condition image
    image_latent: torch.Tensor | list[torch.Tensor] | None = None
    condition_image_latent_ids: torch.Tensor | list[torch.Tensor] | None = None
    vae_image_sizes: list[tuple[int, int]] | None = None

    # Latent dimensions
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None

    # Timesteps
    timesteps: torch.Tensor | None = None
    paired_timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None

    # request-local scheduler used by timestep/denoising stages.
    # This is optional because the normal worker path executes one request at a time, so it can
    # point at the stage-local scheduler and preserve warmup/device caches.
    # Request-local cloned schedulers are only needed when a request can run
    # concurrently with another request or outlive the stage-local scheduler
    # state, such as grouped execution or disaggregation.
    scheduler: Any | None = None

    eta: float = 0.0
    sigmas: list[float] | None = None

    n_tokens: int | None = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    trajectory_timesteps: torch.Tensor | None = None
    trajectory_latents: torch.Tensor | None = None
    rollout_trajectory_data: RolloutTrajectoryData | None = None
    trajectory_audio_latents: torch.Tensor | None = None

    # Extra parameters that might be needed by specific pipeline implementations (e.g., LTX2.3 DenoisingAVStage)
    extra: dict[str, Any] = field(default_factory=dict)

    is_warmup: bool = False

    # STA parameters
    STA_param: list | None = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: list[list] | None = None
    mask_search_final_result_neg: list[list] | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0

    # stage logging
    metrics: Optional["RequestMetrics"] = None

    # tracing context (TraceReqContext or TraceNullContext)
    trace_ctx: Union[TraceReqContext, TraceNullContext] = field(
        default_factory=TraceNullContext
    )

    # results
    output: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    audio_sample_rate: int | None = None

    def __init__(self, **kwargs):
        # Initialize dataclass fields
        for name, field in self.__class__.__dataclass_fields__.items():
            if name in kwargs:
                object.__setattr__(self, name, kwargs.pop(name))
            elif field.default is not MISSING:
                object.__setattr__(self, name, field.default)
            elif field.default_factory is not MISSING:
                object.__setattr__(self, name, field.default_factory())

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.validate()

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to sampling_params if not found in Req.
        This is only called when the attribute is not found in the instance.
        """
        if name == "sampling_params":
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        sampling_params = object.__getattribute__(self, "sampling_params")
        if sampling_params is not None and hasattr(sampling_params, name):
            return getattr(sampling_params, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Smart attribute setting:
        1. If field exists in Req, set it in Req
        2. Else if field exists in sampling_params, set it in sampling_params
        3. Else set it in Req (for dynamic attributes)
        """
        if name == "sampling_params":
            object.__setattr__(self, name, value)
            return

        if name in self.__class__.__dataclass_fields__:
            object.__setattr__(self, name, value)
            return

        try:
            sampling_params = object.__getattribute__(self, "sampling_params")
        except AttributeError:
            sampling_params = None

        if sampling_params is not None and hasattr(sampling_params, name):
            setattr(sampling_params, name, value)
            return

        if sampling_params is None and name in SAMPLING_PARAMS_FIELDS:
            new_sp = SamplingParams()
            object.__setattr__(self, "sampling_params", new_sp)
            setattr(new_sp, name, value)
            return

        object.__setattr__(self, name, value)

    @property
    def batch_size(self):
        # Determine batch size
        if isinstance(self.prompt, list):
            batch_size = len(self.prompt)
        elif self.prompt is not None:
            batch_size = 1
        else:
            batch_size = self.prompt_embeds[0].shape[0]

        # Adjust batch size for number of videos per prompt
        batch_size *= self.num_outputs_per_prompt
        return batch_size

    def output_file_path(self, num_outputs=1, output_idx=None):
        output_file_name = self.output_file_name
        if num_outputs > 1 and output_file_name:
            base, ext = os.path.splitext(output_file_name)
            output_file_name = f"{base}_{output_idx}{ext}"

        if self.output_path is None or not output_file_name:
            return None
        return os.path.join(self.output_path, output_file_name)

    def set_as_warmup(self, warmup_steps: int = 1):
        self.is_warmup = True
        self.save_output = False
        self.suppress_logs = True
        self.extra["cache_dit_num_inference_steps"] = self.num_inference_steps
        self.num_inference_steps = warmup_steps

    def copy_as_warmup(self, warmup_steps: int = 1) -> "Req":
        req = deepcopy(self)
        req.set_as_warmup(warmup_steps)
        return req

    def validate(self):
        """Initialize dependent fields after dataclass initialization."""
        # Prefer true_cfg_scale when it is explicitly provided.
        cfg_scale = (
            self.true_cfg_scale
            if self.true_cfg_scale is not None
            else self.guidance_scale
        )
        if cfg_scale > 1.0 and self.negative_prompt is not None:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []
        if self.guidance_scale_2 is None:
            self.guidance_scale_2 = self.guidance_scale

        self.metrics = RequestMetrics(request_id=self.request_id)

    def adjust_size(self, server_args: ServerArgs):
        pass

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)

    def log(self, server_args: ServerArgs):
        if self.is_warmup or self.suppress_logs:
            return
        # TODO: in some cases (e.g., TI2I), height and weight might be undecided at this moment
        if self.height:
            target_height = align_to(self.height, 16)
        else:
            target_height = -1
        if self.width:
            target_width = align_to(self.width, 16)
        else:
            target_width = -1

        if logger.isEnabledFor(logging.DEBUG):
            display_prompt = self.prompt
            display_neg_prompt = self.negative_prompt
        else:
            display_prompt = _sanitize_for_logging(self.prompt, key_hint="prompt")
            display_neg_prompt = _sanitize_for_logging(
                self.negative_prompt, key_hint="negative_prompt"
            )

        debug_str = f"""Sampling params:
                       width: {target_width}
                      height: {target_height}
                  num_frames: {self.num_frames}
                         fps: {self.fps}
                      prompt: {display_prompt}
                  neg_prompt: {display_neg_prompt}
                        seed: {self.seed}
                 infer_steps: {self.num_inference_steps}
      num_outputs_per_prompt: {self.num_outputs_per_prompt}
              guidance_scale: {self.guidance_scale}
     embedded_guidance_scale: {server_args.pipeline_config.embedded_cfg_scale}
                    n_tokens: {self.n_tokens}
                  flow_shift: {server_args.pipeline_config.flow_shift}
                  image_path: {self.image_path}
                 save_output: {self.save_output}
            output_file_path: {self.output_file_path()}
        """  # type: ignore[attr-defined]
        logger.info(debug_str)


@dataclass
class OutputBatch:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    audio_sample_rate: int | None = None
    trajectory_timesteps: torch.Tensor | None = None
    trajectory_latents: torch.Tensor | None = None
    rollout_trajectory_data: RolloutTrajectoryData | None = None
    trajectory_decoded: list[torch.Tensor] | None = None
    error: str | None = None
    output_file_paths: list[str] | None = None

    # logged metrics info, directly from Req.timings
    metrics: Optional["RequestMetrics"] = None
    metrics_list: Optional[list[Optional["RequestMetrics"]]] = None

    # For ComfyUI integration: noise prediction from denoising stage
    noise_pred: torch.Tensor | None = None
    peak_memory_mb: float = 0.0
