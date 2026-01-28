# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

from __future__ import annotations

import os
import pprint
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Optional

import PIL.Image
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import (
    TeaCacheParams,
    WanTeaCacheParams,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestTimings
from sglang.multimodal_gen.utils import align_to

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
    # Flux-2
    latent_ids: torch.Tensor | None = None

    # Audio Latents (LTX-2)
    audio_latents: torch.Tensor | None = None
    raw_audio_latent_shape: tuple[int, ...] | None = None

    # Audio Parameters
    fps: float = 24.0
    generate_audio: bool = True

    raw_latent_shape: torch.Tensor | None = None
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
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None

    eta: float = 0.0
    sigmas: list[float] | None = None

    n_tokens: int | None = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_audio_latents: torch.Tensor | None = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: dict[str, Any] = field(default_factory=dict)

    is_warmup: bool = False

    # TeaCache parameters
    teacache_params: TeaCacheParams | WanTeaCacheParams | None = None

    # STA parameters
    STA_param: list | None = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: list[list] | None = None
    mask_search_final_result_neg: list[list] | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0

    # stage logging
    timings: Optional["RequestTimings"] = None

    # results
    output: torch.Tensor | None = None

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

        return (
            os.path.join(self.output_path, output_file_name)
            if output_file_name
            else None
        )

    def set_as_warmup(self):
        self.is_warmup = True
        self.extra["cache_dit_num_inference_steps"] = self.num_inference_steps
        self.num_inference_steps = 1

    def validate(self):
        """Initialize dependent fields after dataclass initialization."""
        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0 and self.negative_prompt is not None:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []
        if self.guidance_scale_2 is None:
            self.guidance_scale_2 = self.guidance_scale

        self.timings = RequestTimings(request_id=self.request_id)

        if self.is_warmup:
            self.set_as_warmup()

    def adjust_size(self, server_args: ServerArgs):
        pass

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)

    def log(self, server_args: ServerArgs):
        if self.is_warmup:
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

        # Log sampling parameters
        debug_str = f"""Sampling params:
                       width: {target_width}
                      height: {target_height}
                  num_frames: {self.num_frames}
                      prompt: {self.prompt}
                  neg_prompt: {self.negative_prompt}
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
        if not self.suppress_logs:
            logger.info(debug_str)


@dataclass
class OutputBatch:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    audio_sample_rate: int | None = None
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_decoded: list[torch.Tensor] | None = None
    error: str | None = None

    # logged timings info, directly from Req.timings
    timings: Optional["RequestTimings"] = None

    # For ComfyUI integration: noise prediction from denoising stage
    noise_pred: torch.Tensor | None = None
    peak_memory_mb: float = 0.0
