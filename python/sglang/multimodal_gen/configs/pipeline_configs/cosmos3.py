# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline configuration.

A single config serves T2V, I2V, and T2I — same checkpoint, same DiT, same
VAE. ``task_type`` is ``TI2V`` so the request validator accepts an optional
``image_path`` (for I2V) without requiring it. Per-modality dispatch happens
in the stages from ``num_frames`` and ``image_path``; T2I overrides
``data_type`` to ``IMAGE`` in :meth:`SamplingParams._adjust`.
"""

import functools
import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

COSMOS3_EDGE_BACKBONE_TYPE = "cosmos3_edge_nemotron_dense"


@functools.lru_cache(maxsize=None)
def is_edge_checkpoint(model_path: str) -> bool:
    """Whether the checkpoint is the Edge (dense) variant.

    Read from the transformer config rather than the loaded arch so the answer
    is available before the weights are on device (e.g. when resolving sampling
    defaults in the client process).
    """
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        get_diffusers_component_config,
    )

    config = get_diffusers_component_config(
        component_path=os.path.join(model_path, "transformer")
    )
    return (
        config.get("backbone_type") == COSMOS3_EDGE_BACKBONE_TYPE
        or config.get("hidden_act") == "relu2"
    )


@dataclass
class Cosmos3Config(PipelineConfig):
    """Cosmos3 unified pipeline config.

    Cosmos3 reuses the Wan VAE (48 latent channels, 4× temporal, 16× spatial),
    a UniPC rectified-flow sampler, and a ~15B-parameter dual-pathway DiT.
    There is no separate text encoder — text is tokenized and embedded inside
    the transformer.
    """

    # TI2V (text + image → video) so the request validator accepts ``image_path``
    # without requiring it. T2V ignores it; I2V uses it; T2I disregards it.
    task_type: ModelTaskType = ModelTaskType.TI2V

    dit_config: DiTConfig = field(default_factory=Cosmos3VideoConfig)

    # Wan VAE with 48 latent channels (overridden in __post_init__ below).
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Cosmos3 reference inference uses FlowUniPC even when the checkpoint
    # scheduler_config.json advertises a different scheduler class.
    scheduler_class_override: str | None = "FlowUniPCMultistepScheduler"

    # Per-request mode defaults are applied in Cosmos3TimestepPreparationStage.
    flow_shift: float | None = None

    precision: str = "bf16"
    vae_precision: str = "bf16"

    # Pipeline-level (not sampling) knobs.
    max_sequence_length: int = 512
    use_duration_template: bool = True
    use_system_prompt: bool = False

    # Filesystem path to dataset-derived action stats (JSON) for action
    # (de)normalization. Set at server launch rather than per request, since it
    # names a server-side file. ``None`` disables normalization.
    action_stats_path: str | None = None

    def __post_init__(self):
        self.vae_config.arch_config.z_dim = 48
        # Encoder is needed for I2V; T2V/T2I never invoke it.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        # keep WanVAE encode replicated because parallel encode changes I2V
        # conditioning latents when sp_world_size > 1
        self.vae_config.use_parallel_encode = False
        self.vae_config.use_parallel_decode = True

    def adjust_num_frames(self, num_frames: int) -> int:
        """Round ``num_frames`` so ``(n - 1) % 4 == 0`` for the VAE.

        Skips rounding when ``num_frames == 1`` (T2I path) so the single
        frame survives untouched.
        """
        if num_frames == 1:
            return 1
        vae_scale_factor_temporal = 4
        if (num_frames - 1) % vae_scale_factor_temporal != 0:
            num_frames = (
                (num_frames - 1) // vae_scale_factor_temporal
            ) * vae_scale_factor_temporal + 1
        return num_frames
