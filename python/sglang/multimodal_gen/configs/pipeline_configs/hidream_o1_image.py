# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)

# Key under which the before-denoising stage stashes the per-branch backbone
# inputs, since HiDream-O1 conditions on token ids rather than text embeddings.
HIDREAM_O1_COND_KEY = "hidream_o1_pos_cond"
HIDREAM_O1_NEG_COND_KEY = "hidream_o1_neg_cond"


@dataclass
class HiDreamO1ImagePipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_config: DiTConfig = field(default_factory=HiDreamO1ImageDitConfig)
    dit_precision: str = "bf16"

    # No VAE and no separate text encoder: the backbone embeds the prompt and
    # the "latents" are raw 32x32 pixel patches.
    vae_tiling: bool = False
    vae_sp: bool = False

    # The DiT takes no distilled-guidance embedding; classifier-free guidance is
    # the only guidance mechanism.
    should_use_guidance: bool = False

    # Noise is drawn on the host so a seed reproduces the same image on any
    # accelerator, matching the reference implementation.
    generator_device: str = "cpu"

    def validate_server_args(self, server_args: Any) -> None:
        if server_args.sp_degree > 1:
            raise ValueError(
                "HiDream-O1-Image mixes causal text spans with a bidirectional "
                "image span, which requires a dense [B, 1, S, S] attention mask "
                "that cannot be sharded along the sequence. Run with "
                "--ulysses-degree 1 --ring-degree 1 and use --tp-size or "
                "--cfg-parallel-size for multi-GPU instead."
            )
        super().validate_server_args(server_args)

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del device, rotary_emb, dtype
        return batch.extra[HIDREAM_O1_COND_KEY]

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del device, rotary_emb, dtype
        return batch.extra[HIDREAM_O1_NEG_COND_KEY]

    def post_denoising_loop(self, latents: torch.Tensor, batch) -> torch.Tensor:
        # Patch tokens are never padded, so the base unpad would be a no-op that
        # only risks truncating against a stale raw_latent_shape.
        del batch
        return latents
