# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion XL UNet: the native SD UNet exposed as a loadable DiT component."""

from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.stable_diffusion_xl import (
    StableDiffusionXLUNetConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import (
    BasicTransformerBlock,
    ResnetBlock2D,
    StableDiffusionUNet2DConditionModel,
    StableDiffusionUNetConfig,
)


class StableDiffusionXLUNet2DConditionModel(
    CachableDiT, LayerwiseOffloadableModuleMixin
):
    # SDXL checkpoints declare the generic Diffusers UNet class in unet/config.json.
    _aliases = ["UNet2DConditionModel"]
    _fsdp_shard_conditions = [
        lambda n, m: isinstance(m, (BasicTransformerBlock, ResnetBlock2D)),
    ]
    _compile_conditions = [
        lambda n, m: isinstance(m, BasicTransformerBlock),
    ]
    param_names_mapping = StableDiffusionXLUNetConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        StableDiffusionXLUNetConfig().arch_config.reverse_param_names_mapping
    )
    layer_names = ["unet.down_blocks", "unet.up_blocks"]
    layerwise_offload_dit_group_enabled = True

    def __init__(
        self,
        config: StableDiffusionXLUNetConfig,
        hf_config: dict[str, Any],
        quant_config=None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.unet = StableDiffusionUNet2DConditionModel(
            StableDiffusionUNetConfig.from_dict(hf_config)
        )
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_channels_latents = self.config.num_channels_latents

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.unet(
            hidden_states,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


EntryClass = StableDiffusionXLUNet2DConditionModel
