# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import field

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    EncoderConfig,
    ImageEncoderConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_tp_group,
    get_world_group,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def get_folding_tp_group(config: EncoderConfig):
    """Group an encoder should tensor-parallel over.

    When ``config.parallel_folding`` is set the encoder is folded over a larger
    group than its own TP (the idle DiT replica during the encoding stage);
    otherwise it uses the default TP group. Shared by every text/image encoder
    so the choice lives in one place. See ServerArgs.adjust_pipeline_config().
    """
    if config.parallel_folding:
        if config.parallel_folding_mode == "sp":
            return get_sp_group()
        elif config.parallel_folding_mode == "ulysses":
            return get_sp_group().ulysses_group
        elif config.parallel_folding_mode == "ring":
            return get_sp_group().ring_group
        elif config.parallel_folding_mode == "world":
            # The whole single-replica DiT (all GPUs), regardless of tp/sp/cfg.
            return get_world_group()
    return get_tp_group()


class TextEncoder(nn.Module, ABC, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = [
        "layers",
        "encoder.block",
        "text_model.encoder.layers",
        "model.language_model.layers",
    ]
    _fsdp_shard_conditions: list = field(default_factory=lambda: [])
    _stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)
    _supported_attention_backends: set[AttentionBackendEnum] = (
        TextEncoderConfig()._supported_attention_backends
    )

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._fsdp_shard_conditions = config.arch_config._fsdp_shard_conditions
        self._stacked_params_mapping = config.arch_config.stacked_params_mapping
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        pass

    @property
    def supported_attention_backends(self) -> set[AttentionBackendEnum]:
        return self._supported_attention_backends


class ImageEncoder(nn.Module, ABC, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = [
        "layers",
        "vision_model.encoder.layers",
        "model.visual.blocks",
    ]
    _supported_attention_backends: set[AttentionBackendEnum] = (
        ImageEncoderConfig()._supported_attention_backends
    )

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseEncoderOutput:
        pass

    @property
    def supported_attention_backends(self) -> set[AttentionBackendEnum]:
        return self._supported_attention_backends
