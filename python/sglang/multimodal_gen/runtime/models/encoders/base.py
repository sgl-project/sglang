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
    """group an encoder tensor-parallels over; the default TP group unless a
    fold mode is set"""
    mode = config.parallel_folding_mode
    if mode == "sp":
        return get_sp_group()
    elif mode == "ulysses":
        return get_sp_group().ulysses_group
    elif mode == "ring":
        return get_sp_group().ring_group
    elif mode == "world":
        # the whole single-replica DiT (all GPUs), regardless of tp/sp/cfg.
        return get_world_group()
    return get_tp_group()


# measured on 2/4xH100: folding wins only for wide encoders (T5-XXL 4096: -20%
# at batch 1, R-insensitive); narrower ones lose to the per-layer all_reduce
# (Qwen3 2560: +35%, CLIP 768: +50%)
FOLD_MIN_HIDDEN_SIZE = 4096
# below this width the encoder stays latency-bound across batch sizes, so
# data-parallel encoding saves no compute and the all_gather is a pure loss
# (CLIP 768: dp slower at every batch/R measured)
DP_MIN_HIDDEN_SIZE = 1024


def _encoder_dims(config: EncoderConfig):
    """Best-effort (hidden, attention_heads, mlp_intermediate) from a config,
    spelled differently across families (hidden_size/d_model, num_heads, d_ff)."""

    def first(names):
        for name in names:
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    return (
        first(("hidden_size", "d_model")),
        first(("num_attention_heads", "num_heads", "n_heads")),
        first(("intermediate_size", "d_ff", "ffn_dim")),
    )


def _encoder_dims_divide(config: EncoderConfig, group_size: int) -> bool:
    """Whether the encoder's heads and MLP evenly divide the fold group -- a hard
    requirement to shard (fold) it at all, regardless of whether it is worth it."""
    _, heads, inter = _encoder_dims(config)
    return (
        group_size > 1
        and heads is not None
        and heads % group_size == 0
        and inter is not None
        and inter % group_size == 0
    )


def encoder_folding_worthwhile(config: EncoderConfig, group_size: int) -> bool:
    """size-based, so the same family at different parameter counts differs"""
    hidden, _, _ = _encoder_dims(config)
    return (
        _encoder_dims_divide(config, group_size)
        and hidden is not None
        and hidden >= FOLD_MIN_HIDDEN_SIZE
    )


def encoder_dp_worthwhile(config: EncoderConfig, batch_size: int) -> bool:
    hidden, _, _ = _encoder_dims(config)
    return batch_size > 1 and hidden is not None and hidden >= DP_MIN_HIDDEN_SIZE


def finalize_encoder_folding(config: EncoderConfig, policy: str = "auto") -> None:
    """resolve fold-vs-replicate once real dims are known (post update_model_arch,
    pre construction); folding shards the weights, so it excludes dp/replicate
    for the lifetime of the loaded model"""
    if config.parallel_folding_mode is None:
        return
    if policy in ("dp", "replicate"):
        config.parallel_folding_mode = None
        return
    group_size = getattr(get_folding_tp_group(config), "world_size", 1)
    keep = (
        _encoder_dims_divide(config, group_size)
        if policy == "fold"
        else encoder_folding_worthwhile(config, group_size)
    )
    if not keep:
        config.parallel_folding_mode = None


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
