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

    ``config.parallel_folding_mode`` is set by ServerArgs.adjust_pipeline_config
    when the encoder is folded over a larger group than its own TP (the idle DiT
    replica during the encoding stage); when it is None the encoder uses the
    default TP group. Shared by every text/image encoder so the choice lives in
    one place.
    """
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


# Folding pays off only for wide encoders: measured ~-22% encode latency for
# T5-XXL (hidden 4096) and larger for Mistral-24B (hidden 5120), but a net loss
# for narrower ones (Qwen3 hidden 2560, CLIP 512) whose per-layer all_reduce
# dominates the sharded compute. Decided on the real (post-load) hidden size.
FOLD_MIN_HIDDEN_SIZE = 4096


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
    """Fold only encoders wide enough to benefit whose heads and MLP divide the
    fold group. Size-based (not per-architecture), so the same encoder family at
    different parameter counts is handled correctly."""
    hidden, _, _ = _encoder_dims(config)
    return (
        _encoder_dims_divide(config, group_size)
        and hidden is not None
        and hidden >= FOLD_MIN_HIDDEN_SIZE
    )


def finalize_encoder_folding(config: EncoderConfig, policy: str = "auto") -> None:
    """Loader hook: resolve the load-time encoder layout (fold vs replicate) for
    the ``encoder_parallel`` policy, after the real dims are populated
    (update_model_arch) and before construction.

    adjust_pipeline_config proposes a fold group from the parallelism alone; here
    we keep it only when the policy asks to fold AND the encoder can (dims divide
    the group), otherwise clear the mode so the encoder loads replicated -- a full
    copy per rank, which the encode stage can data-parallel at batch > 1.

      - "auto": fold only encoders wide enough to benefit at their real size;
      - "fold": fold wherever the dims divide the group (size-agnostic override);
      - "dp" / "replicate": never fold -> always replicated.

    fold and dp/replicate are mutually exclusive here: folding shards the weights
    (model-parallel), dp/replicate keep a full copy per rank; a loaded model is
    one or the other.
    """
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
