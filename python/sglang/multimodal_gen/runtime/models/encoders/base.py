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
    get_replica_group,
    get_sp_group,
    get_tp_group,
    get_world_group,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def get_folding_tp_group(config: EncoderConfig):
    """Return the TP group selected for an encoder."""
    mode = config.parallel_folding_mode
    if mode == "sp":
        return get_sp_group()
    if mode == "world":
        # the whole single-replica DiT (all GPUs), regardless of tp/sp/cfg.
        return get_world_group()
    if mode == "replica":
        # the ranks serving this rank's pipeline replica (== world when
        # dp_size is 1); the shape-independent choice for explicit folding
        return get_replica_group()
    if mode is None:
        return get_tp_group()
    raise ValueError(f"Unsupported encoder folding mode: {mode!r}")


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


def group_has_measured_topology(group) -> bool:
    """Whether the measured fold/dp verdicts transfer to this group's topology.

    Both thresholds above were measured on single-node H100s over NVLink. Their
    costs are pure interconnect: folding adds an all_reduce per layer, dp one
    all_gather per encode. Without peer-to-peer between the ranks (multi-node, or
    a host-routed topology) the traffic costs several times more and a rule that
    barely paid on NVLink can invert, so `auto` treats those topologies as
    unmeasured and stays replicated. An explicit --encoder-parallel still wins.
    """
    local_devices = torch.cuda.device_count()
    if group.world_size <= 1 or group.world_size > local_devices:
        return False
    return all(
        torch.cuda.can_device_access_peer(0, peer)
        for peer in range(1, group.world_size)
    )


def encoder_dp_capable(config: EncoderConfig) -> bool:
    """wide enough that splitting a batched encode beats its one all_gather"""
    hidden, _, _ = _encoder_dims(config)
    return hidden is not None and hidden >= DP_MIN_HIDDEN_SIZE


def encoder_dp_worthwhile(
    config: EncoderConfig, batch_size: int, measured_topology: bool
) -> bool:
    return measured_topology and batch_size > 1 and encoder_dp_capable(config)


def finalize_encoder_folding(
    config: EncoderConfig, policy: str = "auto", prefer_dp: bool = False
) -> None:
    """resolve fold-vs-replicate once real dims are known (post update_model_arch,
    pre construction); folding shards the weights, so it rules out dp for the
    lifetime of the loaded model. `prefer_dp` means the runtime can engage dp."""
    if config.parallel_folding_mode is None:
        return
    group = get_folding_tp_group(config)
    if policy == "fold":
        # explicit: shard whenever the dims allow, topology is the caller's call
        keep = _encoder_dims_divide(config, group.world_size)
    elif policy == "auto":
        # a batched encode prefers dp (one all_gather) over folding (an
        # all_reduce per layer), so leave a dp-capable encoder unsharded
        keep = (
            not (prefer_dp and encoder_dp_capable(config))
            and encoder_folding_worthwhile(config, group.world_size)
            and group_has_measured_topology(group)
        )
    else:  # dp / replicate
        keep = False
    if not keep:
        config.parallel_folding_mode = None


class EncoderTensorParallelMixin:
    """Keep an encoder on the TP group that was used to build its shards."""

    _encoder_tp_group: GroupCoordinator | None = None
    checkpoint_quantization_backend = "diffusion"
    packed_modules_mapping: dict[str, list[str]] = {}
    # Some encoders own checkpoint quantization end to end because their weight
    # states or sharding contract cannot use the generic loader lifecycle.
    manages_checkpoint_quantization = False

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return True

    @classmethod
    def configure_component_paths(
        cls,
        config: EncoderConfig,
        component_paths: dict[str, str],
    ) -> None:
        """Apply optional runtime components before parallel layout is resolved."""

    def bind_encoder_tp_group(self, tp_group: GroupCoordinator) -> None:
        self._encoder_tp_group = tp_group

    def __call__(self, *args, **kwargs):
        tp_group = self._encoder_tp_group
        if tp_group is None:
            return super().__call__(*args, **kwargs)
        with use_tensor_parallel_group(tp_group):
            return super().__call__(*args, **kwargs)


class TextEncoder(
    EncoderTensorParallelMixin, nn.Module, ABC, LayerwiseOffloadableModuleMixin
):
    # Opt in per encoder to data-parallel batched encoding: the gather rebuilds a
    # BaseEncoderOutput, and subclasses are free to return their own output type
    # instead (Qwen2_5_VLForConditionalGeneration returns
    # Qwen2_5_VLCausalLMOutputWithPast). Off by default so a new encoder is
    # replicated rather than silently broken; flip it once dp is verified there.
    supports_dp_encode = False
    # Some encoders own checkpoint quantization end to end because their weight
    # states or sharding contract cannot use the generic loader lifecycle.
    manages_checkpoint_quantization = False
    layerwise_offload_dit_group_enabled = False
    layer_names = [
        "layers",
        "encoder.block",
        "text_model.encoder.layers",
        "model.language_model.layers",
    ]
    _fsdp_shard_conditions: list = field(default_factory=lambda: [])
    # Methods that drive a forward pass without going through __call__. FSDP2
    # only unshards around the wrapped module's own forward, so anything the
    # shard conditions left in the root group stays sharded unless the entry
    # point is registered; loaders read this and register each name.
    _fsdp_forward_methods: tuple[str, ...] = ()
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


class ImageEncoder(
    EncoderTensorParallelMixin, nn.Module, ABC, LayerwiseOffloadableModuleMixin
):
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
