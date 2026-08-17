# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

# NOTE: SpectrumMixin lives in runtime.cache.spectrum
from sglang.multimodal_gen.runtime.cache.spectrum import SpectrumMixin

# NOTE: TeaCacheContext and TeaCacheMixin have been moved to
# sglang.multimodal_gen.runtime.cache.teacache
# For backwards compatibility, re-export from the new location
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheContext  # noqa: F401
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


# TODO
class BaseDiT(nn.Module, ABC):
    # These are runtime implementation capabilities, not checkpoint metadata.
    # Concrete DiT implementations override them when their tensor layout or
    # execution semantics support only a subset of the available backends.
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    # Methods that drive a forward pass without going through __call__. FSDP2
    # only unshards around the wrapped module's own forward, so anything the
    # shard conditions left in the root group stays sharded unless the entry
    # point is registered; loaders read this and register each name.
    _fsdp_forward_methods: tuple[str, ...] = ()
    param_names_mapping: dict
    reverse_param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    _supported_attention_backends: set[AttentionBackendEnum] = {
        AttentionBackendEnum.SLIDING_TILE_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.FA,
        AttentionBackendEnum.AITER,
        AttentionBackendEnum.AITER_SAGE,
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN,
        AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
        AttentionBackendEnum.VMOBA_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.LASER_ATTN,
        AttentionBackendEnum.BLOCK_SPARSE_ATTN,
        AttentionBackendEnum.RAIN_FUSION_ATTN,
    }

    def __init_subclass__(cls) -> None:
        required_class_attrs = [
            "_fsdp_shard_conditions",
            "param_names_mapping",
            "_compile_conditions",
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' class variable"
                )

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any], **kwargs) -> None:
        super().__init__()
        # `config.arch_config` contains static model metadata. Runtime
        # capabilities remain class attributes on the model implementation.
        self.config: DiTArchConfig = config.arch_config
        self.prefix = config.prefix
        self.hf_config = hf_config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def __post_init__(self) -> None:
        required_attrs = ["hidden_size", "num_attention_heads", "num_channels_latents"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )

    def post_load_weights(self) -> None:
        """Run model-specific post-load weight fixups after all parameters are materialized."""
        return None

    @property
    def supported_attention_backends(self) -> set[AttentionBackendEnum]:
        return self._supported_attention_backends

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


class CachableDiT(SpectrumMixin, TeaCacheMixin, BaseDiT):
    """
    Base class for DiT models that support inference-time cache accelerators.

    Inherits ``SpectrumMixin`` (Chebyshev step skipping) and ``TeaCacheMixin``
    (temporal L1 similarity caching) plus ``BaseDiT`` core functionality.

    """

    # These are required class attributes that should be overridden by concrete implementations
    _fsdp_shard_conditions = []
    param_names_mapping = {}
    reverse_param_names_mapping = {}
    lora_param_names_mapping: dict = {}
    # Ensure these instance attributes are properly defined in subclasses
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._init_spectrum_state()
        self._init_teacache_state()

    @classmethod
    def get_nunchaku_quant_rules(cls) -> dict[str, dict[str, Any]]:
        """
        Get quantization rules for Nunchaku quantization.

        Returns a dict mapping layer name patterns to quantization configs:
        {
            "skip": [list of patterns to skip quantization],
            "svdq_w4a4": [list of patterns for SVDQ W4A4],
            "awq_w4a16": [list of patterns for AWQ W4A16],
        }
        """
        return {}
