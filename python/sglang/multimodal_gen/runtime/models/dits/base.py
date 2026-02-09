# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models import DiTConfig

# NOTE: TeaCacheContext and TeaCacheMixin have been moved to
# sglang.multimodal_gen.runtime.cache.teacache
# For backwards compatibility, re-export from the new location
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheContext  # noqa: F401
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.cache.magcache import MagCacheMixin
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


# TODO
class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict
    reverse_param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    # always supports torch_sdpa
    _supported_attention_backends: set[AttentionBackendEnum] = (
        DiTConfig()._supported_attention_backends
    )

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
        self.config = config
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

    @property
    def supported_attention_backends(self) -> set[AttentionBackendEnum]:
        return self._supported_attention_backends

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


class CachableDiT(TeaCacheMixin, MagCacheMixin, BaseDiT):
    """
    Intermediate base class that provides both TeaCache and MagCache optimization.

    Inherits from both TeaCacheMixin and MagCacheMixin, and routes between them
    based on enable_teacache / enable_magcache flags.
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
    # always supports torch_sdpa
    _supported_attention_backends: set[AttentionBackendEnum] = (
        DiTConfig()._supported_attention_backends
    )

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # self._init_teacache_state()  # Initializes shared state + TeaCache state
        self._init_magcache_state()  # Adds MagCache-specific state

    def reset_cache_state(self) -> None:
        """Reset both TeaCache and MagCache state."""
        # self.reset_teacache_state()  # Resets shared state + TeaCache state
        self.reset_magcache_state()  # Resets MagCache state

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """
        Cache residual for later retrieval.

        SHARED implementation - both TeaCache and MagCache cache residuals identically.
        """
        residual = hidden_states.squeeze(0) - original_hidden_states
        ic(residual.shape)
        if not self.is_cfg_negative:
            self.previous_residual = residual
        else:
            self.previous_residual_negative = residual

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Retrieve cached residual.

        SHARED implementation - both TeaCache and MagCache retrieve identically.
        """
        ic(hidden_states.shape)
        if not self.is_cfg_negative:
            return hidden_states + self.previous_residual
        else:
            return hidden_states + self.previous_residual_negative

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        """Override in subclass to implement cache decision logic."""
        return False
