# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.runtime.cache.base import DiffusionCache
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


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


_CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}


class CachableDiT(BaseDiT):
    """
    BaseDiT subclass that adds pluggable step-caching support (TeaCache / MagCache).

    self.cache is None until init_cache() is called on the first forward pass,
    at which point the correct DiffusionCache strategy is constructed from the
    runtime forward_batch context.
    """

    _fsdp_shard_conditions = []
    param_names_mapping = {}
    reverse_param_names_mapping = {}
    lora_param_names_mapping: dict = {}
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    _supported_attention_backends: set[AttentionBackendEnum] = (
        DiTConfig()._supported_attention_backends
    )

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.cache: DiffusionCache | None = None
        self.calibrate_cache: bool = False
        self.cnt: int = 0

    def init_cache(self) -> None:
        """Construct the cache strategy from the current forward_batch context.

        Called lazily on the first forward pass because sampling params
        (teacache_params, magcache_params, num_steps) are only available then.
        """
        from sglang.multimodal_gen.runtime.cache.magcache import MagCacheStrategy
        from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheStrategy
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        fb = get_forward_context().forward_batch
        if fb is None:
            return

        if fb.enable_teacache and fb.enable_magcache:
            raise ValueError("TeaCache and MagCache cannot both be enabled")

        self.calibrate_cache = fb.calibrate_cache
        supports_cfg = self.config.prefix.lower() in _CFG_SUPPORTED_PREFIXES

        if fb.enable_teacache:
            self.cache = TeaCacheStrategy(supports_cfg_cache=supports_cfg)
        elif fb.enable_magcache:
            self.cache = MagCacheStrategy(supports_cfg_cache=supports_cfg)
        else:
            self.cache = None

    # todo: only used in hunyuanvideo.py; refactor and remove this method
    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor, ctx
    ) -> None:
        if self.cache is not None:
            self.cache.maybe_cache(hidden_states, original_hidden_states, ctx)

    # todo: only used in hunyuanvideo.py; refactor and remove this method
    def retrieve_cached_states(self, hidden_states: torch.Tensor, ctx) -> torch.Tensor:
        return self.cache.retrieve(hidden_states, ctx)

    # todo: only used in hunyuanvideo.py; refactor and remove this method
    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        if self.cache is None or self.calibrate_cache:
            return False
        ctx = self.cache.get_context(self.cnt)
        if ctx is None:
            return False
        return self.cache.should_skip(ctx, **kwargs)

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
