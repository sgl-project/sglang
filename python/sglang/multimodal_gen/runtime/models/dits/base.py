# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class TeaCacheContext:
    """Common context extracted for TeaCache skip decision."""

    current_timestep: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool  # For CFG branch selection
    teacache_thresh: float
    coefficients: list[float]
    teacache_params: TeaCacheParams  # Full params for model-specific access


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


class CachableDiT(BaseDiT):
    """
    An intermediate base class that adds TeaCache optimization functionality to DiT models.
    TeaCache accelerates inference by selectively skipping redundant computation when consecutive
    diffusion steps are similar enough.
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

    # Models that support CFG cache separation (wan/hunyuan/zimage)
    # Models not in this set (flux/qwen) auto-disable TeaCache when CFG is enabled
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        # Common TeaCache state
        self.cnt = 0
        self.enable_teacache = True
        # Flag indicating if this model supports CFG cache separation
        self._supports_cfg_cache = self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES

        # Always initialize positive cache fields (used in all modes)
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance: float = 0.0
        
        self.is_cfg_negative = False
        # CFG-specific fields initialized to None (created when CFG is used)
        # These are only used when _supports_cfg_cache is True AND do_cfg is True
        if self._supports_cfg_cache:
            self.previous_modulated_input_negative: torch.Tensor | None = None
            self.previous_residual_negative: torch.Tensor | None = None
            self.accumulated_rel_l1_distance_negative: float = 0.0
        # Current branch marker (for maybe_cache_states / retrieve_cached_states)
        


       
            

    def reset_teacache_state(self) -> None:
        """Reset all TeaCache state at the start of each generation task."""
        # Common fields
        self.cnt = 0

        # Primary cache fields (always present)
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0
        self.is_cfg_negative = False
        # Branch marker
        self.enable_teacache = True
        # CFG negative cache fields (always reset, may be unused)
        if self._supports_cfg_cache:
            self.previous_modulated_input_negative = None
            self.previous_residual_negative = None
            self.accumulated_rel_l1_distance_negative = 0.0

        

       

      

    def _compute_l1_and_decide(
        self,
        modulated_inp: torch.Tensor,
        coefficients: list[float],
        teacache_thresh: float,
    ) -> bool:
        """
        Compute L1 distance and decide whether to calculate or use cache.

        Args:
            modulated_inp: Current timestep's modulated input
            prev_input_attr: Attribute name for previous modulated input
            distance_attr: Attribute name for accumulated L1 distance
            coefficients: Polynomial coefficients for L1 rescaling
            teacache_thresh: Threshold for cache decision

        Returns:
            should_calc: True if forward computation is needed, False to use cache
        """

       
        # Compute relative L1 distance
        prev_modulated_inp = self.previous_modulated_input_negative if self.is_cfg_negative else self.previous_modulated_input
        diff = modulated_inp - prev_modulated_inp
        rel_l1 = (diff.abs().mean() / prev_modulated_inp.abs().mean()).cpu().item()

        # Apply polynomial rescaling
        rescale_func = np.poly1d(coefficients)
        
        accumulated_rel_l1_distance = self.accumulated_rel_l1_distance_negative if self.is_cfg_negative else self.accumulated_rel_l1_distance
        accumulated_rel_l1_distance = accumulated_rel_l1_distance + rescale_func(rel_l1)
        
        should_calc = accumulated_rel_l1_distance >= teacache_thresh
        if not should_calc:
            if not self.is_cfg_negative:
                self.accumulated_rel_l1_distance = accumulated_rel_l1_distance
            else:
                self.accumulated_rel_l1_distance_negative = accumulated_rel_l1_distance
        else:
            if not self.is_cfg_negative:
                self.accumulated_rel_l1_distance = 0
            else:
                self.accumulated_rel_l1_distance_negative = 0
        return should_calc
        
    def _compute_teacache_decision(
        self,
        modulated_inp: torch.Tensor,
        is_boundary_step: bool,
        coefficients: list[float],
        teacache_thresh: float,
    ) -> bool:
        """
        Compute cache decision for TeaCache.


        Args:
            modulated_inp: Current timestep's modulated input
            modulated_inp: Current timestep's modulated input
            is_boundary_step: True for boundary timesteps that always compute
            coefficients: Polynomial coefficients for L1 rescaling
            teacache_thresh: Threshold for cache decision

        Returns:
            should_calc: True if forward computation is needed, False to use cache
        """
        # Boundary steps always compute (early return)
        if not self.enable_teacache:
            return True
        if is_boundary_step:
            if not self.is_cfg_negative:
                self.accumulated_rel_l1_distance = 0.0
                self.previous_modulated_input = modulated_inp.clone()
            elif self._supports_cfg_cache :
                self.accumulated_rel_l1_distance_negative = 0.0
                self.previous_modulated_input_negative = modulated_inp.clone()
            return True
        return self._compute_l1_and_decide(
            modulated_inp=modulated_inp,
            coefficients=coefficients,
            teacache_thresh=teacache_thresh,
        )
        
      

    
    def _get_teacache_context(self) -> TeaCacheContext | None:
        """
        Check TeaCache preconditions and extract common context.

        Returns:
            TeaCacheContext if TeaCache is enabled and properly configured,
            None if should return False (skip TeaCache logic entirely).

        This helper reduces code duplication in should_skip_forward_for_cached_states.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch

        # Early return checks (combined)
        if (
            forward_batch is None
            or not forward_batch.enable_teacache
            or forward_batch.teacache_params is None
        ):
            return None

        teacache_params = forward_batch.teacache_params

        # Extract common values
        current_timestep = forward_context.current_timestep
        num_inference_steps = forward_batch.num_inference_steps
        do_cfg = forward_batch.do_classifier_free_guidance
        is_cfg_negative = forward_batch.is_cfg_negative

        # Reset at first timestep
        if current_timestep == 0 and not self.is_cfg_negative:
            self.reset_teacache_state()

        return TeaCacheContext(
            current_timestep=current_timestep,
            num_inference_steps=num_inference_steps,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            teacache_thresh=teacache_params.teacache_thresh,
            coefficients=teacache_params.coefficients,
            teacache_params=teacache_params,
        )

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        pass

    def should_skip_forward_for_cached_states(self, **kwargs: dict[str, Any]) -> bool:
        return False

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("maybe_retrieve_cached_states is not implemented")
