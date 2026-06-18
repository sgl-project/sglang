# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Centralized weight loading utilities for native SGLang models.

This module provides:
- StackedParamsDispatch: reusable stacked-parameter routing (qkv_proj, gate_up_proj).
- filter_pp_weights: generator that drops out-of-range PP layers.
- WeightRemapRegistry: architecture-specific name remap registration.
- Re-exports of AutoWeightsLoader and WeightsMapper from models/utils.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Union

import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper

__all__ = [
    "AutoWeightsLoader",
    "WeightsMapper",
    "StackedParamsDispatch",
    "STANDARD_QKV_MAPPING",
    "STANDARD_GATE_UP_MAPPING",
    "STANDARD_STACKED_MAPPING",
    "filter_pp_weights",
    "register_weight_remap",
    "get_weight_remap",
]


# ---------------------------------------------------------------------------
# Stacked Parameters Dispatch
# ---------------------------------------------------------------------------


@dataclass
class StackedParamsDispatch:
    """Centralized stacked-parameter loading for fused linear layers.

    Handles the common pattern of mapping checkpoint names
    (q_proj, k_proj, v_proj, gate_proj, up_proj) to fused runtime parameters
    (qkv_proj, gate_up_proj) with the correct shard IDs.

    Quantization is handled entirely by ``param.weight_loader`` on the layer —
    this class only routes the tensor to the correct parameter with the correct
    shard_id.

    Usage::

        mapping = StackedParamsDispatch([
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ])
        target = mapping.try_load(name, tensor, params_dict)
    """

    # List of (fused_param_name, checkpoint_source_name, shard_id).
    # shard_id is passed directly to param.weight_loader.
    mappings: list[tuple[str, str, Union[int, str]]] = field(default_factory=list)

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        """Try to load a weight via stacked mapping.

        Returns the loaded runtime parameter name if matched and loaded,
        the target name (for skip tracking) if the target param is missing
        (e.g. optional bias), or None if no mapping matched.
        """
        for fused_name, source_name, shard_id in self.mappings:
            if source_name not in name:
                continue
            target = name.replace(source_name, fused_name)
            param = params_dict.get(target)
            if param is None:
                # Parameter doesn't exist — e.g. GPTQ bias.
                # Return target so caller can track the skip.
                return target
            param.weight_loader(param, tensor, shard_id)
            return target
        return None


# Pre-built instances for the most common decoder patterns.

STANDARD_QKV_MAPPING = StackedParamsDispatch(
    mappings=[
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
)

STANDARD_GATE_UP_MAPPING = StackedParamsDispatch(
    mappings=[
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
)

STANDARD_STACKED_MAPPING = StackedParamsDispatch(
    mappings=[
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
)


# ---------------------------------------------------------------------------
# Pipeline Parallel Weight Filter
# ---------------------------------------------------------------------------


def filter_pp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    start_layer: int,
    end_layer: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Drop checkpoint entries whose layer index is outside [start_layer, end_layer).

    Weights that don't contain a parseable layer index (embed_tokens, lm_head,
    layer norms, etc.) are always passed through.
    """
    for name, tensor in weights:
        layer_id = get_layer_id(name)
        if layer_id is not None and (layer_id < start_layer or layer_id >= end_layer):
            continue
        yield name, tensor


# ---------------------------------------------------------------------------
# Weight Remap Registry
# ---------------------------------------------------------------------------

_REMAP_REGISTRY: dict[str, Callable[[nn.Module], WeightsMapper | None]] = {}


def register_weight_remap(*class_names: str):
    """Decorator to register an architecture-specific weight remap function.

    The decorated function receives a model instance and returns a WeightsMapper
    (or None if no remap is needed for this configuration).

    Example::

        @register_weight_remap("LlamaForCausalLM")
        def _llama_remap(model) -> WeightsMapper:
            return WeightsMapper(orig_to_new_suffix={
                ".activation_scale": ".input_scale",
                ".weight_scale_inv": ".weight_scale",
            })
    """

    def decorator(fn: Callable[[nn.Module], WeightsMapper | None]):
        for cn in class_names:
            _REMAP_REGISTRY[cn] = fn
        return fn

    return decorator


def get_weight_remap(model: nn.Module) -> WeightsMapper | None:
    """Get the registered weight remap for a model instance, or None."""
    fn = _REMAP_REGISTRY.get(type(model).__name__)
    if fn is None:
        return None
    return fn(model)


# ---------------------------------------------------------------------------
# Architecture-Specific Registrations
# ---------------------------------------------------------------------------


@register_weight_remap("LlamaForCausalLM")
def _llama_remap(model: nn.Module) -> WeightsMapper:
    """Llama-family FP8 scale suffix normalization."""
    return WeightsMapper(
        orig_to_new_suffix={
            ".activation_scale": ".input_scale",
            ".weight_scale_inv": ".weight_scale",
        }
    )
