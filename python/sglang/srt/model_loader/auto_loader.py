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
"""Centralized weight loading utilities for native SGLang models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Union

import msgspec
import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper

__all__ = [
    "AutoWeightsLoader",
    "WeightsMapper",
    "StackedParamsDispatch",
    "ExpertParamsDispatch",
    "STANDARD_QKV_MAPPING",
    "STANDARD_GATE_UP_MAPPING",
    "STANDARD_STACKED_MAPPING",
    "LLAMA_STACKED_MAPPING",
    "MOE_EXPERT_STACKED_SKIP_SUBSTRS",
    "try_load_stacked_skip_moe_experts",
    "load_with_stacked_dispatch",
    "load_moe_sparse_block_weights",
    "filter_pp_weights",
    "register_weight_remap",
    "get_weight_remap",
]


class StackedParamsDispatch(msgspec.Struct, frozen=True):
    mappings: tuple[tuple[str, str, Union[int, str]], ...] = ()

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        missing_target: str | None = None
        for fused_name, source_name, shard_id in self.mappings:
            if source_name not in name:
                continue
            target = name.replace(source_name, fused_name)
            param = params_dict.get(target)
            if param is None:
                if missing_target is None:
                    missing_target = target
                continue
            param.weight_loader(param, tensor, shard_id)
            return target
        if missing_target is not None:
            raise ValueError(
                f"Mapped checkpoint weight {name!r} to missing parameter "
                f"{missing_target!r}"
            )
        return None


STANDARD_QKV_MAPPING = StackedParamsDispatch(
    mappings=(
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )
)

STANDARD_GATE_UP_MAPPING = StackedParamsDispatch(
    mappings=(
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
)

STANDARD_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
)

LLAMA_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    )
)

MOE_EXPERT_STACKED_SKIP_SUBSTRS: tuple[str, ...] = ("mlp.experts", "experts.")


def try_load_stacked_skip_moe_experts(
    dispatch: StackedParamsDispatch,
    name: str,
    tensor: torch.Tensor,
    params_dict: dict[str, Parameter],
    *,
    skip_substrs: tuple[str, ...] = MOE_EXPERT_STACKED_SKIP_SUBSTRS,
) -> str | None:
    missing_target: str | None = None
    for fused_name, source_name, shard_id in dispatch.mappings:
        if source_name not in name:
            continue
        if any(skip in name for skip in skip_substrs):
            continue
        target = name.replace(source_name, fused_name)
        param = params_dict.get(target)
        if param is None:
            if missing_target is None:
                missing_target = target
            continue
        param.weight_loader(param, tensor, shard_id)
        return target
    if missing_target is not None:
        raise ValueError(
            f"Mapped checkpoint weight {name!r} to missing parameter "
            f"{missing_target!r}"
        )
    return None


class ExpertParamsDispatch(msgspec.Struct, frozen=True):
    mappings: tuple[tuple[str, str, int, str], ...] = ()

    @classmethod
    def from_fused_moe_mapping(
        cls,
        expert_params_mapping: list[tuple[str, str, int, str]],
    ) -> ExpertParamsDispatch:
        return cls(mappings=tuple(expert_params_mapping))

    @classmethod
    def from_gate_up_down(
        cls,
        *,
        num_experts: int,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str = "up_proj",
    ) -> ExpertParamsDispatch:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        return cls.from_fused_moe_mapping(
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name=ckpt_gate_proj_name,
                ckpt_down_proj_name=ckpt_down_proj_name,
                ckpt_up_proj_name=ckpt_up_proj_name,
                num_experts=num_experts,
            )
        )

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        missing_target: str | None = None
        for param_name, weight_name, expert_id, shard_id in self.mappings:
            if weight_name not in name:
                continue
            target = name.replace(weight_name, param_name)
            param = params_dict.get(target)
            if param is None:
                if missing_target is None:
                    missing_target = target
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                tensor,
                target,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return target
        if missing_target is not None:
            raise ValueError(
                f"Mapped checkpoint expert weight {name!r} to missing "
                f"parameter {missing_target!r}"
            )
        return None


def load_with_stacked_dispatch(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    mapping: StackedParamsDispatch,
    *,
    ignore_unexpected_suffixes: tuple[str, ...] = (".bias", ".kv_scale"),
) -> set[str]:
    """Load submodule weights via stacked dispatch, then direct param loaders."""
    loaded: set[str] = set()
    params_dict = dict(module.named_parameters())
    for name, tensor in weights:
        if name.endswith(ignore_unexpected_suffixes):
            mapped_targets = (
                name.replace(source_name, fused_name)
                for fused_name, source_name, _ in mapping.mappings
                if source_name in name
            )
            if name not in params_dict and not any(
                target in params_dict for target in mapped_targets
            ):
                continue
        target = mapping.try_load(name, tensor, params_dict)
        if target is not None:
            if target in params_dict:
                loaded.add(target)
            continue
        if name.endswith("_scale") and name not in params_dict:
            if abs(tensor.item() - 1.0) >= 1e-6:
                raise AssertionError(
                    f"Expected unit scale 1.0, got {tensor.item()} for {name}"
                )
            continue
        if name in params_dict:
            wl = getattr(params_dict[name], "weight_loader", default_weight_loader)
            wl(params_dict[name], tensor)
            loaded.add(name)
        elif not any(name.endswith(suffix) for suffix in ignore_unexpected_suffixes):
            raise ValueError(
                f"No parameter named {name!r} in {module._get_name()}."
            )
    return loaded


def load_moe_sparse_block_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    expert_dispatch: ExpertParamsDispatch,
    dense_stacked: StackedParamsDispatch = STANDARD_GATE_UP_MAPPING,
    ignore_unexpected_suffixes: tuple[str, ...] = (".bias", "_bias", ".kv_scale"),
) -> set[str]:
    loaded: set[str] = set()
    params_dict = dict(module.named_parameters())
    for name, tensor in weights:
        if name.endswith(ignore_unexpected_suffixes):
            mapped_targets = [
                name.replace(source_name, fused_name)
                for fused_name, source_name, _ in dense_stacked.mappings
                if source_name in name
            ]
            mapped_targets.extend(
                name.replace(weight_name, param_name)
                for param_name, weight_name, _, _ in expert_dispatch.mappings
                if weight_name in name
            )
            if name not in params_dict and not any(
                target in params_dict for target in mapped_targets
            ):
                continue
        target = try_load_stacked_skip_moe_experts(
            dense_stacked, name, tensor, params_dict
        )
        if target is not None:
            if target in params_dict:
                loaded.add(target)
            continue
        target = expert_dispatch.try_load(name, tensor, params_dict)
        if target is not None:
            if target in params_dict:
                loaded.add(target)
            continue
        if name.endswith("_scale") and name not in params_dict:
            if abs(tensor.item() - 1.0) >= 1e-6:
                raise AssertionError(
                    f"Expected unit scale 1.0, got {tensor.item()} for {name}"
                )
            continue
        if name not in params_dict:
            raise ValueError(
                f"No parameter named {name!r} in {module._get_name()}."
            )
        wl = getattr(params_dict[name], "weight_loader", default_weight_loader)
        wl(params_dict[name], tensor)
        loaded.add(name)
    return loaded


def filter_pp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    start_layer: int,
    end_layer: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        layer_id = get_layer_id(name)
        if layer_id is not None and (layer_id < start_layer or layer_id >= end_layer):
            continue
        yield name, tensor


_REMAP_REGISTRY: dict[str, Callable[[nn.Module], WeightsMapper]] = {}


def register_weight_remap(*class_names: str):
    def decorator(fn: Callable[[nn.Module], WeightsMapper]):
        for cn in class_names:
            _REMAP_REGISTRY[cn] = fn
        return fn

    return decorator


def get_weight_remap(model: nn.Module) -> WeightsMapper | None:
    fn = _REMAP_REGISTRY.get(type(model).__name__)
    if fn is None:
        return None
    return fn(model)


@register_weight_remap("LlamaForCausalLM")
def _llama_remap(model: nn.Module) -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_suffix={
            ".activation_scale": ".input_scale",
            ".weight_scale_inv": ".weight_scale",
        }
    )
