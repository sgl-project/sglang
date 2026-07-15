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

import re

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
    "FusedExpertDispatch",
    "STANDARD_QKV_MAPPING",
    "STANDARD_GATE_UP_MAPPING",
    "STANDARD_STACKED_MAPPING",
    "LLAMA_STACKED_MAPPING",
    "QWEN3_NEXT_GDN_STACKED_MAPPING",
    "QWEN35_STACKED_MAPPING",
    "QWEN35_MOE_IGNORE_SUFFIXES",
    "MOE_EXPERT_STACKED_SKIP_SUBSTRS",
    "try_load_stacked_skip_moe_experts",
    "load_with_stacked_dispatch",
    "load_moe_sparse_block_weights",
    "normalize_qwen35_weight_name",
    "load_qwen35_moe_checkpoint_weights",
    "filter_pp_weights",
    "register_weight_remap",
    "get_weight_remap",

    "MultiInputFusion",
    "DEEPSEEK_GATE_UP_MAPPING",
    "remap_fused_shared_expert_names",
    "maybe_remap_deepseek_mla_kv_scale",]


class StackedParamsDispatch(msgspec.Struct, frozen=True):
    mappings: tuple[tuple[str, str, Union[int, str]], ...] = ()

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        for fused_name, source_name, shard_id in self.mappings:
            if source_name not in name:
                continue
            target = name.replace(source_name, fused_name)
            param = params_dict.get(target)
            if param is None:
                return target
            param.weight_loader(param, tensor, shard_id)
            return target
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

QWEN3_NEXT_GDN_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
        ("in_proj_qkvz.", "in_proj_z.", 3),
        ("in_proj_ba.", "in_proj_b.", 0),
        ("in_proj_ba.", "in_proj_a.", 1),
    )
)

QWEN35_STACKED_MAPPING = StackedParamsDispatch(
    mappings=STANDARD_STACKED_MAPPING.mappings + QWEN3_NEXT_GDN_STACKED_MAPPING.mappings
)

QWEN35_MOE_IGNORE_SUFFIXES = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    ".weight_scale",
    "_weight_scale",
    ".input_scale",
    "_input_scale",
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
    for fused_name, source_name, shard_id in dispatch.mappings:
        if source_name not in name:
            continue
        if any(skip in name for skip in skip_substrs):
            continue
        target = name.replace(source_name, fused_name)
        param = params_dict.get(target)
        if param is None:
            return target
        param.weight_loader(param, tensor, shard_id)
        return target
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
        for param_name, weight_name, expert_id, shard_id in self.mappings:
            if weight_name not in name:
                continue
            target = name.replace(weight_name, param_name)
            if (
                target.endswith(".bias") or target.endswith("_bias")
            ) and target not in params_dict:
                return target
            param = params_dict.get(target)
            if param is None:
                return target
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(
                    param,
                    tensor,
                    target,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
            except TypeError:
                weight_loader(param, tensor)
            return target
        return None


class FusedExpertDispatch(msgspec.Struct, frozen=True):
    num_experts: int
    gate_up_ckpt_substr: str = "experts.gate_up_proj"
    down_ckpt_substr: str = "experts.down_proj"
    w13_runtime_substr: str = "experts.w13_weight"
    w2_runtime_substr: str = "experts.w2_weight"

    @staticmethod
    def fan_out_to_experts(
        param: Parameter,
        loaded_weight: torch.Tensor,
        runtime_name: str,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        for expert_id in range(num_experts):
            weight_loader(
                param,
                loaded_weight[expert_id],
                runtime_name,
                shard_id,
                expert_id,
            )
        return True

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        if self.gate_up_ckpt_substr in name:
            target = name.replace(self.gate_up_ckpt_substr, self.w13_runtime_substr)
            param = params_dict.get(target)
            if param is None:
                return target
            w1, w3 = tensor.chunk(2, dim=-2)
            self.fan_out_to_experts(param, w1, target, "w1", self.num_experts)
            self.fan_out_to_experts(param, w3, target, "w3", self.num_experts)
            return target
        if self.down_ckpt_substr in name:
            target = name.replace(self.down_ckpt_substr, self.w2_runtime_substr)
            param = params_dict.get(target)
            if param is None:
                return target
            self.fan_out_to_experts(param, tensor, target, "w2", self.num_experts)
            return target
        return None


def normalize_qwen35_weight_name(name: str) -> str:
    if "language_model" in name:
        name = name.replace(r"model.language_model.", r"model.")
    if ".self_attn." in name:
        name = name.replace(".self_attn", "")
    return name


def load_qwen35_moe_checkpoint_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    num_experts: int,
    expert_dispatch: ExpertParamsDispatch,
    fused_dispatch: FusedExpertDispatch | None,
    stacked_mapping: StackedParamsDispatch = QWEN35_STACKED_MAPPING,
    ignore_suffixes: tuple[str, ...] = QWEN35_MOE_IGNORE_SUFFIXES,
    start_layer: int | None = None,
    end_layer: int | None = None,
    skip_substrs: tuple[str, ...] = ("rotary_emb.inv_freq", "mtp", "visual"),
    enable_shared_expert_fusion: bool = False,
    shared_expert_slot: int | None = None,
    encoder_only: bool = False,
    remap_visual: bool = False,
    on_embed_for_tied_lm_head: Callable[[str, torch.Tensor], None] | None = None,
) -> set[str]:
    import logging

    log = logging.getLogger(__name__)
    loaded_params: set[str] = set()
    params_dict = dict(module.named_parameters(remove_duplicate=False))
    is_fused_expert = False
    active_expert_dispatch = expert_dispatch
    fused_gate_up_mapping: tuple[tuple[str, str, int, str], ...] = (
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
    )
    if enable_shared_expert_fusion and shared_expert_slot is not None:
        slot = shared_expert_slot
        fused_gate_up_mapping = fused_gate_up_mapping + (
            ("experts.w13_", f"experts.{slot}.gate_up_proj.", slot, "w1"),
            ("experts.w2_", f"experts.{slot}.down_proj.", slot, "w2"),
            ("experts.w13_", f"experts.{slot}.gate_proj.", slot, "w1"),
            ("experts.w13_", f"experts.{slot}.up_proj.", slot, "w3"),
        )
    for name, loaded_weight in weights:
        if any(sub in name for sub in skip_substrs):
            continue
        name = normalize_qwen35_weight_name(name)
        if on_embed_for_tied_lm_head is not None:
            on_embed_for_tied_lm_head(name, loaded_weight)
        if enable_shared_expert_fusion and shared_expert_slot is not None:
            if "mlp.shared_expert." in name:
                name = name.replace(
                    "mlp.shared_expert.",
                    f"mlp.experts.{shared_expert_slot}.",
                )
        layer_id = get_layer_id(name)
        if (
            layer_id is not None
            and start_layer is not None
            and end_layer is not None
            and (layer_id < start_layer or layer_id >= end_layer)
        ):
            continue
        if (
            "experts.gate_up_proj" in name
            or "experts.down_proj" in name
            or name.endswith("experts.gate_up_proj")
            or name.endswith("experts.down_proj")
        ):
            is_fused_expert = True
            active_expert_dispatch = ExpertParamsDispatch.from_fused_moe_mapping(
                list(fused_gate_up_mapping)
            )
        target = try_load_stacked_skip_moe_experts(
            stacked_mapping, name, loaded_weight, params_dict
        )
        if target is not None:
            loaded_params.add(name)
            continue
        if is_fused_expert and fused_dispatch is not None:
            fused_target = fused_dispatch.try_load(name, loaded_weight, params_dict)
            if fused_target is not None:
                loaded_params.add(name)
                continue
            if enable_shared_expert_fusion and shared_expert_slot is not None:
                handled = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in fused_gate_up_mapping:
                    if weight_name not in name:
                        continue
                    if "visual" in name or encoder_only:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped not in params_dict:
                        handled = True
                        break
                    param = params_dict[name_mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if f"{shared_expert_slot}.gate_up_proj" in name:
                        w1, w3 = loaded_weight.chunk(2, dim=-2)
                        weight_loader(param, w1, name_mapped, "w1", expert_id)
                        weight_loader(param, w3, name_mapped, "w3", expert_id)
                    else:
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id,
                            expert_id,
                        )
                    loaded_params.add(name)
                    handled = True
                    break
                if handled:
                    continue
        expert_target = active_expert_dispatch.try_load(
            name, loaded_weight, params_dict
        )
        if expert_target is not None:
            loaded_params.add(name)
            continue
        if any(
            weight_name in name
            for _, weight_name, _, _ in active_expert_dispatch.mappings
        ):
            loaded_params.add(name)
            continue
        if remap_visual and "visual" in name:
            name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
            name = name.replace(r"model.visual.", r"visual.")
        if name.endswith(ignore_suffixes) and name not in params_dict:
            continue
        if name in params_dict:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        else:
            log.warning("Parameter %s not found in params_dict", name)
    return loaded_params


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
            pass
    return loaded


def load_moe_sparse_block_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    expert_dispatch: ExpertParamsDispatch,
    dense_stacked: StackedParamsDispatch = STANDARD_GATE_UP_MAPPING,
) -> set[str]:
    loaded: set[str] = set()
    params_dict = dict(module.named_parameters())
    for name, tensor in weights:
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
        if name.endswith(".bias") and name not in params_dict:
            continue
        if name not in params_dict:
            continue
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

DEEPSEEK_GATE_UP_MAPPING = STANDARD_GATE_UP_MAPPING


_SHARED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(.+)$"
)


class MultiInputFusion:
    def __init__(
        self,
        *,
        source_substrs: tuple[str, ...],
        fused_param_substr: str,
        cat_dim: int = 0,
        tensor_cloner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.source_substrs = source_substrs
        self.fused_param_substr = fused_param_substr
        self.cat_dim = cat_dim
        self.tensor_cloner = tensor_cloner
        self._pending: dict[str, dict[str, torch.Tensor]] = {}

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:
        matched = next((s for s in self.source_substrs if s in name), None)
        if matched is None:
            return None
        if self.tensor_cloner is not None:
            tensor = self.tensor_cloner(tensor)
        group_key = name.split(matched, 1)[0]
        bucket = self._pending.setdefault(group_key, {})
        bucket[matched] = tensor
        if len(bucket) < len(self.source_substrs):
            return None
        parts = [bucket[s] for s in self.source_substrs]
        fused = (
            parts[0]
            if all(p.shape == torch.Size([]) for p in parts)
            else torch.cat(parts, dim=self.cat_dim)
        )
        fused_name = (
            name.replace("q_a_proj", self.fused_param_substr)
            if matched == "q_a_proj"
            else name.replace("kv_a_proj_with_mqa", self.fused_param_substr)
        )
        param = params_dict.get(fused_name)
        if param is None:
            self._pending.pop(group_key, None)
            return fused_name
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, fused)
        self._pending.pop(group_key, None)
        return fused_name


def remap_fused_shared_expert_names(
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        match = _SHARED_EXPERT_PATTERN.match(name)
        if match:
            layer_id, suffix = match.group(1), match.group(2)
            name = f"model.layers.{layer_id}.mlp.experts.{n_routed_experts}.{suffix}"
        elif "mlp.shared_experts" in name:
            name = name.replace("mlp.shared_experts", f"mlp.experts.{n_routed_experts}")
        yield name, tensor


def maybe_remap_deepseek_mla_kv_scale(
    name: str,
    params_dict: dict[str, Parameter],
) -> str | None:
    if name in params_dict:
        return name
    if "k_scale" not in name and "v_scale" not in name:
        return name
    for scale in ("k_scale", "v_scale"):
        if scale in name:
            if "kv_a_proj_with_mqa" in name:
                candidate = name.replace("kv_a_proj_with_mqa", "attn_mqa")
            else:
                candidate = name.replace(f"{scale[0]}_proj", "attn_mqa")
            if candidate in params_dict:
                return candidate
    return name


@register_weight_remap(
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Glm4MoeLiteForCausalLM",
)
def _deepseek_mla_remap(model: nn.Module) -> WeightsMapper:
    n_routed = getattr(model.config, "n_routed_experts", None)
    substr_map: dict[str, str | None] = {}
    if n_routed is not None:
        substr_map["mlp.shared_experts"] = f"mlp.experts.{n_routed}"
    return WeightsMapper(orig_to_new_substr=substr_map)
