# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team


import logging
import re
from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn

from sglang.srt.distributed import get_pp_indices
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.runtime_context import get_parallel

from .layers import Style, _normalize_tp_style
from .utils import maybe_prefix

logger = logging.getLogger(__name__)


class ParallelMixin:
    def _get_model_tp_plan(self) -> Mapping[str, str]:
        plan = (
            getattr(self.model, "tp_plan", None)
            or getattr(self.model, "_tp_plan", None)
            or getattr(self.model.config, "base_model_tp_plan", None)
            or getattr(self.text_config, "base_model_tp_plan", None)
        )
        if plan:
            return plan
        plan = self._infer_tp_plan_from_children()
        return plan if plan else {}

    _LANGUAGE_MODEL_CHILD_NAMES = frozenset(
        {"language_model", "text_model", "model", "lm"}
    )

    def _infer_tp_plan_from_children(self) -> dict[str, str]:
        plan: dict[str, str] = {}
        for child_name, child_module in self.model.named_children():
            child_plan = getattr(child_module, "_tp_plan", None)
            if child_plan:
                plan.update({f"{child_name}.{k}": v for k, v in child_plan.items()})
                continue
            child_config = getattr(child_module, "config", None)
            if child_config is not None:
                child_tp = getattr(child_config, "base_model_tp_plan", None)
                if child_tp:
                    plan.update({f"{child_name}.{k}": v for k, v in child_tp.items()})
                    continue
            if child_name not in self._LANGUAGE_MODEL_CHILD_NAMES:
                continue
            if child_config is None:
                continue
            model_type = getattr(child_config, "model_type", "")
            base_type = (
                model_type.replace("_vl_text", "")
                .replace("_vl", "")
                .replace("_text", "")
            )
            if base_type and base_type != model_type:
                try:
                    from transformers import AutoConfig

                    base_cfg = AutoConfig.for_model(base_type)
                    base_tp = getattr(base_cfg, "base_model_tp_plan", None)
                    if base_tp:
                        plan.update(
                            {f"{child_name}.{k}": v for k, v in base_tp.items()}
                        )
                except Exception as e:
                    logger.debug(
                        "Could not infer TP plan from base model type '%s': %s",
                        base_type,
                        e,
                    )
        return plan

    def _normalize_tp_plan(self, tp_plan: Mapping[str, str]) -> dict[str, Style]:
        normalized = {}
        modules = dict(self.model.named_modules())
        linear_names = [
            name for name, module in modules.items() if isinstance(module, nn.Linear)
        ]
        for pattern, style in tp_plan.items():
            pattern = pattern.removeprefix("^").removesuffix("$")
            if pattern.startswith("model\\."):
                pattern = pattern[len("model\\.") :]
            elif pattern.startswith("model."):
                pattern = pattern[len("model.") :]
            targets = [name for name in linear_names if re.fullmatch(pattern, name)]
            if not targets:
                continue
            normalized_style = _normalize_tp_style(style)
            if get_parallel().tp_size > 1:
                if "packed" in style.lower():
                    raise ValueError(
                        f"Packed linear tensor parallelism requires a packed weight adapter: {pattern}"
                    )
                if normalized_style in {"colwise_rep", "rowwise_rep"}:
                    for name in targets:
                        parts = name.split(".")
                        ancestors = [
                            modules[".".join(parts[:end])] for end in range(len(parts))
                        ]
                        if any(
                            hasattr(module, "is_causal")
                            and getattr(module, "layer_idx", None) is not None
                            for module in ancestors
                        ):
                            raise ValueError(
                                f"Gathered attention projections require a tensor-parallel head adapter: {name}"
                            )
            normalized[pattern] = normalized_style
        return normalized

    def _get_model_pp_plan(self) -> Mapping[str, object]:
        return (
            getattr(self.model, "_pp_plan", None)
            or getattr(self.model, "pp_plan", None)
            or getattr(self.model.config, "base_model_pp_plan", None)
            or getattr(self.text_config, "base_model_pp_plan", None)
            or {}
        )

    def _register_missing_prefix(self, prefix: str):
        if not prefix.endswith("."):
            prefix += "."
        if prefix not in self.skip_prefixes:
            self.skip_prefixes.append(prefix)

    @staticmethod
    def _make_pp_missing_layer(original: nn.Module) -> PPMissingLayer:
        """Create a PPMissingLayer that preserves plain attributes from"""
        replacement = PPMissingLayer()
        for key, value in original.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (nn.Module, nn.Parameter, torch.Tensor)):
                continue
            setattr(replacement, key, value)
        return replacement

    def _get_submodule_or_none(self, name: str) -> Optional[nn.Module]:
        try:
            return self.model.get_submodule(name)
        except AttributeError:
            return None

    def _set_submodule(self, name: str, module: nn.Module):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = self.model.get_submodule(parent_name)
        else:
            parent_module = self.model
            child_name = name
        setattr(parent_module, child_name, module)

    def pipeline_parallel(self):
        if self.pp_group.world_size <= 1:
            return
        pp_plan = self._get_model_pp_plan()
        if not pp_plan:
            raise ValueError(
                f"{type(self.model)} does not support pipeline parallel yet!"
            )
        pp_keys = [re.sub("^model\\.", "", name) for name in pp_plan.keys()]
        module_list_idx = None
        module_list_name = None
        for idx, name in enumerate(pp_keys):
            if isinstance(self._get_submodule_or_none(name), nn.ModuleList):
                if module_list_idx is not None:
                    raise ValueError(
                        "Pipeline parallel with multiple ModuleList blocks is not supported."
                    )
                module_list_idx = idx
                module_list_name = name
        if module_list_idx is None or module_list_name is None:
            raise ValueError(f"Could not find ModuleList in {type(self.model)}.")
        keep_prefix_modules = self.pp_group.is_first_rank or (
            getattr(self.text_config, "tie_word_embeddings", False)
            and self.pp_group.is_last_rank
        )
        for name in pp_keys[:module_list_idx]:
            if keep_prefix_modules:
                continue
            self._set_submodule(name, PPMissingLayer())
            self._register_missing_prefix(maybe_prefix("model", name))
        layers = self.model.get_submodule(module_list_name)
        self.start_layer, self.end_layer = get_pp_indices(
            len(layers), self.pp_group.rank_in_group, self.pp_group.world_size
        )
        for idx in range(len(layers)):
            if self.start_layer <= idx < self.end_layer:
                continue
            layers[idx] = self._make_pp_missing_layer(layers[idx])
            self._register_missing_prefix(
                maybe_prefix("model", f"{module_list_name}.{idx}")
            )
        for name in pp_keys[module_list_idx + 1 :]:
            if self.pp_group.is_last_rank:
                continue
            self._set_submodule(name, PPMissingLayer())
            self._register_missing_prefix(maybe_prefix("model", name))
