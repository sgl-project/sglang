# Copyright 2024 SGLang Team
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

"""Utility helpers for automatically loading model weights."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from sglang.srt.layers.utils import PPMissingLayer

from .weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

WeightsMapping = Mapping[str, str | None]


@dataclass
class WeightsMapper:
    """Map weight names before loading them into the target module."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> str | None:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None
                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None
                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None
                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (mapped_name, tensor)
            for name, tensor in weights
            if (mapped_name := self._map_name(name)) is not None
        )


class AutoWeightsLoader:
    """Helper that walks a module tree while streaming weight tensors."""

    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = (skip_substrs or []) + self.ROTARY_EMBEDS_UNUSED_WEIGHTS
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = (
            (weight_name.split(".", 1), weight_tensor)
            for weight_name, weight_tensor in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                (
                    ("" if len(parts) == 1 else parts[1], tensor)
                    for parts, tensor in group
                ),
            )

    @staticmethod
    def _qualname(prefix: str, rest: str) -> str:
        if not prefix:
            return rest
        if not rest:
            return prefix
        return f"{prefix}.{rest}"

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        prefix_match = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        suffix_match = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(prefix_match) or any(suffix_match)

    def _load_param(
        self,
        prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_tensor in weights:
            qualname = self._qualname(prefix, weight_name)

            if self._can_skip(qualname):
                logger.debug("Skipping weight %s", qualname)
                continue

            if weight_name:
                if self._can_ignore_unexpected(qualname):
                    logger.debug("Ignoring unexpected nested weight %s", qualname)
                    continue
                msg = (
                    f"Attempted to load nested weight '{qualname}' into parameter "
                    f"'{prefix}'"
                )
                raise ValueError(msg)

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_tensor)
            logger.debug("Loaded weight %s with shape %s", qualname, tuple(param.shape))
            yield qualname

    def _extend_non_param_tensors(
        self, module: nn.Module, params: dict[str, torch.Tensor]
    ) -> None:
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            stats = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                params.setdefault(stat_name, stats[stat_name])

    def _load_module(
        self,
        prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        if module is not self.module:
            custom_loader = getattr(module, "load_weights", None)
            if callable(custom_loader):
                loaded = custom_loader(weights)
                if loaded is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    for name in loaded:
                        yield self._qualname(prefix, name)

        child_modules = dict(module.named_children())
        direct_params = dict(module.named_parameters(recurse=False))
        self._extend_non_param_tensors(module, direct_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            qualname = self._qualname(prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(f"{qualname}."):
                    logger.debug("Skipping module %s", qualname)
                    continue
                yield from self._load_module(
                    qualname, child_modules[child_prefix], child_weights
                )
                continue

            if child_prefix in direct_params:
                if self._can_skip(qualname):
                    logger.debug("Skipping param %s", qualname)
                    continue
                yield from self._load_param(
                    qualname, direct_params[child_prefix], child_weights
                )
                continue

            if self._can_skip(f"{qualname}.") or self._can_skip(qualname):
                logger.debug("Skipping missing %s", qualname)
                continue

            if self._can_ignore_unexpected(
                f"{qualname}."
            ) or self._can_ignore_unexpected(qualname):
                logger.debug("Ignoring unexpected %s", qualname)
                continue

            raise ValueError(
                f"There is no module or parameter named '{qualname}' in "
                f"{type(self.module).__name__}"
            )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)

        filtered = (
            (name, tensor) for name, tensor in weights if not self._can_skip(name)
        )
        return set(self._load_module("", self.module, filtered))
