# SPDX-License-Identifier: Apache-2.0
"""Whole-parameter adapter keys: ``.diff`` / ``.diff_b`` / ``.set_weight``.

Low-rank ``lora_A`` / ``lora_B`` stay on :class:`LoRAPipeline`. Distilled H3
students also ship exact additive deltas (RMSNorm, bias, small embedders) and,
for VSA, whole parameters the base checkpoint does not carry
(``to_gate_compress``). Those keys are applied here after the transformer
loads, using the same ``param_names_mapping`` as the checkpoint loader.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ADDITIVE_SUFFIXES: dict[str, str] = {".diff_b": ".bias", ".diff": ".weight"}
REPLACEMENT_SUFFIXES: dict[str, str] = {".set_weight": ".weight"}
_LOW_RANK_MARKERS = (
    ".lora_A",
    ".lora_B",
    ".lora_up",
    ".lora_down",
    ".lora_alpha",
    ".lora_rank",
    ".alpha",
    ".dora_scale",
)


def is_dense_payload_key(name: str) -> bool:
    return name.endswith(tuple(ADDITIVE_SUFFIXES) + tuple(REPLACEMENT_SUFFIXES))


def swap_peft_swiglu_fc1_lora_b(
    source_name: str, target_name: str, weight: torch.Tensor
) -> torch.Tensor:
    # FastVideo / PEFT H3 FFN is [value; gate]; native mlp.fc1 is [gate; value].
    # Other models' ff.net.0.proj (e.g. Flux) must not match.
    if (
        weight.dim() != 2
        or ".ff.net.0.proj.lora_B" not in source_name
        or not target_name.endswith(".mlp.fc1.lora_B")
    ):
        return weight
    value, gate = weight.chunk(2, dim=0)
    return torch.cat([gate, value], dim=0)


def adapter_has_dense_payload(keys: object) -> bool:
    return any(is_dense_payload_key(str(key)) for key in keys)


def _map_name(
    param_name: str,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None,
    source_key: str,
) -> str:
    param_name = param_name.replace("diffusion_model.", "")
    if param_names_mapping is None:
        return param_name
    mapped, merge_index, _ = param_names_mapping(param_name)
    if merge_index is not None:
        raise NotImplementedError(
            f"Adapter dense key {source_key} resolves to fused parameter {mapped}; "
            "whole-tensor payloads for fused parameters are not supported"
        )
    return mapped


def resolve_dense_key(
    key: str,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None,
) -> tuple[str, str] | None:
    """Map an adapter key to ``(model parameter name, "add" | "set")``."""
    if any(marker in key for marker in _LOW_RANK_MARKERS):
        return None
    for suffix, param_suffix in ADDITIVE_SUFFIXES.items():
        if key.endswith(suffix):
            return (
                _map_name(key[: -len(suffix)] + param_suffix, param_names_mapping, key),
                "add",
            )
    for suffix, param_suffix in REPLACEMENT_SUFFIXES.items():
        if key.endswith(suffix):
            return (
                _map_name(key[: -len(suffix)] + param_suffix, param_names_mapping, key),
                "set",
            )
    return None


def _find_parameter(module: nn.Module, target: str) -> torch.Tensor | None:
    names = [target]
    parts = target.split(".")
    if len(parts) >= 2:
        names.append(".".join([*parts[:-1], "base_layer", parts[-1]]))
    params = dict(module.named_parameters())
    for name in names:
        found = params.get(name)
        if found is not None:
            return found
    for name in names:
        try:
            return module.get_parameter(name)
        except AttributeError:
            continue
    return None


class AdapterDensePayload:
    """Adapter keys that address a whole parameter rather than a LoRA factor."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        additive: dict[str, str],
        replacement: dict[str, str],
        strength: float = 1.0,
    ) -> None:
        if not math.isfinite(strength):
            raise ValueError(f"Adapter strength must be finite, got {strength}")
        self._tensors = tensors
        # target parameter name -> adapter key
        self._additive = additive
        self._replacement = replacement
        self.strength = float(strength)
        self._applied: set[str] = set()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
        *,
        strength: float = 1.0,
    ) -> AdapterDensePayload | None:
        additive: dict[str, str] = {}
        replacement: dict[str, str] = {}
        kept: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            resolved = resolve_dense_key(key, param_names_mapping)
            if resolved is None:
                continue
            target, kind = resolved
            table = additive if kind == "add" else replacement
            if target in table:
                raise ValueError(
                    f"Adapter maps two dense keys onto parameter {target}; "
                    f"the second is {key}"
                )
            table[target] = key
            kept[key] = tensor
        if not additive and not replacement:
            return None
        logger.info(
            "Adapter dense payload: %d additive (.diff/.diff_b), %d replacement (.set_weight)",
            len(additive),
            len(replacement),
        )
        return cls(kept, additive, replacement, strength)

    @property
    def has_gate_compress(self) -> bool:
        return any("gate_compress" in name for name in self._replacement)

    def apply_to_module(self, module: nn.Module) -> tuple[int, list[str]]:
        """Add ``.diff`` / replace ``.set_weight`` on ``module``. Returns applied, unmatched."""
        unmatched: list[str] = []
        applied = 0
        for target, key in {**self._additive, **self._replacement}.items():
            if target in self._applied:
                continue
            param = _find_parameter(module, target)
            if param is None:
                unmatched.append(f"{key} -> {target}")
                continue
            payload = self._tensors[key]
            if payload.shape != tuple(param.shape):
                raise ValueError(
                    f"Adapter dense key {key} has shape {tuple(payload.shape)}, "
                    f"but {target} is {tuple(param.shape)}"
                )
            updated = payload.to(device=param.device, dtype=torch.float32)
            if target in self._additive:
                updated = param.detach().float() + updated * self.strength
            param.data.copy_(updated.to(dtype=param.dtype))
            self._applied.add(target)
            applied += 1
        if unmatched:
            logger.warning(
                "Adapter dense payload: %d applied, %d unmatched (first: %s)",
                applied,
                len(unmatched),
                unmatched[0],
            )
        else:
            logger.info("Adapter dense payload fully applied: %d parameters", applied)
        return applied, unmatched
