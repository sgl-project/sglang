"""Adapt PEFT checkpoint semantics to native diffusion LoRA layers."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

_ADAPTER_SLOT = re.compile(r"(\.lora_[AB])\.([^.]+)\.weight$")
_WRAPPER_PREFIXES = ("peft_model.base_model.model.", "base_model.model.")
_UNSUPPORTED_CONFIG_FIELDS = (
    "alora_invocation_tokens",
    "layer_replication",
    "modules_to_save",
    "target_parameters",
    "trainable_token_indices",
    "use_bdlora",
    "use_qalora",
)
_SAFETENSORS_ALPHA_KEYS = ("lora_alpha", "network_alpha", "alpha")
_NATIVE_LORA_A_SUFFIXES = (
    ".lora_A.weight",
    ".lora_down.weight",
    ".lora.down.weight",
)


def _has_unambiguous_global_alpha(file: Any) -> bool:
    """Reject bare mixed-rank files whose global alpha semantics are ambiguous."""
    keys = list(file.keys())
    if any(_ADAPTER_SLOT.search(name) is not None for name in keys):
        return True

    ranks = set()
    for name in keys:
        if not name.endswith(_NATIVE_LORA_A_SUFFIXES):
            continue
        shape = file.get_slice(name).get_shape()
        if len(shape) >= 2:
            ranks.add(shape[-2])
    return len(ranks) == 1


def _load_safetensors_lora_alpha(weight_path: str) -> int | None:
    if Path(weight_path).suffix.lower() != ".safetensors":
        return None
    with safe_open(weight_path, framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        if not _has_unambiguous_global_alpha(file):
            return None
    declared = []
    for key in _SAFETENSORS_ALPHA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"safetensors metadata {key!r} must be a positive integer"
            ) from error
        if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise ValueError(f"safetensors metadata {key!r} must be a positive integer")
        declared.append((key, int(numeric)))
    values = {value for _, value in declared}
    if len(values) > 1:
        raise ValueError(f"conflicting safetensors LoRA alpha metadata: {declared}")
    return declared[0][1] if declared else None


def load_peft_config(weight_path: str) -> dict[str, Any]:
    path = Path(weight_path).with_name("adapter_config.json")
    config = {}
    if path.is_file():
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("PEFT adapter_config.json must contain a JSON object")
    metadata_alpha = _load_safetensors_lora_alpha(weight_path)
    config_alpha = get_peft_lora_alpha(config)
    if (
        metadata_alpha is not None
        and config_alpha is not None
        and metadata_alpha != config_alpha
    ):
        raise ValueError(
            "adapter_config.json lora_alpha conflicts with safetensors metadata: "
            f"{config_alpha} != {metadata_alpha}"
        )
    if metadata_alpha is not None:
        config.setdefault("lora_alpha", metadata_alpha)
    return config


def get_peft_lora_alpha(config: Mapping[str, Any]) -> int | None:
    alpha = config.get("lora_alpha")
    if alpha is None:
        return None
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or alpha <= 0
        or isinstance(alpha, float)
        and not alpha.is_integer()
    ):
        raise ValueError("PEFT lora_alpha must be a positive integer")
    return int(alpha)


def normalize_peft_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove a uniform PEFT model wrapper and named adapter slot."""
    prefix = next(
        (
            prefix
            for prefix in _WRAPPER_PREFIXES
            if state_dict and all(name.startswith(prefix) for name in state_dict)
        ),
        "",
    )
    normalized: dict[str, torch.Tensor] = {}
    slots = set()
    has_bare_weights = False
    for name, tensor in state_dict.items():
        name = name.removeprefix(prefix)
        match = _ADAPTER_SLOT.search(name)
        if match is not None:
            slots.add(match.group(2))
        elif name.endswith((".lora_A.weight", ".lora_B.weight")):
            has_bare_weights = True
        target = _ADAPTER_SLOT.sub(r"\1.weight", name)
        if target in normalized:
            raise ValueError(
                "LoRA checkpoint contains multiple PEFT adapter slots for "
                f"the same tensor: {target!r}"
            )
        normalized[target] = tensor
    if len(slots) > 1:
        raise ValueError(
            f"LoRA checkpoint contains multiple PEFT adapter slots: {sorted(slots)}"
        )
    if slots and has_bare_weights:
        raise ValueError("LoRA checkpoint mixes named and unnamed PEFT adapter slots")
    return normalized


def _validate_peft_features(
    state_dict: Mapping[str, torch.Tensor], config: Mapping[str, Any]
) -> None:
    unsupported = {name for name in _UNSUPPORTED_CONFIG_FIELDS if config.get(name)}
    if config.get("bias") not in (None, "none"):
        unsupported.add("bias")
    unsupported.update(
        name for name in ("fan_in_fan_out", "lora_bias") if config.get(name, False)
    )
    if unsupported:
        raise ValueError(
            "PEFT adapter requires unsupported auxiliary/runtime features: "
            f"{sorted(unsupported)}"
        )
    if config.get("use_dora", False) or any(
        "lora_magnitude_vector" in name or "dora_scale" in name for name in state_dict
    ):
        raise ValueError(
            "DoRA adapters are not supported by the native diffusion LoRA layers"
        )
    auxiliary = [
        name
        for name in state_dict
        if any(
            marker in name
            for marker in ("lora_embedding_", "modules_to_save", "trainable_tokens")
        )
    ]
    if auxiliary:
        raise ValueError(
            f"PEFT adapter contains unsupported auxiliary tensors: {auxiliary[:8]}"
        )


def apply_peft_config(
    state_dict: dict[str, torch.Tensor], config: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Represent PEFT scaling variants with native LoRA tensors and alpha."""
    _validate_peft_features(state_dict, config)
    normalized = dict(state_dict)
    patterns = config.get("alpha_pattern", {})
    if not isinstance(patterns, dict):
        raise ValueError("PEFT alpha_pattern must be an object")
    compiled_patterns = []
    for pattern, value in patterns.items():
        if not isinstance(pattern, str):
            raise ValueError("PEFT alpha_pattern keys must be strings")
        try:
            expression = re.compile(rf"(.*\.)?({pattern})$")
        except re.error as error:
            raise ValueError(
                f"Invalid PEFT alpha_pattern expression {pattern!r}: {error}"
            ) from error
        alpha = get_peft_lora_alpha({"lora_alpha": value})
        if alpha is None:
            raise ValueError(
                f"PEFT alpha_pattern value for {pattern!r} must be a positive integer"
            )
        compiled_patterns.append((expression, alpha))

    suffix = ".lora_A.weight"
    for name in state_dict:
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)]
        alpha = next(
            (alpha for pattern, alpha in compiled_patterns if pattern.match(base)), None
        )
        if alpha is None:
            continue
        alpha_key = f"{base}.alpha"
        existing = normalized.get(alpha_key)
        if existing is not None and (
            existing.numel() != 1 or int(existing.item()) != alpha
        ):
            raise ValueError(f"PEFT alpha_pattern conflicts with tensor {alpha_key!r}")
        normalized[alpha_key] = torch.tensor(alpha)

    use_rslora = config.get("use_rslora", False)
    if not isinstance(use_rslora, bool):
        raise ValueError("PEFT use_rslora must be boolean")
    if not use_rslora:
        return normalized

    suffix = ".lora_B.weight"
    pairs = 0
    for name, weight in state_dict.items():
        if not name.endswith(suffix):
            continue
        lora_a = state_dict.get(f"{name[: -len(suffix)]}.lora_A.weight")
        if lora_a is None or lora_a.ndim < 2 or lora_a.shape[-2] <= 0:
            raise ValueError(f"RSLoRA weight {name!r} has no valid rank-bearing A")
        normalized[name] = weight * math.sqrt(lora_a.shape[-2])
        pairs += 1
    if not pairs:
        raise ValueError("PEFT use_rslora is true, but no LoRA A/B pairs were found")
    return normalized


def scale_fused_sections(
    a_parts: Mapping[int, torch.Tensor],
    b_parts: Mapping[int, torch.Tensor],
    alpha_parts: Mapping[int, torch.Tensor],
    default_alpha: int | None,
) -> list[torch.Tensor] | None:
    """Fold per-section PEFT alpha values into fused LoRA B weights."""
    if not alpha_parts:
        return None
    scaled = []
    for index in range(len(a_parts)):
        rank = a_parts[index].shape[0]
        alpha = float(
            alpha_parts[index].item()
            if index in alpha_parts
            else default_alpha
            if default_alpha is not None
            else rank
        )
        scale = alpha / rank
        weight = b_parts[index]
        scaled.append(
            weight if scale == 1.0 else (weight.float() * scale).to(weight.dtype)
        )
    return scaled
