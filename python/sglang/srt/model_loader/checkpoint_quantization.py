# SPDX-License-Identifier: Apache-2.0

"""Read quantization metadata declared by Hugging Face configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

from transformers import PretrainedConfig

__all__ = [
    "CheckpointQuantSpec",
    "QuantMetadataSource",
    "resolve_checkpoint_quant_spec",
]


QuantMetadataSource: TypeAlias = Literal[
    "quantization_config",
    "text_config.quantization_config",
    "compression_config",
]
ConfigMapping: TypeAlias = Mapping[str, Any] | PretrainedConfig


@dataclass(slots=True)
class CheckpointQuantSpec:
    """Quantization metadata declared by a checkpoint configuration.

    ``declared_method`` preserves ``quant_method`` verbatim and is never inferred
    from backend-specific fields. This intentionally contains no runtime
    quantization classes, model construction, or layer hierarchy.
    """

    declared_method: str | None
    config: dict[str, Any]
    source: QuantMetadataSource


def _as_config_mapping(value: ConfigMapping, source: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, PretrainedConfig):
        return value.to_dict()
    raise TypeError(
        f"{source} must be a mapping or transformers.PretrainedConfig, "
        f"got {type(value).__name__}"
    )


def _select_hf_quant_metadata(
    hf_config: ConfigMapping,
) -> tuple[QuantMetadataSource, object] | None:
    config = _as_config_mapping(hf_config, "HF config")
    value = config.get("quantization_config")
    if value is not None:
        return "quantization_config", value

    text_config = config.get("text_config")
    if text_config is not None:
        text_config_mapping = _as_config_mapping(text_config, "text_config")
        value = text_config_mapping.get("quantization_config")
        if value is not None:
            return "text_config.quantization_config", value

    value = config.get("compression_config")
    if value is not None:
        return "compression_config", value

    return None


def resolve_checkpoint_quant_spec(
    hf_config: ConfigMapping,
) -> CheckpointQuantSpec | None:
    """Resolve quantization metadata from an HF configuration.

    The lookup order matches both serving runtimes: top-level
    ``quantization_config``, the text sub-config used by some multimodal
    checkpoints, then ``compression_config``. Returned metadata is deep-copied
    so callers can attach runtime-only fields without mutating the source config.
    """

    selected = _select_hf_quant_metadata(hf_config)
    if selected is None:
        return None

    source, value = selected
    config = _as_config_mapping(value, source)
    copied_config = deepcopy(dict(config))
    declared_method = copied_config.get("quant_method")
    return CheckpointQuantSpec(
        declared_method=(declared_method if isinstance(declared_method, str) else None),
        config=copied_config,
        source=source,
    )
