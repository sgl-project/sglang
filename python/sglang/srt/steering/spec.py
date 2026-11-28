# Copyright 2023-2026 SGLang Team
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
"""Per-request activation steering vector specifications.

Kept free of torch imports so request/IPC/scheduler code can import it without
pulling heavy runtime dependencies.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional, Tuple

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema


class SteeringScope(str, Enum):
    """Token phase scope for applying a steering vector."""

    ALL = "all"
    PREFILL = "prefill"
    DECODE = "decode"


_VALID_SCOPES = frozenset(scope.value for scope in SteeringScope)


class SteeringVectorSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Per-request specification for an additive residual steering vector.

    Fields:
        name: Identifier of a steering vector registered in SteeringVectorManager.
        scale: Multiplicative coefficient applied as ``h + scale * v``.
        scope: Token phase scope: ``"all"``, ``"prefill"``, or ``"decode"``.
        layers: Optional subset of decoder layer indices to restrict application
            to. When ``None``, applies to all layers present in the registered
            steering vector.
    """

    name: str
    scale: float = 1.0
    scope: str = SteeringScope.ALL.value
    layers: Optional[Tuple[int, ...]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)

    def affects_prefill(self) -> bool:
        """Whether this spec modifies prefill/extend hidden states and KV cache."""
        return self.scale != 0.0 and self.scope in (
            SteeringScope.ALL.value,
            SteeringScope.PREFILL.value,
        )

    def applies_to_phase(self, *, is_decode: bool) -> bool:
        """Whether this spec applies to a request in the given forward phase."""
        if self.scale == 0.0:
            return False
        if self.scope == SteeringScope.ALL.value:
            return True
        if self.scope == SteeringScope.DECODE.value:
            return is_decode
        if self.scope == SteeringScope.PREFILL.value:
            return not is_decode
        return False

    def radix_cache_key(self) -> Optional[str]:
        """Return a canonical radix-cache namespace key, or None if prefill is unsteered.

        Decode-only and zero-scale steering do not alter prompt KV activations, so
        they return ``None`` and can safely reuse unsteered prefix cache entries.
        """
        if not self.affects_prefill():
            return None
        layers_tag = (
            "all"
            if self.layers is None
            else ",".join(str(layer_idx) for layer_idx in self.layers)
        )
        return (
            f"steering:{self.name}:scale={self.scale!r}"
            f":scope={self.scope}:layers={layers_tag}"
        )


def _validate_and_normalize_spec(
    *,
    name: Any,
    scale: Any,
    scope: Any,
    layers: Any,
) -> SteeringVectorSpec:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("steering_vector.name must be a non-empty string.")
    normalized_name = name.strip()

    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(
            f"steering_vector.scale must be a finite number, got {type(scale).__name__}."
        )
    normalized_scale = float(scale)
    if not math.isfinite(normalized_scale):
        raise ValueError(
            f"steering_vector.scale must be finite, got {normalized_scale!r}."
        )

    if isinstance(scope, SteeringScope):
        normalized_scope = scope.value
    elif isinstance(scope, str):
        normalized_scope = scope
    else:
        raise ValueError(
            f"steering_vector.scope must be one of {sorted(_VALID_SCOPES)}, "
            f"got {scope!r}."
        )
    if normalized_scope not in _VALID_SCOPES:
        raise ValueError(
            f"steering_vector.scope must be one of {sorted(_VALID_SCOPES)}, "
            f"got {normalized_scope!r}."
        )

    normalized_layers: Optional[Tuple[int, ...]] = None
    if layers is not None:
        if isinstance(layers, (str, bytes)) or not isinstance(layers, (list, tuple)):
            raise ValueError(
                "steering_vector.layers must be a list or tuple of non-negative integers."
            )
        if len(layers) == 0:
            raise ValueError("steering_vector.layers must not be empty when specified.")
        parsed_layers = []
        for idx in layers:
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"steering_vector.layers entries must be non-negative integers, got {idx!r}."
                )
            parsed_layers.append(idx)
        normalized_layers = tuple(sorted(set(parsed_layers)))

    return SteeringVectorSpec(
        name=normalized_name,
        scale=normalized_scale,
        scope=normalized_scope,
        layers=normalized_layers,
    )


def decode_steering_vector_spec(value: Any) -> SteeringVectorSpec:
    """Convert a string shorthand, dict, or SteeringVectorSpec into a validated spec.

    Raises ``ValueError`` on any malformed input.
    """
    if isinstance(value, SteeringVectorSpec):
        return _validate_and_normalize_spec(
            name=value.name,
            scale=value.scale,
            scope=value.scope,
            layers=value.layers,
        )
    if isinstance(value, str):
        return _validate_and_normalize_spec(
            name=value,
            scale=1.0,
            scope=SteeringScope.ALL.value,
            layers=None,
        )
    if isinstance(value, dict):
        unknown_keys = set(value.keys()) - {"name", "scale", "scope", "layers"}
        if unknown_keys:
            raise ValueError(
                f"Unknown steering_vector field(s): {sorted(unknown_keys)}."
            )
        if "name" not in value:
            raise ValueError("steering_vector dict must include 'name'.")
        return _validate_and_normalize_spec(
            name=value["name"],
            scale=value.get("scale", 1.0),
            scope=value.get("scope", SteeringScope.ALL.value),
            layers=value.get("layers", None),
        )
    raise ValueError(
        "steering_vector must be a SteeringVectorSpec, dict, or string name, "
        f"got {type(value).__name__}."
    )
