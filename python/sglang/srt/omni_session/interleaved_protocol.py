# SPDX-License-Identifier: Apache-2.0
"""Small shared types used by the SRT-backed omni middle layer."""

from dataclasses import dataclass, field
from typing import Any, Literal

GenerationKind = str
DEFAULT_OMNI_TEXT_MAX_NEW_TOKENS = 128


@dataclass(frozen=True, slots=True)
class GeneratedSegmentResult:
    type: Literal["image"]
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    commit_image: Any | None = None
