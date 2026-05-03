# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class UGSessionHandle:
    """Opaque handle for a SRT-owned UG session.

    This is intentionally free of KV allocator details. Diffusion-side code may
    keep the handle, but only SRT-side runtime code can resolve it to model/KV
    state.
    """

    session_id: str
    anchor_request_id: str
    context_length: int
    context_version: int


@dataclass(slots=True)
class UGContextHandle:
    request_id: str
    token_count: int
    session: UGSessionHandle | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UGContextBundle:
    full: UGContextHandle
    text_cfg: UGContextHandle
    image_cfg: UGContextHandle
