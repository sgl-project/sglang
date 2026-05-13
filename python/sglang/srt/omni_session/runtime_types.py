# SPDX-License-Identifier: Apache-2.0
"""Opaque handles exchanged across SRT-owned omni_session boundaries"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class OmniSessionHandle:
    """Opaque handle for a SRT-owned omni session.

    This is intentionally free of KV allocator details. Diffusion-side code may
    keep the handle, but only SRT-side runtime code can resolve it to model/KV
    state.
    """

    session_id: str
    anchor_request_id: str
    context_length: int
    context_version: int


@dataclass(frozen=True, slots=True)
class OmniSRTKVTokenBinding:
    """Opaque SRT request token locations for omni model-side cache policies"""

    session_id: str
    request_id: str
    token_count: int
    token_indices: Any
    position_count: int | None = None


@dataclass(slots=True)
class OmniContextHandle:
    """Reference to one prepared AR-side context stream"""

    request_id: str
    token_count: int
    session: OmniSessionHandle | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OmniContextBundle:
    """Main and CFG condition path contexts prepared from the same user turn"""

    full: OmniContextHandle
    text_cfg: OmniContextHandle
    image_cfg: OmniContextHandle
