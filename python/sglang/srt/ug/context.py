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


@dataclass(frozen=True, slots=True)
class UGSRTRequestView:
    """Safe view of a materialized SRT request exposed to UG model adapters."""

    session: UGSessionHandle
    state: str
    request_id: str
    origin_input_len: int
    origin_input_ids: tuple[int, ...]
    output_ids: tuple[int, ...] = ()
    max_new_tokens: int = 0
    input_text: str = ""
    mm_offsets: tuple[tuple[int, int], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGSRTKVTokenBinding:
    """Opaque SRT request token locations for UG model-side cache adapters."""

    session_id: str
    request_id: str
    token_count: int
    token_indices: Any


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
