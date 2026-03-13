"""
IOChain core abstractions.

An IOChain is an ordered pipeline of IOFilters. Each filter exposes two async
hooks — on_request (ingress) and on_response (egress) — and declares whether
it runs inline (blocking=True) or as a fire-and-forget background task
(blocking=False).

This module has no dependency on any SGLang serving or inference code and can
be imported standalone.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IOContext:
    """
    Carries request/response data through the filter chain for one request.

    Lifecycle
    ---------
    Ingress: raw_request and adapted_request populated; response is None.
    Egress:  response populated after inference.

    Filters may stash private per-request state in `metadata` using their
    class name as a key to avoid collisions.
    """

    request_id: str
    raw_request: Any        # OpenAI-protocol request object
    adapted_request: Any    # Internal GenerateReqInput (after tokenisation)
    response: Any = None    # Non-streaming response; None on ingress / for streaming
    metadata: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)


class IOFilter(ABC):
    """
    Abstract base for a single pipeline filter.

    Class attributes
    ----------------
    blocking : bool
        True  (default) — filter is awaited inline; the request/response waits.
                          Use for content checks, auth, rate-limiting.
        False           — filter is scheduled as asyncio.create_task and does
                          not delay the caller.
                          Use for telemetry, token counting, audit logging.
    """

    blocking: bool = True

    @abstractmethod
    async def on_request(self, ctx: IOContext) -> None:
        """Ingress hook — called after tokenisation, before inference."""

    @abstractmethod
    async def on_response(self, ctx: IOContext) -> None:
        """Egress hook — called after a non-streaming response is built."""


async def _run_safe(coro: Any, filter_name: str, phase: str) -> None:
    """Await a filter coroutine; log exceptions without re-raising."""
    try:
        await coro
    except Exception:
        logger.exception("IOChain %s filter %s raised", phase, filter_name)


class IOChain:
    """
    Ordered pipeline of IOFilters.

    Ingress: filters run in insertion order (0 → N).
    Egress:  filters run in reverse order (N → 0), mirroring the network-stack
             convention where the outermost layer wraps/unwraps symmetrically.

    Per-filter dispatch:
      blocking=True  → awaited before moving to the next filter.
      blocking=False → scheduled via asyncio.create_task; chain continues
                       immediately without waiting.
    """

    def __init__(self) -> None:
        self._filters: list[IOFilter] = []

    def add(self, f: IOFilter) -> "IOChain":
        """Append a filter; returns self for chaining: chain.add(A).add(B)."""
        self._filters.append(f)
        return self

    def make_context(self, raw_request: Any, adapted_request: Any) -> IOContext:
        return IOContext(
            request_id=uuid.uuid4().hex,
            raw_request=raw_request,
            adapted_request=adapted_request,
        )

    async def run_ingress(self, ctx: IOContext) -> None:
        for f in self._filters:
            coro = _run_safe(f.on_request(ctx), type(f).__name__, "ingress")
            if f.blocking:
                await coro
            else:
                asyncio.create_task(coro)

    async def run_egress(self, ctx: IOContext) -> None:
        for f in reversed(self._filters):
            coro = _run_safe(f.on_response(ctx), type(f).__name__, "egress")
            if f.blocking:
                await coro
            else:
                asyncio.create_task(coro)
