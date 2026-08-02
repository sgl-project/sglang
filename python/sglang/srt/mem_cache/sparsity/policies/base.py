"""Policy interface for post-hoc KV-cache sparsity."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sglang.srt.mem_cache.sparsity.contracts import (
    RequestIdentity,
    SelectionContext,
    SelectionResult,
    SparsityCapabilities,
)


class SparsityPolicy(ABC):
    """Selects logical KV entries without mutating backend metadata or pools."""

    @property
    @abstractmethod
    def capabilities(self) -> SparsityCapabilities: ...

    @abstractmethod
    def select(self, context: SelectionContext) -> SelectionResult: ...

    def on_request_begin(self, identity: RequestIdentity) -> None:
        """Initialize policy state for a generation-aware request identity."""

    def on_request_end(self, identity: RequestIdentity) -> None:
        """Release policy state for a generation-aware request identity."""

    def begin_forward(self, context: SelectionContext) -> None:
        """Prepare policy state once for a new prefill or decode forward."""

    def on_attention_complete(self, context: SelectionContext) -> None:
        """Optionally update an incremental policy representation."""
