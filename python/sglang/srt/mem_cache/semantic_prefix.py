"""SemanticPrefixProvider — interface for approximate KV cache matching.

When an exact radix-tree lookup returns zero cached tokens, the provider
can supply an alternate set of token IDs whose KV is already resident in
the RadixCache.  The engine then reuses that donor KV, skipping full
prefill recomputation.

Typical use-cases
-----------------
* Semantic KV sharing (e.g. SemBlend): look up semantically similar
  documents already in the cache.
* Fuzzy prefix matching: tolerate small edits at prefix boundaries.
* RAG-aware caching: reuse cached KV for retrieved contexts.
* Topic-based KV sharing: share computation across requests with the
  same subject matter.

Usage
-----
Implement :class:`SemanticPrefixProvider` and register it with the
server's prefix cache::

    server.prefix_cache.set_semantic_provider(my_provider)

``on_prefix_miss`` is called synchronously inside the scheduler step
(inside ``RadixCache.match_prefix``), so it must be fast.  Heavy
embedding or similarity search should be done asynchronously and the
result staged before the call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SemanticPrefixResult:
    """Result returned by :meth:`SemanticPrefixProvider.on_prefix_miss`.

    Attributes
    ----------
    alternate_token_ids:
        Token IDs of the donor request whose KV is already resident in
        the RadixCache.  The cache will be queried with these tokens
        instead of the query's own tokens.
    num_cached_tokens:
        Hint for the expected number of cached tokens (used for logging
        only; the actual count is determined by the radix lookup).
    skip_insert:
        When ``True`` (the default) the query result is *not* inserted
        into the RadixCache under the query's own token IDs after the
        request completes, preventing cache pollution.
    metadata:
        Arbitrary application-defined data passed through to
        :meth:`on_request_cached` for bookkeeping.  Must be picklable
        when used in multi-process deployments.
    source_id:
        Optional label used in log messages.
    """

    alternate_token_ids: list[int]
    num_cached_tokens: int
    skip_insert: bool = True
    metadata: Any = None
    source_id: str = ""


class SemanticPrefixProvider(ABC):
    """Abstract base class for approximate / semantic KV cache matching.

    Subclasses implement :meth:`on_prefix_miss` to supply a donor request
    whenever the standard exact-match radix lookup returns zero hit tokens,
    and :meth:`on_request_cached` to update internal state after each
    request's KV is committed to the cache.

    The two optional lifecycle hooks (:meth:`on_init` and
    :meth:`on_shutdown`) allow the provider to integrate with SGLang's
    startup / teardown sequence.

    Thread-safety
    -------------
    :meth:`on_prefix_miss` and :meth:`on_request_cached` are called from
    the scheduler thread.  Implementations are responsible for their own
    locking where necessary.
    """

    @abstractmethod
    def on_prefix_miss(
        self,
        rid: str,
        token_ids: list[int],
    ) -> Optional[SemanticPrefixResult]:
        """Called when the exact radix-tree lookup returns zero hit tokens.

        The implementation should return a :class:`SemanticPrefixResult`
        whose ``alternate_token_ids`` are already resident in the
        RadixCache, or ``None`` to fall back to a normal cold prefill.

        Parameters
        ----------
        rid:
            SGLang request ID (unique per request).
        token_ids:
            Full prompt token IDs for the incoming request.

        Returns
        -------
        :class:`SemanticPrefixResult` or ``None``
        """
        ...

    @abstractmethod
    def on_request_cached(
        self,
        rid: str,
        token_ids: list[int],
    ) -> None:
        """Called after a request's KV is committed to the RadixCache.

        Implementations should use this to register the request as a
        potential future donor and update any per-request state.

        Parameters
        ----------
        rid:
            SGLang request ID of the cached request.
        token_ids:
            Full token IDs (prompt + generated output) of the cached
            request.
        """
        ...

    def on_init(self, model_config: Any = None) -> None:  # noqa: B027
        """Called once when the RadixCache initialises.

        Parameters
        ----------
        model_config:
            SGLang ``ModelConfig`` instance, or ``None`` when not
            available at init time.
        """

    def on_shutdown(self) -> None:  # noqa: B027
        """Called once when the server shuts down."""
