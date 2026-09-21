# SPDX-License-Identifier: Apache-2.0
"""Scheduler contract for externally packaged prefix-cache connectors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, CacheRequestHandle

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.kv_transfer_config import KVTransferConfig
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext


class BaseKVConnector(BasePrefixCache):
    """A BasePrefixCache with an external KV transfer lifecycle.

    A provider may reuse UnifiedRadixCache through inheritance or composition.
    The configured class is constructed with ``(context, config)`` once in each
    scheduler worker, after KV pools and process groups exist. It must implement
    the BasePrefixCache contract as well as the hooks below. This is a SGLang
    interface, not the vLLM KVConnectorBase interface.
    """

    @classmethod
    @abstractmethod
    def validate_config(
        cls, context: TreeCacheBuildContext, config: KVTransferConfig
    ) -> None:
        """Reject unsupported model, layout, parallelism, and role combinations.

        Called before construction. Do not allocate resources here. In
        particular, check speculative decoding, SWA/Mamba, DCP/DP, and device
        support against the provider's actual transfer implementation.
        """

    @abstractmethod
    def prefetch_request(self, req: Req) -> None:
        """Start lookup for this request attempt without allocating GPU slots.

        The scheduler has refreshed req's local prefix first. Preserve cache
        salt/extra key isolation. In producer-only mode this should do no I/O.
        Repeated calls for one CacheRequestHandle must be safe.
        """

    @abstractmethod
    def check_prefetch_progress(self, handle: CacheRequestHandle) -> bool:
        """Whether this attempt may proceed to matching and admission.

        A miss or failed lookup should become ready for ordinary prefill.
        An external hit is surfaced by match_prefix; init_load_back runs only
        after PrefillAdder has admitted it. Returning None from init_load_back
        defers the request until a subsequent scheduling iteration.
        """

    def pop_prefetch_loaded_span(
        self, handle: CacheRequestHandle
    ) -> tuple[int, Optional[int]]:
        """Return actually loaded tokens and their absolute prefix start once.

        A lookup hit alone must not be counted as transferred KV. Providers
        loading at admission may instead account for storage hits there.
        """
        return 0, None

    @abstractmethod
    def check_hicache_events(self) -> None:
        """Poll transfers, including when no requests are runnable.

        Any rank-wide completion decisions must use consistent collective
        ordering. Keep GPU slots and producer events alive until I/O completes.
        """

    @abstractmethod
    def ready_to_load_host_cache(self) -> int:
        """Establish transfer/forward ordering before the admitted batch runs.

        Return the layer-counter consumer index when using SGLang's layer
        transfer counters, or -1 when the provider establishes ordering itself.
        """

    @abstractmethod
    def has_pending_cache_operations(self) -> bool:
        """Include every transfer retaining KV pages or other live resources.

        The scheduler will not flush/reset or report fully idle until false.
        """

    def clear_storage_backend(self) -> bool:
        """Clear external storage, returning false if unsupported.

        Providers implementing this must synchronize with their pending work.
        This is separate from reset(), which clears the local prefix cache.
        """
        return False

    @abstractmethod
    def release_host_resources(self) -> None:
        """Idempotently drain and close provider resources on worker shutdown.

        Request completion/abort must also implement BasePrefixCache.finish and
        release_aborted_request as needed; shutdown is not per-request cleanup.
        """
