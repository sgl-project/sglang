from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.runtime_context import get_memory

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinker,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


def _linker_class(storage_backend: str) -> type[UnifiedCacheLinker]:
    if storage_backend == "mooncake":
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            MooncakeDirectLinker,
        )

        return MooncakeDirectLinker
    if storage_backend == "mori":
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            UMBPDirectLinker,
        )

        return UMBPDirectLinker
    raise ValueError(
        f"Unknown unified cache external linker backend: {storage_backend!r}"
    )


def runtime_change_rejection(
    *, prefetch_policy: Optional[str] = None, write_policy: Optional[str] = None
) -> Optional[str]:
    if get_memory().enable_linker_mla_dedup:
        return (
            "--enable-linker-mla-dedup does not support runtime attach or detach "
            "of the external linker."
        )
    if prefetch_policy is not None or write_policy is not None:
        return "The external linker takes no prefetch or write policy."
    return None


class CacheLinkerAttachment:
    def __init__(
        self,
        cache: UnifiedRadixCache,
        *,
        server_args: Any,
        params: CacheInitParams,
        register_layer_transfer_counter: Callable[[Optional[object]], None],
        forward_stream: Optional[torch.cuda.Stream] = None,
    ):
        self._cache = cache
        self._server_args = server_args
        self._params = params
        self._kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self._register_layer_transfer_counter = register_layer_transfer_counter
        self._forward_stream = forward_stream
        self.storage_backend: Optional[str] = None
        self.storage_backend_extra_config: Optional[str] = None

    def attach(
        self, storage_backend: str, storage_backend_extra_config: Optional[str] = None
    ) -> str:
        if self.storage_backend is not None:
            if (storage_backend, storage_backend_extra_config) == (
                self.storage_backend,
                self.storage_backend_extra_config,
            ):
                return f"External linker already attached to {storage_backend!r}."
            raise RuntimeError(
                f"External linker is attached to {self.storage_backend!r}; "
                "detach it before attaching another configuration."
            )

        linker_cls = _linker_class(storage_backend)
        cache = self._cache
        linker, error = None, None
        try:
            linker = linker_cls(
                self._server_args,
                self._params,
                components=set(cache.components),
                storage_backend_extra_config=storage_backend_extra_config,
                forward_stream=self._forward_stream,
            )
        except Exception as e:
            logger.exception("Failed to build the %s external linker.", storage_backend)
            error = e
        # The linker hooks run collectives, so every rank links or none does.
        linked = torch.tensor([int(error is None)], dtype=torch.int)
        cache._all_reduce_attn_groups(linked, torch.distributed.ReduceOp.MIN)
        if not linked.item():
            if linker is not None:
                linker.close()
            raise RuntimeError(
                f"Failed to attach the {storage_backend!r} external linker: "
                f"{error or 'failed on another rank'}"
            ) from error

        try:
            filled = cache.tree_core.backfill_missing_hash_values()
            cache.init_cache_linker(linker)
        except BaseException:
            linker.close()
            raise
        counter = linker.layer_done_counter
        self._kvcache.register_layer_transfer_counter(counter)
        self._register_layer_transfer_counter(counter)
        self.storage_backend = storage_backend
        self.storage_backend_extra_config = storage_backend_extra_config
        logger.info(
            "Attached the %s external linker; hashed %d existing radix nodes.",
            storage_backend,
            filled,
        )
        return f"External linker attached to {storage_backend!r}."

    def detach(self, *, synchronize: bool = True) -> str:
        if not synchronize:
            # Process teardown must not wait for peers that may have exited.
            return self._detach_local()

        error = None
        message = ""
        try:
            message = self._detach_local()
        except Exception as e:
            logger.exception("Failed to detach the external linker.")
            error = e

        # Even an already-detached rank must participate. A failed close can
        # leave peers detached and this rank holding a partially stopped linker.
        # Never resume scheduler collectives in that state.
        detached = torch.tensor([int(error is None)], dtype=torch.int)
        self._cache._all_reduce_attn_groups(detached, torch.distributed.ReduceOp.MIN)
        if not detached.item():
            raise RuntimeError(
                "External linker detach failed; scheduler must stop: "
                f"{error or 'failed on another rank'}"
            ) from error
        return message

    def _detach_local(self) -> str:
        if self.storage_backend is None:
            return "External linker is not attached."
        self._cache.close_cache_linker()
        self._kvcache.register_layer_transfer_counter(None)
        self._register_layer_transfer_counter(None)
        storage_backend, self.storage_backend = self.storage_backend, None
        self.storage_backend_extra_config = None
        logger.info("Detached the %s external linker.", storage_backend)
        return f"External linker detached from {storage_backend!r}."
