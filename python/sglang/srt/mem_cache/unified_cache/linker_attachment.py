from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_memory

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinker,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


def runtime_change_rejection() -> Optional[str]:
    if get_memory().enable_linker_mla_dedup:
        return (
            "--enable-linker-mla-dedup does not support runtime attach or detach "
            "of the external linker."
        )
    return None


class CacheLinkerAttachment:
    def __init__(
        self,
        cache: UnifiedRadixCache,
        *,
        build_linker: Callable[[], UnifiedCacheLinker],
        publish_linker: Callable[[Optional[UnifiedCacheLinker]], None],
    ):
        self._cache = cache
        self._build_linker = build_linker
        self._publish_linker = publish_linker

    def attach(self) -> str:
        cache = self._cache
        if cache.linker is not None:
            return "External linker is already attached."

        linker, error = None, None
        try:
            linker = self._build_linker()
        except Exception as e:
            logger.exception("Failed to build the external linker.")
            error = e
        # The linker hooks run collectives, so every rank links or none does.
        linked = torch.tensor([int(error is None)], dtype=torch.int)
        cache._all_reduce_attn_groups(linked, torch.distributed.ReduceOp.MIN)
        if not linked.item():
            if linker is not None:
                linker.close()
            raise RuntimeError(
                "Failed to attach the external linker: "
                f"{error or 'failed on another rank'}"
            ) from error

        try:
            filled = cache.tree_core.backfill_missing_hash_values()
            cache.init_cache_linker(linker)
        except BaseException:
            linker.close()
            raise
        self._publish_linker(linker)
        logger.info(
            "Attached the external linker; hashed %d existing radix nodes.", filled
        )
        return "External linker attached."

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
        cache = self._cache
        if cache.linker is None:
            return "External linker is not attached."
        cache.linker.close()
        cache.linker = None
        self._publish_linker(None)
        logger.info("Detached the external linker.")
        return "External linker detached."
