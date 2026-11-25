"""Runtime attach / detach of the unified cache external linker.

The linker backend and config are launch-time declarations: the server starts
attached, and a runtime re-attach rebuilds the same linker.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinker,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class LinkerAttachment:
    """Attach / detach the external linker of one ``UnifiedRadixCache``."""

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

    def launch(self) -> None:
        self._link(self._build_linker())

    def attach(self) -> str:
        if self._cache.linker is not None:
            return "External linker is already attached."
        cache_linker, error = None, None
        try:
            cache_linker = self._build_linker()
        except Exception as e:
            logger.exception("Failed to build the external linker.")
            error = e
        # The linker hooks run collectives, so every rank links or none does.
        if not self._all_ranks_succeeded(error is None):
            if cache_linker is not None:
                cache_linker.close()
            raise RuntimeError(
                "Failed to attach the external linker: "
                f"{error or 'failed on another rank'}"
            ) from error
        filled = self._cache.tree_core.backfill_missing_hash_values()
        self._link(cache_linker)
        logger.info(
            "Attached the external linker; hashed %d existing radix nodes.", filled
        )
        return "External linker attached."

    def detach(self) -> str:
        cache = self._cache
        message, error = "External linker is not attached.", None
        if cache.linker is not None:
            try:
                cache.linker.close()
                cache.linker = None
                self._publish_linker(None)
                message = "External linker detached."
            except Exception as e:
                logger.exception("Failed to detach the external linker.")
                error = e
        # Already-detached ranks join too. A failure on any rank leaves mixed
        # linker states, so every scheduler stops instead of resuming collectives.
        if not self._all_ranks_succeeded(error is None):
            raise RuntimeError(
                "External linker detach failed; scheduler must stop: "
                f"{error or 'failed on another rank'}"
            ) from error
        logger.info(message)
        return message

    def _link(self, cache_linker: UnifiedCacheLinker) -> None:
        self._cache.init_cache_linker(cache_linker)
        self._publish_linker(cache_linker)

    def _all_ranks_succeeded(self, succeeded: bool) -> bool:
        flag = torch.tensor([int(succeeded)], dtype=torch.int)
        self._cache._all_reduce_attn_groups(flag, torch.distributed.ReduceOp.MIN)
        return bool(flag.item())
