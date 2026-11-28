"""Runtime attach / detach of the unified cache external linker.

The linker backend and config are launch-time declarations: the server starts
attached, and a runtime re-attach rebuilds the same linker.
"""

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


class _LinkerRollbackError(RuntimeError):
    """An incomplete rollback must stop the scheduler on every rank."""


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

    def update(self, attach: bool) -> tuple[bool, str]:
        if get_memory().enable_linker_mla_dedup:
            return False, (
                "--enable-linker-mla-dedup does not support runtime attach or detach."
            )
        if not attach:
            # A failed detach is fatal on every rank; let it terminate the scheduler.
            return True, self.detach()
        try:
            return True, self.attach()
        except _LinkerRollbackError:
            raise
        except Exception as e:
            logger.exception("Failed to attach the external linker.")
            return False, str(e)

    def attach(self) -> str:
        if self._cache.linker is not None:
            return "External linker is already attached."
        threshold = self._cache.write_through_threshold
        cache_linker, error = None, None
        try:
            cache_linker = self._build_linker()
        except Exception as e:
            error = e
        # The linker hooks run collectives, so every rank links or none does.
        if self._all_ranks_succeeded(error is None):
            try:
                filled = self._cache.tree_core.backfill_missing_hash_values()
                self._link(cache_linker)
            except Exception as e:
                error = e
            if self._all_ranks_succeeded(error is None):
                logger.info(
                    "Attached the external linker; hashed %d existing radix nodes.",
                    filled,
                )
                return "External linker attached."

        # No requests run during attachment, so the new backend has no queued
        # transfers. Close it even if wrapper construction failed before assignment.
        rollback_error = None
        try:
            if cache_linker is not None:
                cache_linker.close()
            self._cache.linker = None
            self._cache.tree_core.enable_external_cache_linker = False
            self._cache.write_through_threshold = threshold
            self._publish_linker(None)
        except Exception as e:
            rollback_error = e
        if not self._all_ranks_succeeded(rollback_error is None):
            raise _LinkerRollbackError(
                "External linker attach rollback failed; scheduler must stop: "
                f"{rollback_error or 'failed on another rank'}"
            ) from rollback_error
        raise RuntimeError(
            f"Failed to attach the external linker: {error or 'failed on another rank'}"
        ) from error

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
