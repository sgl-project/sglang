from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.shared_hicache.source import ResolvedHostPage

logger = logging.getLogger(__name__)


class SharedHiCacheTarget:
    def __init__(self, *, tree_cache, metrics_collector=None):
        self.tree_cache = tree_cache
        self.metrics_collector = metrics_collector
        self.quarantined_device_indices: list[torch.Tensor] = []
        self.quarantined_tokens_by_backend: dict[str, int] = {}

    def _observe_quarantine(
        self,
        *,
        backend: str,
        reason: str,
        tokens: int,
        current_tokens: int,
    ) -> None:
        if self.metrics_collector is None:
            return
        self.metrics_collector.observe_shared_hicache_quarantine(
            backend=backend,
            reason=reason,
            tokens=tokens,
            current_tokens=current_tokens,
        )

    def alloc_device_indices(self, token_count: int) -> Optional[torch.Tensor]:
        allocator = self.tree_cache.cache_controller.mem_pool_device_allocator
        device_indices = allocator.alloc(token_count)
        if device_indices is None:
            logger.debug(
                "Shared HiCache skipped direct transfer; failed to allocate %d free device tokens",
                token_count,
            )
        return device_indices

    def free_device_indices(self, device_indices: Optional[torch.Tensor]) -> None:
        if device_indices is None:
            return
        self.tree_cache.cache_controller.mem_pool_device_allocator.free(device_indices)

    def device_indices_to_page_indices(
        self, device_indices: torch.Tensor
    ) -> Optional[list[int]]:
        page_size = self.tree_cache.page_size
        indices = device_indices.detach().cpu().numpy()
        if len(indices) == 0 or len(indices) % page_size != 0:
            return None

        page_rows = indices.reshape(-1, page_size)
        starts = page_rows[:, 0]
        offsets = np.arange(page_size, dtype=page_rows.dtype)
        if np.any(starts % page_size != 0) or np.any(
            page_rows != starts[:, None] + offsets[None, :]
        ):
            return None

        page_indices = starts // page_size
        if np.any(page_indices < 0) or np.any(page_indices > np.iinfo(np.int32).max):
            return None
        return page_indices.astype(np.int32, copy=False).tolist()

    def quarantine_device_indices(
        self, device_indices: torch.Tensor, reason: str, *, backend: str
    ) -> None:
        self.quarantined_device_indices.append(device_indices)
        backend_label = str(backend or "unknown")
        token_count = int(device_indices.numel())
        current_tokens = (
            int(self.quarantined_tokens_by_backend.get(backend_label, 0)) + token_count
        )
        self.quarantined_tokens_by_backend[backend_label] = current_tokens
        self._observe_quarantine(
            backend=backend_label,
            reason=reason,
            tokens=token_count,
            current_tokens=current_tokens,
        )
        logger.error(
            "Quarantining %d SharedHiCache target KV indices after indeterminate direct transfer: %s",
            token_count,
            reason,
        )

    def release_quarantined_device_indices(self) -> None:
        if not self.quarantined_device_indices:
            return
        quarantined = self.quarantined_device_indices
        tokens_by_backend = self.quarantined_tokens_by_backend
        self.quarantined_device_indices = []
        self.quarantined_tokens_by_backend = {}
        for device_indices in quarantined:
            self.free_device_indices(device_indices)
        for backend in tokens_by_backend:
            self._observe_quarantine(
                backend=backend,
                reason="released",
                tokens=0,
                current_tokens=0,
            )

    def insert_device_pages(
        self,
        req,
        pages: list[ResolvedHostPage],
        *,
        device_indices: torch.Tensor,
        start_block: int,
    ) -> int:
        page_size = self.tree_cache.page_size
        token_count = len(pages) * page_size
        token_start = start_block * page_size
        token_end = token_start + token_count
        allocated_tokens = len(device_indices)

        if token_end > len(req.fill_ids):
            token_count = ((len(req.fill_ids) - token_start) // page_size) * page_size
            pages = pages[: token_count // page_size]
            token_end = token_start + token_count

        if token_count <= 0:
            self.free_device_indices(device_indices)
            return 0

        if token_count < allocated_tokens:
            self.free_device_indices(device_indices[token_count:])
            device_indices = device_indices[:token_count]

        try:
            prefix_indices = getattr(
                req, "prefix_indices", torch.empty((0,), dtype=torch.int64)
            )
            if token_start != len(prefix_indices):
                logger.debug(
                    "Shared HiCache direct insert cannot attach suffix rid=%s token_start=%d prefix_indices=%d",
                    getattr(req, "rid", None),
                    token_start,
                    len(prefix_indices),
                )
                self.free_device_indices(device_indices)
                return 0

            key = RadixKey(
                req.fill_ids[:token_end],
                extra_key=req.extra_key,
                is_bigram=self.tree_cache.is_eagle,
            )
            if token_start > 0:
                prefix_indices = prefix_indices.to(
                    dtype=torch.int64, device=device_indices.device, copy=False
                )
                insert_value = torch.cat([prefix_indices, device_indices])
            else:
                insert_value = device_indices

            insert_shared_blocks = getattr(
                self.tree_cache, "insert_shared_hicache_device_blocks", None
            )
            if not callable(insert_shared_blocks):
                raise RuntimeError(
                    "tree cache does not support insert_shared_hicache_device_blocks"
                )
            result = insert_shared_blocks(key=key, value=insert_value)
            matched_length = result.prefix_len
            matched_new_tokens = min(max(0, matched_length - token_start), token_count)
            if matched_new_tokens > 0:
                self.free_device_indices(device_indices[:matched_new_tokens])
            staged_tokens = token_count - matched_new_tokens
            if staged_tokens <= 0:
                return 0
            return staged_tokens
        except Exception:
            self.free_device_indices(device_indices)
            raise
