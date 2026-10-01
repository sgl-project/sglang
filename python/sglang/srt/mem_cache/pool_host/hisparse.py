from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class HiSparseHostPoolMixin:
    def _round_up_to_page_size(self, size: int) -> int:
        return (size + self.page_size - 1) // self.page_size * self.page_size

    def alloc_page(self, num_pages: int) -> Optional[torch.Tensor]:
        return self.alloc(num_pages * self.page_size)

    def alloc_paged_token_slots(
        self,
        req_to_host_pool: torch.Tensor,
        req_to_host_pool_allocated_len: torch.Tensor,
        req_pool_idx: int,
        start_pos: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """Allocate request host slots by page and return token-granular slots."""
        device = req_to_host_pool.device
        if num_tokens <= 0:
            return torch.empty((0,), dtype=torch.int64, device=device)

        allocated_len = int(req_to_host_pool_allocated_len[req_pool_idx])
        end_pos = start_pos + num_tokens
        page_end = self._round_up_to_page_size(end_pos)
        assert start_pos <= allocated_len

        if page_end > allocated_len:
            num_new_pages = (page_end - allocated_len) // self.page_size
            host_locs = self.alloc_page(num_new_pages)
            if host_locs is None:
                logger.error(
                    "HiSparse: host mem pool alloc failed for %d host pages "
                    "(req_pool_idx=%d, start_pos=%d, num_tokens=%d)",
                    num_new_pages,
                    req_pool_idx,
                    start_pos,
                    num_tokens,
                )
                raise RuntimeError(
                    f"HiSparse host mem pool alloc failed for {num_new_pages} pages"
                )

            req_to_host_pool[req_pool_idx, allocated_len:page_end] = host_locs.to(
                device=device, non_blocking=True
            )
            req_to_host_pool_allocated_len[req_pool_idx] = page_end

        return req_to_host_pool[req_pool_idx, start_pos:end_pos]

    def allocated_host_indices(
        self,
        req_to_host_pool: torch.Tensor,
        req_pool_idx: int,
        allocated_len: int,
    ) -> torch.Tensor:
        allocated_len = int(allocated_len)
        host_len = min(
            self._round_up_to_page_size(allocated_len),
            req_to_host_pool.shape[1],
        )
        host_indices = req_to_host_pool[req_pool_idx, :host_len]
        return host_indices[host_indices >= 0]
