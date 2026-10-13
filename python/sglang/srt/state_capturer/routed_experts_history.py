from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.state_capturer.base import BaseTopkCapturer


class RoutedExpertsHistory:
    """Request-owned routing rows, independent of recyclable KV-cache slots.

    Snapshot only at retraction boundaries, after the routes for accepted output
    tokens have been finalized. Re-prefill may overwrite the entire slot cache,
    but must not change the routes that produced previously accepted tokens.
    Keep incremental chunks so repeated retractions do not copy the
    full prefix each time. The final response concatenates them once.

    This preserves historical expert choices, not historical hidden/KV states:
    a response spanning weight updates is still a mixed-policy response.
    """

    def __init__(self, start_len: int):
        self.start_len = start_len
        self.end_len = start_len
        self.chunks: list[torch.Tensor] = []

    def extend(
        self,
        capturer: "BaseTopkCapturer",
        req_to_token_pool: "ReqToTokenPool",
        req_pool_idx: int,
        end_len: int,
    ) -> None:
        if end_len <= self.end_len:
            # Includes a second retraction during re-prefill, before any new
            # output is accepted. Its uncomputed prefix slots must not be read.
            return
        rows = capturer.get_topk(
            req_pool_idx=req_pool_idx,
            seqlen=end_len + 1,
            req_to_token_pool=req_to_token_pool,
            start_len=self.end_len,
        )
        if rows.shape[0] != end_len - self.end_len:
            raise RuntimeError(
                "Cannot preserve routed experts: "
                f"expected {end_len - self.end_len} rows, got {rows.shape[0]}"
            )
        # get_topk uses advanced indexing and returns owned CPU storage, not a
        # view into the slot cache. Do not add another full-prefix copy here.
        self.chunks.append(rows)
        self.end_len = end_len

    def collect(
        self,
        capturer: "BaseTopkCapturer",
        req_to_token_pool: "ReqToTokenPool",
        req_pool_idx: int,
        end_len: int,
    ) -> torch.Tensor:
        self.extend(capturer, req_to_token_pool, req_pool_idx, end_len)
        remaining = max(0, end_len - self.start_len)
        parts = []
        for chunk in self.chunks:
            if remaining == 0:
                break
            parts.append(chunk[:remaining])
            remaining -= parts[-1].shape[0]
        if not parts:
            return torch.empty(
                (0, capturer.num_layers, capturer.topk_size), dtype=torch.int32
            )
        # Stop-token truncation can shorten the visible output even after a
        # snapshot. Concatenate only visible rows, not the truncated suffix.
        return torch.cat(parts, dim=0)
