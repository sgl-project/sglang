"""Full DFlash draft KV transfer in the draft pool's logical token-id domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class DFlashDraftTransfer(msgspec.Struct, frozen=True):
    """Full DFlash draft KV sent as its own PD state component.

    Under DCP the target KV travels as rank-local rows, while the full draft
    pool is replicated and indexed by the target allocator's logical token ids.
    """

    pool: MHATokenToKVPool
    # Draft tokens decode attends to; None transfers the whole prompt.
    window: Optional[int]

    def buffer_infos(self, wire_page_size: int) -> Tuple[List, List, List]:
        """Draft buffers in target wire-page units; draft allocation pages are wider."""
        if wire_page_size <= 0 or self.pool.page_size % wire_page_size:
            raise ValueError("Draft allocation pages must contain whole transfer pages")
        ptrs, lengths, item_lengths = self.pool.get_contiguous_buf_infos()
        ratio = self.pool.page_size // wire_page_size
        if any(length % ratio for length in item_lengths):
            raise ValueError("Draft buffer page bytes must divide into transfer pages")
        return ptrs, lengths, [length // ratio for length in item_lengths]

    def page_indices(
        self,
        *,
        req_to_token: torch.Tensor,
        prefix_len: int,
        seq_len: int,
        wire_page_size: int,
    ) -> np.ndarray:
        """Wire pages of one request's draft KV past the decode prefix."""
        # The decode prefix already holds its draft KV, like the SWA payload.
        start = prefix_len
        if self.window is not None:
            start = max(prefix_len, seq_len - self.window)
        start = start // wire_page_size * wire_page_size
        return kv_to_page_indices(req_to_token[start:seq_len], wire_page_size)


def resolve_dflash_draft_transfer(
    *,
    allocator: BaseTokenToKVPoolAllocator,
    draft_worker: Optional[DFlashWorkerV2],
    transfer_backend: TransferBackend,
) -> Optional[DFlashDraftTransfer]:
    # Without DCP the target's KV indices are logical too, so the draft rides
    # the target KV entries (num_draft_entries) instead.
    if (
        allocator.full_draft_kv_pool is None
        or transfer_backend != TransferBackend.NIXL
        or get_parallel().attn_dcp_size == 1
    ):
        return None
    return DFlashDraftTransfer(
        pool=allocator.full_draft_kv_pool,
        window=draft_transfer_window(
            draft_worker.draft_model_runner.model_config.hf_config
        ),
    )


def draft_transfer_window(draft_hf_config) -> Optional[int]:
    """Draft KV tokens decode needs, or None unless every draft layer slides."""
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return None
    # The helper returns window_left; the HF window also counts the current token.
    return get_dflash_attention_sliding_window_size(draft_hf_config) + 1
