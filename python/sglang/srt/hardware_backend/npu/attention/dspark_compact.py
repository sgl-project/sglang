"""DSpark compact verification contracts for Ascend attention.

The device sequence lengths are committed prefix lengths. The DSpark caller
temporarily extends the CPU mirror, so adding a window to that mirror again
would double count it. Build both KV views from the device prefix instead.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class CompactVerifyMetadata(NamedTuple):
    query_ends: torch.Tensor
    kv_lens: torch.Tensor
    max_kv_len: int


def build_compact_verify_metadata(
    *, prefix_lens: torch.Tensor, layout, max_verify_len: int, num_tokens: int
) -> CompactVerifyMetadata:
    """Build cumulative TND Q ends and per-request (not cumulative) KV lengths.

    Eager-only: the host copy intentionally synchronizes once per verify batch,
    never once per attention layer. No graph padding or fabricated requests are
    accepted here; those require a separate graph/DP contract.
    """
    lens = layout.verify_lens_cpu
    if lens is None:
        lens = layout.verify_lens.cpu().tolist()
    bs = prefix_lens.numel()
    if len(lens) != bs or layout.verify_lens.numel() != bs or bs == 0:
        raise ValueError("NPU compact verify requires one length per live request.")
    if min(lens) < 1 or max(lens) > max_verify_len:
        raise ValueError("NPU compact verify lengths must be in [1, gamma + 1].")
    total = sum(lens)
    if layout.graph_num_tokens != total or num_tokens != total:
        raise ValueError("NPU compact eager verify does not support padded layouts.")
    if layout.total_verify_tokens not in (None, total):
        raise ValueError("NPU compact verify total does not match its lengths.")
    # This copy also prevents mutation of the schedule batch's CPU mirror.
    kv_lens = prefix_lens.to(torch.int32) + layout.verify_lens.to(torch.int32)
    kv_lens_cpu = kv_lens.cpu()
    return CompactVerifyMetadata(
        query_ends=layout.qo_indptr_device[1:].to(torch.int32),
        kv_lens=kv_lens,
        max_kv_len=int(kv_lens_cpu.max()),
    )


class NpuCompactGraphMetadata:
    """Pointer-stable DSA inputs, with a separate final padding-only request.

    Slots with no live request have zero Q/KV lengths. Tier padding belongs to
    the final ghost slot, never to the last real request: extending that request
    would shift right-aligned causal masking. Page 0 is the pool's reserved sink.
    Both capture and replay fill these buffers outside the model graph.
    """

    def __init__(self, *, num_slots, num_tokens, table_width, device):
        self.num_slots = num_slots
        self.num_tokens = num_tokens
        self.query_ends = torch.zeros(num_slots + 1, dtype=torch.int32, device=device)
        self.kv_lens = torch.zeros_like(self.query_ends)
        self.block_tables = torch.zeros(
            (num_slots + 1, table_width), dtype=torch.int32, device=device
        )

    def update(
        self, *, prefix_lens, req_pool_indices, req_to_token, verify_lens, page_size
    ):
        bs = verify_lens.numel()
        if not 0 < bs <= self.num_slots or prefix_lens.numel() < bs:
            raise ValueError("NPU compact metadata has too few request slots")
        ends = torch.cumsum(verify_lens, 0, dtype=torch.int32)
        self.query_ends[:bs].copy_(ends)
        self.query_ends[bs:-1].copy_(ends[-1].expand(self.num_slots - bs))
        self.query_ends[-1].fill_(self.num_tokens)
        self.kv_lens.zero_()
        # Capture layouts may contain zero-length slots after staging.
        self.kv_lens[:bs].copy_(
            torch.where(verify_lens > 0, prefix_lens[:bs] + verify_lens, 0)
        )
        self.kv_lens[-1].copy_((self.num_tokens - ends[-1]).clamp(min=1))
        self.block_tables.zero_()
        pages = req_to_token[req_pool_indices[:bs].long(), ::page_size] // page_size
        # Mask unallocated/stale pages, including inactive capture slots.
        page_ids = torch.arange(pages.shape[1], device=pages.device)
        pages = torch.where(
            page_ids[None, :] * page_size < self.kv_lens[:bs, None], pages, 0
        )
        self.block_tables[:bs, : pages.shape[1]].copy_(pages)
