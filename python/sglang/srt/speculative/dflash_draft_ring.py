"""A per-request ring of draft KV rows for an all-sliding-window MLA DFLASH draft.

The draft attends only to the last ``window_left + 1`` positions, and the KV of
a position depends on that position's target hidden state alone. So instead of
one draft row per target slot, each request owns ``rows_per_req`` rows per DCP
rank, enough for the window plus a verify block:

    local row(req, pos) = 1 + req * rows_per_req + (pos // dcp) % rows_per_req
    loc(req, pos)       = local row * dcp + pos % dcp

``loc`` keeps the DCP owner rule of the shared pool (``loc % dcp`` is the owner
rank, the row is ``loc // dcp``), so the existing sharded writers apply as is.
Row 0 is the write sink. The window kernel computes the same rows directly
(``mla_window_sparse_indices(ring_rows_per_req=...)``).

A request's ring holds no positions from before its first prefill chunk (its
prefix-cache hit) or from a previous request in the same req pool slot;
``min_pos`` floors the window there.

Under PD disaggregation prefill sends a request's last window of ring rows
(``transfer_rows``, positionally matched like DeepSeek-V4's SWA ring) and its
``min_pos`` row; a decode rank keeps the positions it owns.
"""

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.kernels.ops.speculative.dflash_ring_locs import dflash_ring_block_locs
from sglang.srt.environ import envs


def draft_kv_ring_enabled() -> bool:
    """The flag as resolved at startup (``_resolve_dflash_draft_kv_ring``)."""
    return envs.SGLANG_DFLASH_DRAFT_KV_RING.get()


def draft_kv_ring_unsupported_reason(
    draft_hf_config,
    *,
    draft_backend: Optional[str],
    draft_window_size,
    dcp_size: int = 1,
) -> Optional[str]:
    """Why a draft KV ring cannot serve this DFLASH draft, or None if it can."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_layer_types,
        is_dflash_mla_draft_config,
    )
    from sglang.srt.speculative.draft_worker_common import (
        DFLASH_MLA_ONLY_DRAFT_BACKENDS,
    )
    from sglang.srt.utils import is_hip

    if not is_dflash_mla_draft_config(draft_hf_config):
        return "the draft is not MLA"
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return f"not every draft layer is sliding_attention ({layer_types})"
    if draft_backend not in DFLASH_MLA_ONLY_DRAFT_BACKENDS and not (
        # ROCm's triton draft backend reads the window by index under DCP
        draft_backend == "triton" and dcp_size > 1 and is_hip()
    ):
        return f"draft backend {draft_backend!r} has no window index path"
    if draft_window_size is not None:
        return "--speculative-draft-window-size keeps the compact draft table"
    return None


def draft_kv_ring_window_left(draft_hf_config) -> int:
    """The draft's sliding window (window_left); a ring needs every layer to slide."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
        get_dflash_layer_types,
    )

    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            "SGLANG_DFLASH_DRAFT_KV_RING needs a draft whose every layer is "
            f"sliding_attention; got layer_types={layer_types}."
        )
    return get_dflash_attention_sliding_window_size(draft_hf_config)


def draft_kv_ring_rows_per_req(
    *, window_left: int, block_size: int, dcp_size: int
) -> int:
    """Rows per request per DCP rank: a window plus one verify block, plus one
    row so the block never lands on a key the window still reads."""
    span = window_left + 1 + block_size
    return -(-span // dcp_size) + 1


def draft_kv_ring_pool_rows(
    *, req_pool_rows: int, rows_per_req: int, page_size: int
) -> int:
    rows = 1 + req_pool_rows * rows_per_req
    return -(-rows // page_size) * page_size


class DFlashDraftKVRing:
    def __init__(
        self,
        *,
        rows_per_req: int,
        dcp_size: int,
        req_pool_rows: int,
        device: torch.device,
        window_left: int = 0,
        dcp_rank: int = 0,
        pool=None,
        mem_pool: Optional[torch.cuda.MemPool] = None,
    ):
        self.rows_per_req = rows_per_req
        self.dcp_size = dcp_size
        self.window_left = window_left
        self.dcp_rank = dcp_rank
        self.pool = pool
        # Positions a ring holds without wrapping onto each other.
        self.span = rows_per_req * dcp_size
        # PD writes min_pos rows remotely; Mooncake's NVLink transport only
        # reaches memory from the KV pools' cuMemCreate pool.
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            self.min_pos = torch.zeros(req_pool_rows, dtype=torch.int32, device=device)
        self._floor_rids: Dict[int, str] = {}
        self._fused_block_locs = envs.SGLANG_OPT_DFLASH_RING_LOCS_FUSED.get()

    def locs(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Ring locs of ``positions``, ``req_pool_indices`` given per row."""
        positions = positions.to(torch.int64)
        local = (
            1
            + req_pool_indices.to(torch.int64) * self.rows_per_req
            + torch.remainder(positions // self.dcp_size, self.rows_per_req)
        )
        return local * self.dcp_size + torch.remainder(positions, self.dcp_size)

    def block_locs(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Ring locs of a draft block: ``block_size`` positions per request."""
        if not self._fused_block_locs:
            return self.locs(req_pool_indices.repeat_interleave(block_size), positions)

        return dflash_ring_block_locs(
            req_pool_indices,
            positions,
            block_size=block_size,
            rows_per_req=self.rows_per_req,
            dcp_size=self.dcp_size,
        )

    def extend_locs(
        self,
        *,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        extend_lens: torch.Tensor,
        seq_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Ring locs for one extend's tokens (request-major). A chunk longer than
        the ring wraps onto itself; only its last ``span`` positions, the ones a
        later window can read, keep a ring row, the rest go to the sink."""
        num_tokens = int(positions.numel())
        req_of_token = torch.repeat_interleave(
            req_pool_indices.to(torch.int64), extend_lens, output_size=num_tokens
        )
        end_of_token = torch.repeat_interleave(
            seq_ends.to(torch.int64), extend_lens, output_size=num_tokens
        )
        locs = self.locs(req_of_token, positions)
        keep = positions.to(torch.int64) >= end_of_token - self.span
        return torch.where(keep, locs, torch.zeros_like(locs))

    def record_floors(self, reqs, prefix_lens) -> None:
        """On a request's first extend, its prefix-cache hit length becomes the
        earliest position its window may read."""
        slots, floors = [], []
        for req, prefix_len in zip(reqs, prefix_lens):
            slot = req.kv.req_pool_idx
            if self._floor_rids.get(slot) != req.rid:
                self._floor_rids[slot] = req.rid
                slots.append(slot)
                floors.append(int(prefix_len))
        if slots:
            self.min_pos[
                torch.tensor(slots, dtype=torch.int64).to(
                    self.min_pos.device, non_blocking=True
                )
            ] = torch.tensor(floors, dtype=torch.int32).to(
                self.min_pos.device, non_blocking=True
            )

    def transfer_rows(self, req_pool_idx: int, seq_len: int) -> np.ndarray:
        """This rank's ring rows of the last window before ``seq_len``, in
        position order; row 0 (the sink) for a position another rank owns."""
        positions = np.arange(
            max(0, seq_len - self.window_left - 1), seq_len, dtype=np.int64
        )
        rows = (
            1
            + req_pool_idx * self.rows_per_req
            + (positions // self.dcp_size) % self.rows_per_req
        )
        rows[positions % self.dcp_size != self.dcp_rank] = 0
        return rows.astype(np.int32)

    def reset_floor(self, req_pool_idx: int, seq_len: int) -> None:
        """PD decode, at preallocation: no draft context until prefill's floor
        row lands, so a prefill that ships no ring leaves an empty window
        rather than a stale one."""
        self.min_pos[req_pool_idx].fill_(seq_len)

    def ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """The draft pool's buffers with one item per ring row."""
        ptrs, lens, item_lens = self.pool.get_contiguous_buf_infos()
        return ptrs, lens, [n // self.pool.page_size for n in item_lens]

    def floor_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """``min_pos`` with one item per req pool row."""
        return (
            [self.min_pos.data_ptr()],
            [self.min_pos.nbytes],
            [self.min_pos.element_size()],
        )


_active_ring: Optional[DFlashDraftKVRing] = None


def set_draft_kv_ring(ring: Optional[DFlashDraftKVRing]) -> None:
    global _active_ring
    _active_ring = ring


def get_draft_kv_ring() -> Optional[DFlashDraftKVRing]:
    """This process's draft KV ring, for PD state registration and payloads."""
    return _active_ring
