"""A per-request ring of draft KV rows for an all-sliding-window MLA DFLASH draft.

The draft reads only the last ``window_left + 1`` positions, and a position's
draft KV depends on its own target hidden state alone, so each request keeps
``rows_per_req`` rows per DCP rank, the window plus a verify block:

    local row(req, pos) = 1 + req * rows_per_req + (pos // dcp) % rows_per_req
    loc(req, pos)       = local row * dcp + pos % dcp

``loc % dcp`` stays the owner rank, so the draft pool's sharded writers work
unchanged. Row 0 takes the writes no window reads. ``DFlashDraftKVRing.locs``,
``dflash_ring_block_locs`` and ``mla_window_sparse_indices(ring_rows_per_req=)``
compute this layout and change together.

``min_pos`` floors each request's window at its prefix-cache hit, whose draft
KV the ring never had.

Under PD disaggregation prefill sends a request's last window of ring rows
(``transfer_rows``, positionally matched like the SWA_RING state) and its
``min_pos`` row; a decode rank keeps the positions it owns.
"""

from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch

from sglang.kernels.ops.speculative.dflash_ring_locs import dflash_ring_block_locs
from sglang.srt.arg_groups.choices import DFLASH_MLA_ONLY_DRAFT_BACKENDS
from sglang.srt.runtime_context import get_parallel, get_spec


def draft_kv_ring_unsupported_reason(
    load_draft_hf_config: Callable[[], Any],
    *,
    draft_backend: Optional[str],
    disaggregation_mode: str = "null",
    disaggregation_transfer_backend: str = "mooncake",
    disaggregation_decode_host_receive_threshold: float = 1.0,
    enable_unified_memory: bool = False,
    dcp_size: int = 1,
) -> Optional[str]:
    """Why a ring cannot serve this draft, or None. Loads the draft config only
    once the launch flags allow a ring."""
    if disaggregation_mode != "null" and disaggregation_transfer_backend != "mooncake":
        return (
            f"the {disaggregation_transfer_backend} PD transfer backend does not "
            "send the ring"
        )
    if disaggregation_decode_host_receive_threshold < 1:
        return "decode host KV buffering receives only SWA state"
    if enable_unified_memory:
        return "--enable-unified-memory places the draft KV in the target's pool"
    if draft_backend not in DFLASH_MLA_ONLY_DRAFT_BACKENDS:
        return f"draft backend {draft_backend!r} has no window index path"

    from sglang.srt.configs.model_config import mla_attention_dcp_size
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_layer_types,
        parse_dflash_draft_config,
    )

    draft_hf_config = load_draft_hf_config()
    if not parse_dflash_draft_config(draft_hf_config=draft_hf_config).is_mla:
        return "the draft is not MLA"
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return f"not every draft layer is sliding_attention ({layer_types})"
    if dcp_size > 1 and (
        mla_attention_dcp_size(draft_hf_config, dcp_size=dcp_size, is_nextn=True)
        != dcp_size
    ):
        return "the draft does not run DCP, so its rows are not DCP-collapsed"
    return None


def draft_kv_ring_rows_per_req(
    *, window_left: int, block_size: int, dcp_size: int
) -> int:
    """Rows per request per DCP rank: a window plus one verify block, plus one
    row so the block never lands on a key the window still reads."""
    span = window_left + 1 + block_size
    return -(-span // dcp_size) + 1


def draft_kv_ring_rows_per_req_from_config(draft_hf_config) -> int:
    """For this launch's draft, block size and DCP; the draft pool's size and
    the ring's locs both take it from here."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
    )

    return draft_kv_ring_rows_per_req(
        window_left=get_dflash_attention_sliding_window_size(draft_hf_config),
        block_size=int(get_spec().speculative_num_draft_tokens),
        dcp_size=get_parallel().attn_dcp_size,
    )


def draft_kv_ring_pool_rows(
    *, req_to_token_rows: int, rows_per_req: int, page_size: int
) -> int:
    rows = 1 + req_to_token_rows * rows_per_req
    return -(-rows // page_size) * page_size


def draft_kv_ring_pool_rows_from_config(
    draft_hf_config, *, req_to_token_rows: int, page_size: int
) -> int:
    """The target's budget charge and the draft's allocation both size the
    ring pool here."""
    return draft_kv_ring_pool_rows(
        req_to_token_rows=req_to_token_rows,
        rows_per_req=draft_kv_ring_rows_per_req_from_config(draft_hf_config),
        page_size=page_size,
    )


class DFlashDraftKVRing:
    def __init__(
        self,
        *,
        rows_per_req: int,
        dcp_size: int,
        req_to_token_rows: int,
        device: torch.device,
        window_left: int,
        dcp_rank: int,
        pool,
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
        mem_pool = pool.maybe_get_custom_mem_pool()
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            self.min_pos = torch.zeros(
                req_to_token_rows, dtype=torch.int32, device=device
            )
        # The req_generation each slot's floor was recorded at.
        self._floor_generation = torch.full((req_to_token_rows,), -1, dtype=torch.int64)

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
        """Ring locs of a draft block: ``block_size`` positions per request, in
        one launch (bitwise equal to ``locs`` on the repeated indices)."""
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
        """Ring locs for one extend's tokens (request-major); a chunk longer than
        the ring keeps only its last ``span`` positions, the rest go to row 0."""
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

    def record_floors(self, reqs, prefix_lens, req_generation: torch.Tensor) -> None:
        """Sets a slot's floor on the first extend after the slot is allocated;
        ``req_generation`` is the req pool's per-slot allocation count."""
        slots = torch.tensor([req.kv.req_pool_idx for req in reqs], dtype=torch.int64)
        generation = req_generation[slots]
        fresh = self._floor_generation[slots] != generation
        if not bool(fresh.any()):
            return
        slots = slots[fresh]
        self._floor_generation[slots] = generation[fresh]
        floors = torch.tensor([int(p) for p in prefix_lens], dtype=torch.int32)[fresh]
        device = self.min_pos.device
        self.min_pos[slots.to(device, non_blocking=True)] = floors.to(
            device, non_blocking=True
        )

    def transfer_rows(self, req_pool_idx: int, seq_len: int) -> np.ndarray:
        """This rank's ring rows of the last window before ``seq_len``, in
        position order; row 0 (the sink) for a position another rank owns."""
        positions = torch.arange(
            max(0, seq_len - self.window_left - 1), seq_len, dtype=torch.int64
        )
        rows = self.locs(torch.full_like(positions, req_pool_idx), positions)
        rows = rows // self.dcp_size
        rows[positions % self.dcp_size != self.dcp_rank] = 0
        return rows.to(torch.int32).numpy()

    def reset_floor(self, req_pool_idx: int, seq_len: int) -> None:
        """PD decode, only for a request no transfer sends a floor to: prefill's
        remote write must be the floor's only writer."""
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
