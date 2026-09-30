# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP): per-rank lengths,
the owner-rule local-index filter, and the NPU extend-gather plan."""

import logging
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once

logger = logging.getLogger(__name__)


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return the KV length owned by one DCP rank.

    Consecutive runs of ``interleave_size`` positions rotate across ranks.  The
    default value preserves the token-interleaved layout used by the generic
    CUDA/ROCm DCP paths; NPU DSA passes its physical KV page size.
    """
    if dcp_size == 1:
        return lens
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    cycle_size = dcp_size * interleave_size

    def _count_before(end: torch.Tensor) -> torch.Tensor:
        full_cycles = end // cycle_size
        remainder = end % cycle_size
        rank_remainder = torch.clamp(
            remainder - dcp_rank * interleave_size,
            min=0,
            max=interleave_size,
        )
        return full_cycles * interleave_size + rank_remainder

    if start is None:
        return _count_before(lens)
    return _count_before(start + lens) - _count_before(start)


def localize_dcp_indices(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Map global DCP indices to one rank, using ``-1`` for non-local rows."""
    if dcp_size == 1:
        return indices
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    interleave_block = torch.div(indices, interleave_size, rounding_mode="floor")
    is_local = (indices >= 0) & (interleave_block % dcp_size == dcp_rank)
    local_block = torch.div(interleave_block, dcp_size, rounding_mode="floor")
    local_indices = local_block * interleave_size + indices % interleave_size
    return torch.where(is_local, local_indices, torch.full_like(indices, -1))


def maybe_dcp_kernel_indices(
    indices: torch.Tensor, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """Widened logical slots -> this rank's physical rows.

    Owner rule: slot % dcp_size == dcp_rank, row = slot // dcp_size. The run
    starts page-aligned, so a strided view selects the owned slots without a mask.
    """
    if dcp_size == 1:
        return indices
    return indices[dcp_rank::dcp_size] // dcp_size


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    """Keep this rank's share of a read-index tensor, still WIDENED.

    Selection only; the caller collapses via translate_dcp_read_ids.
    """
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
    return kv_indices


def filter_dcp_local_chunk_kv_indices(
    kv_indices: torch.Tensor,
    chunk_starts_cpu: torch.Tensor,
    chunk_seq_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return kv_indices

    dcp_size = parallel.dcp_size
    parts = []
    offset = 0
    for start, length in zip(chunk_starts_cpu.tolist(), chunk_seq_lens_cpu.tolist()):
        first = (parallel.dcp_rank - start) % dcp_size
        parts.append(kv_indices[offset + first : offset + length : dcp_size])
        offset += length
    return torch.cat(parts)


def remap_dcp_sparse_indices(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Map global sparse token indices to one rank's compact DCP KV layout.

    Keep indices owned by this rank, convert them to local KV positions, and
    stably move them before ``-1`` padding while preserving their score order.
    """
    if dcp_size == 1:
        return topk_indices
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    # Float32 is faster than integer division/remainder on Ascend for this hot
    # path.  Keep the math equivalent to localize_dcp_indices above.
    topk_indices_fp32 = topk_indices.to(torch.float32)
    interleave_blocks = torch.floor(topk_indices_fp32 / interleave_size)
    local_owner_mask = (topk_indices_fp32 >= 0) & (
        torch.remainder(interleave_blocks, dcp_size) == dcp_rank
    )
    local_offsets = torch.remainder(topk_indices_fp32, interleave_size)
    local_indices = (
        torch.floor(interleave_blocks / dcp_size) * interleave_size + local_offsets
    )
    remapped_indices = torch.where(
        local_owner_mask,
        local_indices,
        torch.full_like(topk_indices_fp32, -1.0),
    ).to(topk_indices.dtype)

    # Move valid entries before padding without changing their top-k order.
    topk_count = topk_indices.shape[-1]
    original_order = torch.arange(
        topk_count, dtype=torch.float32, device=topk_indices.device
    ).expand_as(topk_indices_fp32)
    pack_keys = original_order + (~local_owner_mask).to(torch.float32) * topk_count
    pack_order = torch.argsort(pack_keys, dim=-1).to(torch.int64)
    return torch.gather(remapped_indices, dim=-1, index=pack_order)


def get_dcp_chain_spec_lens(
    total_kv_lens: torch.Tensor,
    tokens_per_req: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return request-major local KV frontiers for a speculative chain."""
    if tokens_per_req < 1:
        raise ValueError(f"tokens_per_req must be >= 1, got {tokens_per_req}")
    total_kv_lens = total_kv_lens.int()
    steps = torch.arange(
        1, tokens_per_req + 1, dtype=total_kv_lens.dtype, device=total_kv_lens.device
    )
    global_query_lens = total_kv_lens.unsqueeze(1) - tokens_per_req + steps
    global_query_lens = torch.where(
        total_kv_lens.unsqueeze(1) >= tokens_per_req,
        global_query_lens,
        torch.zeros_like(global_query_lens),
    )
    return get_dcp_lens(
        global_query_lens.reshape(-1),
        dcp_size,
        dcp_rank,
        interleave_size=interleave_size,
    ).int()


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))


def plan_dcp_owner_write(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``loc`` this rank owns, and the physical rows they land on.

    The owner rule is CUDA's, from the Triton kernel at
    ``kernels/ops/kvcache/mla_buffer.py:42``::

        is_valid = loc % DCP_WORLD_SIZE == DCP_RANK
        loc      = loc // DCP_WORLD_SIZE

    Returns ``(owned_idx, dest)``: positions into ``loc`` that this rank owns,
    and their destinations in its own pool. Together across the group the
    ``owned_idx`` partition every row of ``loc`` exactly once, so a write
    filtered through this covers the same rows as an unfiltered one.

    ``torch.nonzero`` makes the output shape data-dependent, which costs a
    stream synchronisation and cannot run inside a captured stream -- so the
    NPU decode path aims non-owned rows at a padding row instead. Extend is not
    captured and shares one write location across layers, so there the filter
    is computed once per forward.
    """
    owned_idx = torch.nonzero((loc % dcp_size) == dcp_rank).squeeze(1)
    return owned_idx, loc[owned_idx] // dcp_size


# Reusable device buffers for the extend gather, keyed by purpose, dtype, device
# and row shape. See dcp_extend_gather_buffer.
_dcp_extend_gather_buffers: Dict[Tuple, torch.Tensor] = {}


def dcp_extend_gather_buffer(name: str, ref: torch.Tensor, rows: int) -> torch.Tensor:
    """Return a reusable ``rows``-row buffer shaped and typed like ``ref``.

    The extend gather's context-sized tensors were allocated fresh on each of
    the model's layers, so the peak depended on what the allocator happened to
    hold when a layer asked for another ~1.1 GiB. Reserving them once turns
    that into a budget that no longer moves between layers.

    Grow-only, so a longer context raises the reservation and keeps it; warm-up
    prefills the longest context a deployment serves, so the growth lands
    there. ``name`` is part of the key with dtype, device and row shape, so
    buffers that must not alias do not -- two callers sharing a key share one
    buffer and must not hold their slices across each other's calls.
    """
    key = (name, ref.dtype, ref.device, tuple(ref.shape[1:]))
    buf = _dcp_extend_gather_buffers.get(key)
    if buf is None or buf.shape[0] < rows:
        # Release the old buffer before asking for the new one, or the peak is
        # briefly both of them.
        _dcp_extend_gather_buffers.pop(key, None)
        buf = None
        buf = torch.empty((rows, *ref.shape[1:]), dtype=ref.dtype, device=ref.device)
        _dcp_extend_gather_buffers[key] = buf
        logger.info(
            "DCP extend gather buffer %r reserved: %d rows, %.3f GiB",
            name,
            rows,
            buf.numel() * buf.element_size() / (1 << 30),
        )
    return buf[:rows]


class DcpExtendGatherPiece(NamedTuple):
    """One collective of the extend-time prefix gather.

    Every rank sends rows ``[send_start, send_end)`` of its padded send, and the
    all-gather lays them out rank-major; rows ``[extend_start, extend_end)`` of
    this chunk's own KV are appended after them. ``index`` maps the output rows
    ``[out_start, out_end)`` -- a contiguous run of the position-ordered output
    -- to rows of that scratch.
    """

    send_start: int
    send_end: int
    extend_start: int
    extend_end: int
    out_start: int
    out_end: int
    index: torch.Tensor
    scratch_rows: int


class DcpExtendGatherPlan(NamedTuple):
    """How one rank gathers a batch's prefix KV at extend.

    ``local_lens[i]`` rows of request i live on this rank; each rank sends them
    padded to ``padded_lens[i]`` so every rank's send is ``send_rows`` long. The
    output is position-ordered -- request 0's prefix then extend, request 1's
    prefix then extend, ... -- and ``pieces`` tile it in order.
    """

    local_lens: List[int]
    padded_lens: List[int]
    send_rows: int
    scratch_rows: int
    pieces: List[DcpExtendGatherPiece]


def plan_dcp_extend_gather(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    dcp_size: int,
    dcp_rank: int,
    max_piece_gather_rows: int,
) -> DcpExtendGatherPlan:
    """Plan the extend-time gather of the prefix KV into position order.

    Under the owner rule ``pos % dcp_size == rank`` with each request's prefix
    starting at position 0, rank r holds positions ``r, r + dcp_size, ...`` of a
    request in order, so position p sits in rank ``p % dcp_size``'s send at
    local row ``p // dcp_size``. Local shards are concatenated per request, as
    the planner's ``dcp_local_prefix_kv_indices`` lists them.

    The sends are cut into pieces of ``max_piece_gather_rows // dcp_size`` rows
    (at least one), so no collective gathers more than ``max_piece_gather_rows``
    rows (at least ``dcp_size``) and the caller can write each piece into place
    before gathering the next. Cuts fall on whole local rows, which keeps every
    piece's output a contiguous run; small requests share a piece, and a
    request's own KV rides in the piece its prefix ends in. The pieces and their
    indices are the same on every rank; only ``local_lens`` depends on
    ``dcp_rank``.
    """
    prefix_lens = [int(p) for p in prefix_lens]
    extend_lens = [int(e) for e in extend_lens]
    local_lens = [p // dcp_size + int(dcp_rank < p % dcp_size) for p in prefix_lens]
    padded_lens = [-(-p // dcp_size) for p in prefix_lens]
    piece_rows = max(1, max_piece_gather_rows // dcp_size)

    pieces = []
    parts = []
    send_start = extend_start = out_start = 0
    send_end = extend_end = out_end = 0
    send_offset = 0
    for prefix_len, extend_len, padded_len in zip(
        prefix_lens, extend_lens, padded_lens
    ):
        row = 0
        while row < padded_len:
            if send_end - send_start == piece_rows:
                pieces.append(
                    _dcp_extend_gather_piece(
                        parts,
                        dcp_size,
                        send_start,
                        send_end,
                        extend_start,
                        extend_end,
                        out_start,
                        out_end,
                    )
                )
                parts = []
                send_start, extend_start, out_start = send_end, extend_end, out_end
            take = min(padded_len - row, piece_rows - (send_end - send_start))
            positions = torch.arange(
                row * dcp_size,
                min((row + take) * dcp_size, prefix_len),
                dtype=torch.int64,
            )
            parts.append(("prefix", positions, send_offset))
            row += take
            send_end += take
            out_end += positions.numel()
        if extend_len:
            parts.append(("extend", extend_end, extend_len))
            extend_end += extend_len
            out_end += extend_len
        send_offset += padded_len
    if parts:
        pieces.append(
            _dcp_extend_gather_piece(
                parts,
                dcp_size,
                send_start,
                send_end,
                extend_start,
                extend_end,
                out_start,
                out_end,
            )
        )
    return DcpExtendGatherPlan(
        local_lens=local_lens,
        padded_lens=padded_lens,
        send_rows=sum(padded_lens),
        scratch_rows=max((p.scratch_rows for p in pieces), default=0),
        pieces=pieces,
    )


def _dcp_extend_gather_piece(
    parts,
    dcp_size: int,
    send_start: int,
    send_end: int,
    extend_start: int,
    extend_end: int,
    out_start: int,
    out_end: int,
) -> DcpExtendGatherPiece:
    """Close one piece: index its output rows into its rank-major scratch."""
    send_len = send_end - send_start
    gathered_rows = send_len * dcp_size
    index = []
    for kind, first, second in parts:
        if kind == "prefix":
            # Position p of a request whose send starts at `second` is rank
            # p % dcp_size's local row second + p // dcp_size.
            positions, request_send_offset = first, second
            index.append(
                (positions % dcp_size) * send_len
                + (request_send_offset - send_start)
                + positions // dcp_size
            )
        else:
            start = gathered_rows + first - extend_start
            index.append(torch.arange(start, start + second, dtype=torch.int64))
    return DcpExtendGatherPiece(
        send_start=send_start,
        send_end=send_end,
        extend_start=extend_start,
        extend_end=extend_end,
        out_start=out_start,
        out_end=out_end,
        index=torch.cat(index),
        scratch_rows=gathered_rows + extend_end - extend_start,
    )


def dcp_crop_free_extend(forward_batch, index_topk: Optional[int]) -> bool:
    """May this extend forward drop the operator's causal crop (sparse_mode 0)?

    Only if every request's prefix reaches ``index_topk``: below that the top-k
    selects every key it is offered, so the crop is what makes the result
    causal. The crop is set once per call, so the answer is decided once per
    forward and cached. ``index_topk`` of None answers False -- keeping the
    crop is always correct.
    """
    cached = getattr(forward_batch, "npu_dcp_crop_free", None)
    if cached is not None:
        return cached
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    ok = bool(
        index_topk is not None
        and prefix_lens
        and all(p + 1 >= index_topk for p in prefix_lens)
    )
    if not ok and index_topk is not None and prefix_lens:
        print_info_once(
            f"DCP extend keeps the operator's causal crop: a request here has a "
            f"prefix under index_topk={index_topk} (shortest {min(prefix_lens)})"
        )
    forward_batch.npu_dcp_crop_free = ok
    return ok
