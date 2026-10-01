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
from functools import lru_cache
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils import print_info_once

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _dcp_page_interleave_flag() -> bool:
    # Cached, not read at import: a module-level read happens before any test
    # can set the variable. Same reason as _dsa_token_shard_flag.
    return envs.SGLANG_NPU_DCP_PAGE_INTERLEAVE.get()


def dcp_interleave_size() -> int:
    """Positions a rank holds in a row before the next rank's.

    NPU only. CUDA's Triton store (``kernels/ops/kvcache/mla_buffer.py:42``)
    and the MHA read paths hardcode 1, so shared callers pass 1 explicitly.
    """
    return get_schedule().page_size if _dcp_page_interleave_flag() else 1


def dcp_owner_count(length: int, dcp_size: int, dcp_rank: int, interleave: int) -> int:
    """Scalar ``get_dcp_lens``: how many of ``[0, length)`` this rank owns."""
    cycle = dcp_size * interleave
    full, rem = divmod(length, cycle)
    return full * interleave + min(max(rem - dcp_rank * interleave, 0), interleave)


def dcp_local_row(loc: torch.Tensor, dcp_size: int, interleave: int) -> torch.Tensor:
    """The row ``loc`` occupies in its owner's pool."""
    if interleave == 1:
        return loc // dcp_size
    return (loc // (interleave * dcp_size)) * interleave + loc % interleave


def dcp_owner_and_row(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, interleave: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(owned mask, destination row)``, for write paths.

    Not ``localize_dcp_indices``: that returns a ``-1`` sentinel the write
    paths would only have to mask away again, and at ``interleave == 1`` its
    divide-by-1 and mod-1 still launch kernels on a per-layer path.
    """
    block = loc if interleave == 1 else loc // interleave
    return (block % dcp_size) == dcp_rank, dcp_local_row(loc, dcp_size, interleave)


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
    # path.  Keep the math equivalent to localize_dcp_indices above. Exact only
    # below 2**24 positions, 16.4x the 1M target, and it degrades to a wrong KV
    # row rather than failing -- test_the_float32_bound_on_the_position_itself.
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
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, interleave_size: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``loc`` this rank owns, and the physical rows they land on.

    Across the group the ``owned_idx`` partition every row of ``loc`` exactly
    once, so a filtered write covers the same rows as an unfiltered one.

    ``torch.nonzero`` has a data-dependent output shape, so it syncs the stream
    and cannot run under capture -- extend only, where one write location is
    shared by every layer. Decode aims non-owned rows at a padding row instead.
    """
    owned, dest = dcp_owner_and_row(loc, dcp_size, dcp_rank, interleave_size)
    owned_idx = torch.nonzero(owned).squeeze(1)
    return owned_idx, dest[owned_idx]


# Reusable device buffers for the extend gather, keyed by purpose, dtype, device
# and row shape. See dcp_extend_gather_buffer.
_dcp_extend_gather_buffers: Dict[Tuple, torch.Tensor] = {}


def dcp_extend_gather_buffer(name: str, ref: torch.Tensor, rows: int) -> torch.Tensor:
    """A reusable ``rows``-row buffer shaped and typed like ``ref``. Grow-only.

    Allocated per layer, the peak moved with whatever the allocator held when a
    layer asked for another ~1.1 GiB. Two callers sharing ``name`` share the
    buffer and must not hold slices across each other's calls.
    """
    key = (name, ref.dtype, ref.device, tuple(ref.shape[1:]))
    buf = _dcp_extend_gather_buffers.get(key)
    if buf is None or buf.shape[0] < rows:
        # Drop every reference first, or the peak is briefly both buffers.
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


class DcpSharedPrefix(NamedTuple):
    """What a batch's prefixes share, by radix node.

    ``walked`` is each request's tree-owned prefix rebuilt from its match and
    ``protected`` is the scheduler's own count of it, so the two disagreeing
    means the walk is wrong. ``union_rows`` counts a shared node once: the rows
    a deduplicated gather would send for the tree-owned part.
    """

    walked: List[int]
    protected: List[int]
    union_rows: int


def _dcp_node_rows(node: object) -> Optional[int]:
    """Tokens a radix node holds on device, or None when it holds none.

    Two node shapes reach here. ``UnifiedTreeNode`` -- what the default
    ``UnifiedRadixCache`` builds, so what a served run uses -- keeps the Full KV
    under ``component_data``; ``RadixCache``'s ``TreeNode`` has a plain
    ``value``. Reading only the latter counts every node as empty.
    """
    data = getattr(node, "component_data", None)
    value = (
        data[BASE_COMPONENT_TYPE].value
        if data is not None
        else getattr(node, "value", None)
    )
    return None if value is None else len(value)


def dcp_shared_prefix(
    last_nodes: Sequence[object], protected_lens: Sequence[int]
) -> DcpSharedPrefix:
    """Measure prefix sharing across a batch from the radix match alone.

    A request's tree-owned prefix is its root-to-``last_node`` path, so sharing
    is the paths' common ancestors -- no device read and no index comparison.
    Duck-typed on ``.parent``/``.value`` to stay import-free and CPU-testable.

    Only the tree-owned part is shareable. Under ``page_size > 1`` a chunked
    request also carries a partial page that ``cache_unfinished_req`` keeps in
    ``prefix_indices`` but not in the tree; it belongs to that request alone, so
    it is outside both ``walked`` and ``union_rows`` and a dedup must still send
    it per request.
    """
    seen: Dict[int, int] = {}
    walked: List[int] = []
    for last_node in last_nodes:
        node, rows = last_node, 0
        while node is not None:
            n = _dcp_node_rows(node)
            if n:
                rows += n
                seen.setdefault(id(node), n)
            node = getattr(node, "parent", None)
        walked.append(rows)
    return DcpSharedPrefix(
        walked=walked, protected=list(protected_lens), union_rows=sum(seen.values())
    )


class DcpExtendGatherPiece(NamedTuple):
    """One collective of the extend gather. The all-gather lays the sends out
    rank-major, this chunk's own KV follows, and ``index`` maps output rows
    ``[out_start, out_end)`` onto that scratch."""

    send_start: int
    send_end: int
    extend_start: int
    extend_end: int
    out_start: int
    out_end: int
    index: torch.Tensor
    scratch_rows: int


class DcpExtendGatherPlan(NamedTuple):
    """How one rank gathers a batch's prefix KV at extend. Sends are padded
    to ``padded_lens`` so every rank's is ``send_rows`` long; the output is each
    request's prefix then its extend, in order, tiled by ``pieces``."""

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
    interleave_size: int = 1,
) -> DcpExtendGatherPlan:
    """Plan the extend-time gather of the prefix KV into position order.

    Position p lives on rank ``(p // B) % dcp_size`` at local row
    ``(p // (B * dcp_size)) * B + p % B``, ``B = interleave_size``. Local shards
    are concatenated per request, as ``dcp_local_prefix_kv_indices`` lists them.

    Pieces gather at most ``max_piece_gather_rows`` rows each, and cut on whole
    ``B``-blocks: that keeps local rows ``[row, row + take)`` covering global
    positions ``[row * dcp_size, (row + take) * dcp_size)`` on every rank. The
    pieces are identical on every rank; only ``local_lens`` is not.
    """
    prefix_lens = [int(p) for p in prefix_lens]
    extend_lens = [int(e) for e in extend_lens]
    b = interleave_size
    local_lens = [dcp_owner_count(p, dcp_size, dcp_rank, b) for p in prefix_lens]
    # Rank 0 holds the most; every rank pads to it.
    padded_lens = [dcp_owner_count(p, dcp_size, 0, b) for p in prefix_lens]
    # Rounded down to a whole block, so a cut never splits one rank's run.
    piece_rows = max(b, (max_piece_gather_rows // dcp_size) // b * b)

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
                        b,
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
                b,
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
    interleave_size: int = 1,
) -> DcpExtendGatherPiece:
    """Close one piece: index its output rows into its rank-major scratch."""
    send_len = send_end - send_start
    gathered_rows = send_len * dcp_size
    b = interleave_size
    index = []
    for kind, first, second in parts:
        if kind == "prefix":
            positions, request_send_offset = first, second
            owner = (positions // b) % dcp_size
            local_row = (positions // (b * dcp_size)) * b + positions % b
            index.append(
                owner * send_len + (request_send_offset - send_start) + local_row
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

    Only if every prefix reaches ``index_topk``: below it the top-k takes every
    key, so the crop is what keeps the result causal. Cached per forward, and
    the FIRST caller's ``index_topk`` wins -- the cache does not key on it.
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
