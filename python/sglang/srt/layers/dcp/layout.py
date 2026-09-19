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

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once

logger = logging.getLogger(__name__)


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


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


def dcp_local_kv_block_table(loc_rows: torch.Tensor, page_size: int) -> torch.Tensor:
    """The rank-local KV page table, from the same ``req_to_token`` slice.

    Under DCP the attention backend needs *two* page tables over one allocation,
    because the indexer and sparse attention read different pools:

        indexer   replicated, pages of ``page_size`` over the full virtual span
        latent KV sharded,     pages of ``page_size`` over this rank's rows

    The existing expression in the backend --
    ``req_to_token[reqs, :n][:, ::page_size] // page_size`` -- already produces
    the **indexer's** table under DCP and needs no change. This produces the
    other one.

    The derivation, writing P for page_size and c for dcp_size. The allocator
    pages in *virtual* space at ``P * c``, so allocator page number k of a
    request has some physical id q_k and covers virtual locations
    ``q_k * P * c + i`` for ``i`` in ``[0, P*c)``. Rank r owns the ones with
    ``v % c == r``; since ``q_k * P * c`` is divisible by c that is ``i % c == r``,
    and those map under ``// c`` to ``q_k * P + i // c`` -- physical page q_k,
    offset ``i // c``. So a rank-local page is exactly an allocator page, and
    reading the location at every ``P * c``-th position and dividing by ``P * c``
    recovers its physical id.

    At ``c == 1`` this is character-for-character the existing expression, which
    is what makes it safe to route both through here.
    """
    stride = page_size * get_parallel().attn_dcp_size
    return loc_rows[:, ::stride] // stride


def remap_dcp_local_topk_indices(
    topk_indices: torch.Tensor, invalid: int = -1
) -> torch.Tensor:
    """Global sparse top-k positions -> rank-local KV coordinates, shape preserved.

    The sparse indexer selects top-k over the *replicated* index-K buffer, so it
    returns positions in the full sequence's coordinate system. Sparse attention
    reads the *sharded* latent KV and needs rank-local ones. This is the
    translation between the two.

    ``filter_dcp_local_kv_indices`` cannot be reused here even though the owner
    rule is identical, because it *selects* and so returns a shorter tensor. A
    top-k tensor is ``[..., K]`` with K fixed by ``index_topk``, and each row
    keeps a different number of entries -- selecting would make it ragged, which
    no kernel can take. So instead of dropping non-owned entries this marks them
    ``invalid`` and stably compacts the survivors to the front, leaving the
    shape untouched and the top-k order intact. Same approach as vLLM-Ascend
    (``attention/context_parallel/sfa_cp.py:1023-1045``), which is the only
    working reference for DCP composed with a sparse indexer.

    The ``>= 0`` guard is load-bearing, not defensive: the indexer pads short
    rows with -1, and ``-1 % dcp_size`` is ``dcp_size - 1`` under torch's
    Python-style modulo, so on the highest rank every padding entry would
    otherwise look owned and translate to a real row.
    """
    parallel = get_parallel()
    dcp_size = parallel.attn_dcp_size
    if dcp_size == 1:
        return topk_indices

    # Checked before any work, because it is a precondition on the compaction
    # below rather than a property of the result -- see the sort-key note.
    k = topk_indices.shape[-1]
    assert 2 * k <= 1 << 24, (
        f"top-k width {k} exceeds the exactly-representable float32 sort-key "
        "range; keys would collide and the compaction would stop being a "
        "permutation. Sort integer keys here instead, and pay AiCpu on Ascend."
    )

    owned = (topk_indices >= 0) & (topk_indices % dcp_size == parallel.attn_dcp_rank)
    local = torch.where(
        owned,
        topk_indices // dcp_size,
        torch.full_like(topk_indices, invalid),
    )

    # Stable compaction without relying on a stable sort: offsetting a
    # non-owned entry's position by K keeps every key distinct, so the
    # permutation is unique and the ordering is exact rather than tie-broken.
    #
    # The keys are float32 rather than the index dtype because Ascend has no
    # AiCore ArgSort for int32/int64 -- it silently falls back to AiCpu and says
    # so at warning level ("please cast dtype to float32",
    # ArgSortKernelNpuOpApi.cpp:26). This runs once per layer per decode step,
    # 78 times, and measured on A3 that fallback was the dominant term in the
    # DCP decode penalty: ~9.6 ms per request per step against dcp1's own
    # 2.5 ms, flat in dcp_size and flat in context length, which is the
    # signature of [batch, K] work rather than anything touching the sequence.
    # vLLM-Ascend sorts float32 keys (sfa_cp.py:1023-1045) for the same reason.
    #
    # Exact, not approximate. The keys are the distinct integers [0, 2K) and
    # float32 represents every integer below 2**24 exactly, so the resulting
    # permutation is identical to the integer sort's -- the distinctness the
    # offset was introduced for is what makes the dtype change free. The assert
    # states the bound rather than leaving it implicit; index_topk is three
    # orders of magnitude below it, and the bound is asserted above.
    order = torch.arange(k, device=topk_indices.device, dtype=torch.float32)
    keys = order + (~owned).to(torch.float32) * k
    return torch.gather(local, -1, torch.argsort(keys, dim=-1))


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
    stream synchronisation and **cannot run inside a captured stream**. That is
    why the NPU decode path does not use this and aims its non-owned rows at a
    padding row instead (``_resolve_dcp_write``). Extend is not captured, and
    one forward's write location is shared by every layer, so there the filter
    is computed once and reused 78 times.
    """
    owned_idx = torch.nonzero((loc % dcp_size) == dcp_rank).squeeze(1)
    return owned_idx, loc[owned_idx] // dcp_size


# Reusable device buffers for the extend gather, keyed by purpose, dtype, device
# and row shape. See dcp_extend_gather_buffer.
_dcp_extend_gather_buffers: Dict[Tuple, torch.Tensor] = {}


def dcp_extend_gather_buffer(name: str, ref: torch.Tensor, rows: int) -> torch.Tensor:
    """Return a reusable ``rows``-row buffer shaped and typed like ``ref``.

    The NPU extend gather's context-sized tensors -- the position-ordered latent
    output, the rope-key output, and the per-piece all-gather scratch -- were
    allocated fresh on each of the model's 78 layers. The bytes moved are the
    same either way, but the *peak* was then whatever the allocator happened to
    hold when a layer asked for another ~1.1 GiB, which is data-dependent and is
    what OOM'd at ``--mem-fraction-static 0.76``. Reserving them once turns that
    peak into a budget: the reservation is logged, it is identical on every
    rank, and everything allocated after it -- the MoE above all -- comes out of
    a pool whose size no longer moves between layers.

    Grow-only and never shrunk, so a longer context raises the reservation and
    keeps it. No env var pins it up front because none is needed in practice:
    warm-up prefills the longest context the deployment serves, so the growth
    lands there rather than mid-serving.

    ``name`` separates buffers that must not alias. It is part of the key
    together with dtype, device and row shape, so the latent and the rope key
    get different buffers even where their row counts agree, and so does a
    second device or a dtype change. Two callers that pass the same key share
    one buffer and must not hold their slices across each other's calls.

    This is also what makes overlapping the gather with compute possible at all.
    Overlap means two layers' outputs are alive at once, and allocating those
    per layer is exactly the pattern that stalled before -- see
    ``_dcp_gather_extend_kv_npu`` in the NPU MLA module, which has the history.
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
    )


class DcpPackedReadPlan(NamedTuple):
    """Where each KV position of a ONE-REQUEST extend lives in the packed buffer.

    ``plan_dcp_extend_gather`` permutes the all-gather's rank-major output back
    into position order so the sparse operator can read it. That permutation is
    a context-sized ``index_select`` -- 784 launches x 0.275 ms = 216 ms a
    forward -- plus the position-ordered output buffer it writes into, ~1.04 GiB
    a rank at a 972k context. Neither is needed: the operator reads the KV
    through ``sparse_indices``, so it does not care what order the rows are in,
    only that the indices point at the right ones. Remapping a ``[tokens, k]``
    index tensor is arithmetic on a few million integers; permuting the KV is a
    gigabyte of traffic. Upstream reached the same conclusion for CUDA in
    ``#39215``'s ``create_packed_dsa_dcp_kv_indices``.

    The packed buffer is exactly what the collective produces, in one piece::

        [ rank 0's send rows | rank 1's | ... | rank C-1's ][ this chunk's KV ]
          <--- send_rows ---> each                            <-- extend_len -->

    Under the owner rule ``pos % C == rank`` a request's position ``p`` sits in
    rank ``p % C``'s send at local row ``p // C``, so::

        p <  prefix_len   ->  (p % C) * send_rows + p // C
        p >= prefix_len   ->  gathered_rows + (p - prefix_len)

    ``send_rows`` is the padded local length, so ranks below ``prefix_len % C``
    hold one row more than the rest and the spare rows are never addressed.

    **ONE REQUEST ONLY, and the reason is the same one DSA-CP has.** With two
    requests the all-gather still lays out rank-major over the whole send, so a
    request's rows land in C separate blocks and the operator's cumulative
    ``actual_seq_lengths_kv`` -- which needs each request contiguous -- cannot
    describe them. Multi-request extends keep the permuting path. That covers
    87% of the tokens in the AISBench run as measured on 2026-09-18.
    """

    prefix_len: int
    extend_len: int
    dcp_size: int
    send_rows: int
    gathered_rows: int
    rows: int


def plan_dcp_packed_read(
    prefix_len: int, extend_len: int, dcp_size: int
) -> DcpPackedReadPlan:
    """Sizes of the packed rank-major buffer for a one-request extend."""
    send_rows = -(-int(prefix_len) // dcp_size)
    gathered_rows = send_rows * dcp_size
    return DcpPackedReadPlan(
        prefix_len=int(prefix_len),
        extend_len=int(extend_len),
        dcp_size=dcp_size,
        send_rows=send_rows,
        gathered_rows=gathered_rows,
        rows=gathered_rows + int(extend_len),
    )


def packed_row_of(pos: int, plan: DcpPackedReadPlan) -> int:
    """Reference implementation of the remap, one position at a time.

    Exists to be the thing ``remap_topk_to_packed`` is checked against on CPU:
    the tensor version has to fuse the branch into ``where`` and do floor
    division on a signed tensor, and those are the two places this kind of
    arithmetic goes wrong.
    """
    if pos < 0:
        return pos
    if pos < plan.prefix_len:
        return (pos % plan.dcp_size) * plan.send_rows + pos // plan.dcp_size
    return plan.gathered_rows + (pos - plan.prefix_len)


def remap_topk_to_packed(
    topk_indices: torch.Tensor, plan: DcpPackedReadPlan
) -> torch.Tensor:
    """Top-k positions within a request -> rows of the packed buffer.

    Shape and dtype are preserved, and so are the negatives: ``_pad_topk_indices``
    fills unused slots with -1 and the operator reads that tail as "no more
    entries", so mapping them into a real row would silently add keys. Every
    branch here is elementwise, which is what keeps this ~ a millisecond against
    the 216 ms it replaces.
    """
    c = plan.dcp_size
    # floor division, not trunc: torch's // on a signed tensor rounds toward
    # zero for negatives, and the -1 sentinels must come through untouched.
    local_row = torch.div(topk_indices, c, rounding_mode="floor")
    prefix_row = (topk_indices % c) * plan.send_rows + local_row
    extend_row = plan.gathered_rows + (topk_indices - plan.prefix_len)
    packed = torch.where(topk_indices < plan.prefix_len, prefix_row, extend_row)
    return torch.where(topk_indices < 0, topk_indices, packed)


_enable_dcp_packed_read = envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get()


class _PackedMissing:
    """Distinguishes "not resolved yet" from "resolved, and it is None"."""


_PACKED_MISSING = _PackedMissing()


def dcp_packed_read_enabled() -> bool:
    """Whether the packed rank-major read is on for this process."""
    return _enable_dcp_packed_read


def dcp_packed_causal_crop_is_dead(prefix_len: int, index_topk: int) -> bool:
    """Whether the operator's causal crop can be dropped for this chunk.

    THE PACKED READ NEEDS ``sparse_mode = 0``: the crop aligns query i to key
    ``K - Q + i``, which means nothing once the keys are in rank-major order.
    The first version of this claimed the top-k is causal by construction, so
    the crop was redundant. That is false, and a stage-A run measured it --
    scattered logprob differences up to 2.09 against a 0.354 noise floor,
    starting at position 2 and confined to a region about ``index_topk`` wide.

    The top-k is causal only where there is something to select FROM. Upstream
    says it plainly at dsa_indexer.py:402 -- "topk_transform selects every valid
    page slot when kv_len <= index_topk". At or below that length the top-k is
    not a selection, it is everything, and the crop is the only thing making the
    result causal. The NPU indexer itself leans on this: it calls
    npu_lightning_indexer with sparse_mode=3 (dsa_npu_indexer.py:527), so the
    indices it hands back are not pre-masked.

    Above it, every query in the chunk has at least ``index_topk`` keys strictly
    before it, the scores of later keys are -inf, and a top-k of that can only
    name causal keys. Then, and only then, the crop is dead weight and dropping
    it changes nothing.

    The bound is on the FIRST query in the chunk, which is the one with the
    least context: it sees ``prefix_len`` keys before it plus itself. Every
    later query in the chunk sees more.

    This is also exactly where the packed read pays: it exists for a 16-32k tail
    on a ~958k cached prefix, where ``prefix_len`` clears 2048 by five hundred
    times. At ``prefix_len = 0`` it saves nothing anyway -- there is no gathered
    prefix to permute, and the remap is the identity.
    """
    return prefix_len + 1 >= index_topk


def dcp_crop_free_extend(forward_batch, index_topk: Optional[int]) -> bool:
    """May this extend forward drop the operator's causal crop?

    Only if EVERY request in the batch clears the bound, because the crop is set
    once for the whole call. Decided once per forward and cached, so the model
    and the backend cannot disagree halfway down a forward.

    Three features want this answer and all three must get the same one:
      * the packed read, which cannot use the crop at all once rows are
        rank-major;
      * the DSA-CP multi-request lift, which drops the crop so it can stop
        shortening per-request KV lengths;
      * the DSA-CP single-request path under that lift, which the lift also
        moves onto sparse_mode 0 on purpose.

    ``index_topk`` of None means the caller had no top-k to measure, which is
    answered False -- the crop stays, which is always correct and sometimes
    slower.
    """
    cached = getattr(forward_batch, "npu_dcp_crop_free", None)
    if cached is not None:
        return cached
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    ok = bool(
        index_topk is not None
        and prefix_lens
        and all(dcp_packed_causal_crop_is_dead(p, index_topk) for p in prefix_lens)
    )
    if not ok and index_topk is not None and prefix_lens:
        print_info_once(
            f"DCP extend keeps the operator's causal crop: a request here has a "
            f"prefix under index_topk={index_topk} (shortest {min(prefix_lens)}), "
            "and below that the top-k selects every key it is offered, so the "
            "crop is what makes the result causal"
        )
    forward_batch.npu_dcp_crop_free = ok
    return ok


def dcp_packed_read_plan(
    forward_batch, index_topk: Optional[int]
) -> Optional[DcpPackedReadPlan]:
    """This forward's packed-buffer plan, or None to keep the permuting path.

    Resolved once per forward and cached on the batch. It lives here, beside the
    arithmetic, rather than in the NPU model module, because BOTH the model (to
    fill the buffer and remap the top-k) and the attention backend (to set the
    operator's KV length and drop its causal crop) have to agree on it, and a
    backend importing a model module inverts the layering -- the same reason
    ``dsa_cp.py`` sits where it does.

    Every refusal is logged once. A silent refusal here would look exactly like
    a feature that is on and doing nothing, which is a mistake this port has
    already paid four weeks for.
    """
    if not _enable_dcp_packed_read:
        return None
    cached = getattr(forward_batch, "npu_dcp_packed_plan", _PACKED_MISSING)
    if cached is not _PACKED_MISSING:
        return cached

    plan = None
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if index_topk is None:
        # No top-k to measure, so the causal bound below cannot be checked.
        # Refuse, and CACHE the refusal: a forward has to take one path the
        # whole way down. Resolving later would leave earlier layers' top-k
        # unremapped while later layers read a packed buffer.
        print_info_once(
            "DCP packed read is off: this forward reached the extend gather "
            "with no top-k, so the causal bound cannot be checked"
        )
    elif not extend_lens or prefix_lens is None:
        print_info_once("DCP packed read is off: no CPU length metadata on this extend")
    elif len(extend_lens) != 1:
        # ONE REQUEST, for the layout reason in DcpPackedReadPlan: the
        # all-gather is rank-major over the WHOLE send, so with two requests a
        # request's rows land in dcp_size separate blocks and the operator's
        # cumulative actual_seq_lengths_kv cannot describe them. The permuting
        # path handles those; it is 13% of the tokens in the served benchmark.
        print_info_once(
            f"DCP packed read is off for multi-request extends "
            f"({len(extend_lens)} requests here); the all-gather is rank-major "
            "over the whole send, so no request is contiguous in it"
        )
    elif not dcp_crop_free_extend(forward_batch, index_topk):
        # Measured, stage A, 2026-09-19: dropping the crop below this bound
        # moved prefill logprobs by up to 2.09 against a 0.354 noise floor.
        # dcp_crop_free_extend has already logged which request fell short.
        print_info_once(
            "DCP packed read is off: this chunk needs the operator's causal "
            "crop, and the packed layout cannot use one"
        )
    else:
        plan = plan_dcp_packed_read(
            prefix_lens[0], extend_lens[0], get_parallel().dcp_size
        )
        print_info_once(
            "DCP packed read is ON: the sparse operator reads the all-gather's "
            "own rank-major output and the top-k is remapped instead"
        )
    forward_batch.npu_dcp_packed_plan = plan
    return plan


def dcp_packed_kv_lens(forward_batch, plan: DcpPackedReadPlan, device) -> torch.Tensor:
    """The operator's ``actual_seq_lengths_kv`` for a packed read, built once.

    One request, so one entry, and it is the whole buffer: with the causal crop
    off there is nothing to shorten, and the padding rows the ranks sent to keep
    their sends equal are never named by any remapped index.

    Cached on the batch rather than built where it is used, because that use is
    once per layer and ``torch.tensor(list, device=npu)`` is a blocking
    host-to-device copy -- 78 of them a forward, each draining the queue. The
    same mistake was already made and fixed once on the DSA-CP path.
    """
    cached = getattr(forward_batch, "npu_dcp_packed_kv_lens", None)
    if cached is None:
        cached = torch.tensor([plan.rows], dtype=torch.int32, device=device)
        forward_batch.npu_dcp_packed_kv_lens = cached
    return cached
