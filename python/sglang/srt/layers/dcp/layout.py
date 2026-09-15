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

from typing import List, NamedTuple, Sequence

import torch

from sglang.srt.runtime_context import get_parallel


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
