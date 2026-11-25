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

"""Decode context parallel (DCP) layout: per-rank lengths, the owner-rule
local-index filter, and the LSE base of each MLA backend's partials."""

from typing import NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import print_info_once

# NOTE(yhyang201): under DCP, Kimi-K3.1 shards the DSA index-K cache by token like
# the MLA KV instead of replicating it; the indexer merges per-rank top-k exactly.
_dsa_shards_index_k = False


def set_dsa_shards_index_k() -> None:
    """Mark index-K as token-sharded; must run before pools and backends are built."""
    global _dsa_shards_index_k
    _dsa_shards_index_k = True


def dsa_shards_index_k() -> bool:
    return _dsa_shards_index_k and get_parallel().attn_dcp_size > 1


def is_mla_dcp_lse_base_on_e(attention_backend: Optional[str]) -> bool:
    if attention_backend == "dsa":
        # trtllm-gen partials carry a base-2 LSE.
        return dsa_shards_index_k() and get_exec().kernel.dsa_decode_backend != "trtllm"
    return attention_backend in {"flashmla", "cutedsl_mla", "aiter"}


def dsa_index_k_page_tokens(page_size: int) -> int:
    if dsa_shards_index_k():
        return page_size * get_parallel().attn_dcp_size
    return page_size


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
    local_indices = dcp_local_row(indices, dcp_size, interleave_size)
    return torch.where(is_local, local_indices, torch.full_like(indices, -1))


def dcp_local_row(
    indices: torch.Tensor, dcp_size: int, interleave_size: int = 1
) -> torch.Tensor:
    """The row each index occupies in its owner's pool, whichever rank that is."""
    interleave_block = torch.div(indices, interleave_size, rounding_mode="floor")
    local_block = torch.div(interleave_block, dcp_size, rounding_mode="floor")
    return local_block * interleave_size + indices % interleave_size


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
    interleave_size: int = 1,
) -> torch.Tensor:
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return kv_indices

    dcp_size = parallel.dcp_size
    if interleave_size != 1:
        offsets = _dcp_local_chunk_offsets(
            chunk_starts_cpu.tolist(),
            chunk_seq_lens_cpu.tolist(),
            dcp_size,
            parallel.dcp_rank,
            interleave_size,
        )
        return kv_indices[offsets.to(kv_indices.device)]
    parts = []
    offset = 0
    for start, length in zip(chunk_starts_cpu.tolist(), chunk_seq_lens_cpu.tolist()):
        first = (parallel.dcp_rank - start) % dcp_size
        parts.append(kv_indices[offset + first : offset + length : dcp_size])
        offset += length
    return torch.cat(parts)


def _dcp_local_chunk_offsets(
    chunk_starts, chunk_seq_lens, dcp_size: int, dcp_rank: int, interleave_size: int
) -> torch.Tensor:
    """Host offsets into the concatenated chunks of the positions this rank owns.

    Per chunk, in position order. Walks owned runs, not positions, so a 1M-token
    prefix costs ~1M / dcp_size host ints and no device sync.
    """
    cycle = dcp_size * interleave_size
    run = torch.arange(interleave_size, dtype=torch.int64)
    parts = [torch.empty(0, dtype=torch.int64)]
    offset = 0
    for start, length in zip(chunk_starts, chunk_seq_lens):
        end = start + length
        # Start of this rank's run at or before `start` (may be negative).
        first = start - (start - dcp_rank * interleave_size) % cycle
        run_starts = torch.arange(first, end, cycle, dtype=torch.int64)
        positions = (run_starts[:, None] + run).reshape(-1)
        positions = positions[(positions >= start) & (positions < end)]
        parts.append(positions - start + offset)
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
    # D2v2: the sub-all-gather count the buffer was (or will be) filled with.
    # 1 is the monolithic rank-major layout above; above 1 the prefix part is
    # chunk-major (chunk, then rank, then row) and the remap gains a chunk-major
    # branch. See dcp_gather_chunks_for.
    chunks: int = 1


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
        chunks=dcp_gather_chunks_for(send_rows),
    )


def dcp_gather_chunks_for(send_rows: int) -> int:
    """The effective D2v2 sub-all-gather count for a send this long.

    The env value clamped to what the send can actually be cut into: a chunk of
    zero rows would be a collective with nothing in it, and one row per chunk
    (send shorter than the count) keeps the arithmetic honest rather than being
    a special case. 1 is "off": the monolithic collective and the rank-major
    layout, bit for bit.
    """
    chunks = _dcp_gather_chunks_env
    if chunks <= 1:
        return 1
    return min(chunks, max(1, int(send_rows)))


def _chunk_bounds(plan: DcpPackedReadPlan):
    """(q, rem, n): the D2v2 chunk sizes.

    The first ``n - 1`` chunks hold ``q = send_rows // n`` local rows each; the
    last holds ``q + send_rows % n``. ``n`` is 1 only for the monolithic plan,
    where this is never called.
    """
    n = plan.chunks
    q, rem = divmod(plan.send_rows, n)
    return q, rem, n


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
        c = plan.dcp_size
        if plan.chunks == 1:
            return (pos % c) * plan.send_rows + pos // c
        # Chunk-major: local row i of rank r = chunk k, in-chunk row j, and
        # chunk k's block starts at c * (k * q) rows in (chunks before it each
        # held q local rows x c ranks).
        q, rem, n = _chunk_bounds(plan)
        r = pos % c
        i = pos // c
        if i < (n - 1) * q:
            k = i // q
            j = i - k * q
            sk = q
        else:
            k = n - 1
            j = i - (n - 1) * q
            sk = q + rem
        return c * (k * q) + r * sk + j
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
    if plan.chunks == 1:
        prefix_row = (topk_indices % c) * plan.send_rows + local_row
    else:
        # D2v2 chunk-major: local row i of rank r sits in chunk k = i // q
        # (the last chunk absorbs the remainder) at in-chunk row j, and chunk
        # k's block starts c * (k * q) rows in. Same arithmetic as
        # packed_row_of, fused into where/clamp because this runs on a
        # [tokens, k] tensor 21 times a forward.
        q, rem, n = _chunk_bounds(plan)
        k = torch.div(local_row, q, rounding_mode="floor").clamp(max=n - 1)
        j = torch.where(k == n - 1, local_row - (n - 1) * q, local_row - k * q)
        sk = q + (k == n - 1).to(local_row.dtype) * rem
        prefix_row = c * (k * q) + (topk_indices % c) * sk + j
    extend_row = plan.gathered_rows + (topk_indices - plan.prefix_len)
    packed = torch.where(topk_indices < plan.prefix_len, prefix_row, extend_row)
    return torch.where(topk_indices < 0, topk_indices, packed)


_enable_dcp_packed_read = envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get()
_dcp_gather_chunks_env = envs.SGLANG_NPU_DCP_GATHER_CHUNKS.get()


class _PackedMissing:
    """Distinguishes "not resolved yet" from "resolved, and it is None"."""


_PACKED_MISSING = _PackedMissing()


def _dsa_kv_store_is_fp8() -> bool:
    """Whether the active DSA KV pool stores FP8 (``dsa_kv_cache_store_fp8`` /
    MXFP8). The packed single-record read does not support it: under the FP8
    store ``get_mla_kv_buffer`` returns the packed uint8 view with ``rope=None``,
    which the packed path's ``_pad_dcp_packed_send`` cannot consume. Defensive:
    the forward context is not always up (a plan probe can run before a forward),
    so a missing context means "not FP8" and the other refusals still apply.
    """
    try:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        return bool(getattr(get_token_to_kv_pool(), "dsa_kv_cache_store_fp8", False))
    except Exception:
        return False


def dcp_packed_read_plan(
    forward_batch, index_topk: Optional[int]
) -> Optional[DcpPackedReadPlan]:
    """This forward's packed-buffer plan, or None to keep the permuting path.

    Resolved once per forward and cached on the NPU DCP metadata (the subclass
    built for the DSA-DCP extend path, whose fields these are). It lives here,
    beside the arithmetic, rather than in the NPU model module, because BOTH the
    model (to fill the buffer and remap the top-k) and the attention backend (to
    set the operator's KV length and drop its causal crop) have to agree on it,
    and a backend importing a model module inverts the layering -- the same
    reason ``dsa_cp.py`` sits where it does.

    Every refusal is logged once. A silent refusal here would look exactly like
    a feature that is on and doing nothing.
    """
    if not _enable_dcp_packed_read:
        return None
    md = forward_batch.attn_dcp_metadata
    if md.packed_read_plan_done:
        return md.packed_read_plan

    plan = None
    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if _dsa_kv_store_is_fp8():
        # FP8 DSA KV store (dsa_kv_cache_store_fp8 / MXFP8): the packed
        # single-record read is not supported for it (get_mla_kv_buffer returns
        # rope=None). The permuting path handles FP8 (it packs uint8 and the
        # backend's quant branch dequants), so decline to that fallback -- the
        # v1 packed-read refusal, re-expressed as a plan admission.
        print_info_once(
            "DCP packed read is off: the DSA KV is stored in FP8 "
            "(dsa_kv_cache_store_fp8); the packed single-record read is not "
            "supported for it, so the permuting path is used"
        )
    elif index_topk is None:
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
        # path handles those.
        print_info_once(
            f"DCP packed read is off for multi-request extends "
            f"({len(extend_lens)} requests here); the all-gather is rank-major "
            "over the whole send, so no request is contiguous in it"
        )
    elif not md.extend_crop_free:
        # Computed once at prepare (dcp_extend_crop_free); it has already
        # logged which request fell short.
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
    md.packed_read_plan = plan
    md.packed_read_plan_done = True
    return plan


def dcp_packed_kv_lens(forward_batch, plan: DcpPackedReadPlan, device) -> torch.Tensor:
    """The operator's ``actual_seq_lengths_kv`` for a packed read, built once.

    One request, so one entry, and it is the whole buffer: with the causal crop
    off there is nothing to shorten, and the padding rows the ranks sent to keep
    their sends equal are never named by any remapped index.

    Cached on the DCP metadata rather than built where it is used, because that
    use is once per layer and ``torch.tensor(list, device=npu)`` is a blocking
    host-to-device copy -- 78 of them a forward, each draining the queue.
    """
    md = forward_batch.attn_dcp_metadata
    cached = md.packed_kv_lens
    if cached is None:
        cached = torch.tensor([plan.rows], dtype=torch.int32, device=device)
        md.packed_kv_lens = cached
    return cached
