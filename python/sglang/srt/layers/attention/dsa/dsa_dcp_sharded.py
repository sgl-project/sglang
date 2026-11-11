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

"""DSA indexer top-k under decode context parallelism (DCP).

The index-K cache is sharded by token: global position p of a request lives on
rank p % N at local position p // N. Each rank selects the top-k of its own
keys, the ranks all-gather (score, global position) candidates, and every rank
selects the global top-k from them. A key in the global top-k ranks within the
top-k of its own shard, so the merge is exact.

Both selections run the v2 top-k kernel, which orders by score descending,
then column ascending. Local columns are in global position order, and the
merge scatters candidates into a row indexed by global position, so ties
resolve by global position at both steps. With 8 or 16 ranks and multicast,
decode rows instead merge in one kernel (``DcpTopKMerge``); with 8 ranks,
they can also select and merge in one kernel (``DcpFusedTopK``).
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.attention.dcp_kernels import _gather_candidate_prefix_kernel
from sglang.kernels.ops.attention.dsa.dcp_fused_topk import (
    DCP_FUSED_TOPK_K,
    DCP_FUSED_TOPK_MAX_CONTEXT,
    DCP_FUSED_TOPK_PAGE_SIZE,
    DCP_FUSED_TOPK_RANKS,
    dcp_fused_topk,
    dcp_fused_topk_buffer_numel,
)
from sglang.kernels.ops.attention.dsa.dcp_topk_merge import (
    dcp_topk_merge,
    dcp_topk_merge_buffer_numel,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_candidates_v2,
    topk_transform_paged_v2,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_one_nvlink_clique,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled
from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_dcp
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.utils import print_info_once

_SIDE_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _SIDE_STREAMS.get(device)
    if stream is None:
        stream = _SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


class DcpCandidateAllGather:
    def __init__(self, group, max_rows: int, topk: int):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.max_rows = max_rows
        self.topk = topk
        self.merge_buffer: Optional[torch.Tensor] = None
        self.out: Optional[torch.Tensor] = None
        self.prefix_handle = None
        self.group_name = group.device_group.group_name
        if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
            group.cpu_group, group.device
        ):
            return
        prefix = group.world_size == 8 and topk == 2048 and get_platform().is_sm100
        if prefix:
            symm_mem.set_signal_pad_size(
                max(symm_mem.get_signal_pad_size(), 32 * group.world_size * 4)
            )
        per_rank = max_rows * topk * 2  # int32 (score bits, position) pairs
        out = symm_mem.empty(
            group.world_size * per_rank, dtype=torch.int32, device=group.device
        )
        handle = symm_mem.rendezvous(out, self.group_name)
        if handle.multicast_ptr != 0:
            self.out = out
            if prefix:
                self.prefix_handle = handle

    def __call__(
        self,
        t: torch.Tensor,
        *,
        local_lens: torch.Tensor,
        compact_prefix: bool = False,
    ) -> torch.Tensor:
        n = t.numel()
        if (
            self.out is None
            or t.shape[1] != self.topk
            or n * self.group.world_size > self.out.numel()
        ):
            return self.group.all_gather(t, dim=0)
        out = self.out[: self.group.world_size * n]
        if self.prefix_handle is None:
            torch.ops.symm_mem.multimem_all_gather_out(t.view(-1), self.group_name, out)
        else:
            handle = self.prefix_handle
            _gather_candidate_prefix_kernel[(8 if t.shape[0] <= 512 else 32,)](
                t,
                local_lens,
                handle.multicast_ptr,
                handle.signal_pad_ptrs_dev,
                t.shape[0],
                COMPACT_PREFIX=compact_prefix,
                TOPK=t.shape[1],
                RANK=handle.rank,
                WORLD=handle.world_size,
                BLOCK=512,
                num_warps=16,
            )
        return out.view(self.group.world_size * t.shape[0], *t.shape[1:])

    def merge_scores(self, rows: int, width: int) -> torch.Tensor:
        # Reserved outside capture and reused by every graph; never rebound once captured.
        # TODO(kpham-sgl): reserve at init inside the KV budget; it is max_rows x context fp32.
        if (
            self.merge_buffer is None
            and 0 < rows <= self.max_rows
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.merge_buffer = torch.empty(
                (self.max_rows, width), dtype=torch.float32, device=self.group.device
            )
        if (
            self.merge_buffer is not None
            and rows <= self.max_rows
            and width <= self.merge_buffer.shape[1]
        ):
            return self.merge_buffer[:rows, :width]
        return torch.empty((rows, width), dtype=torch.float32, device=self.group.device)


_CANDIDATE_ALL_GATHER: Optional[DcpCandidateAllGather] = None


def init_dcp_candidate_all_gather(max_rows: int, topk: int) -> None:
    """Collective over the DCP group; call on every rank before CUDA graph
    capture."""
    global _CANDIDATE_ALL_GATHER
    if _CANDIDATE_ALL_GATHER is not None:
        return
    _CANDIDATE_ALL_GATHER = DcpCandidateAllGather(
        get_parallel().dcp_group, max_rows, topk
    )


# Candidates per rank in the merge. Over 109824 decode rows of six real
# workloads (32K and 100K contexts, all 24 DSA layers), a rank held at most 339
# of a row's global top-2048 keys under DCP8, so 384 needs no fallback there.
_DCP_MERGE_CAND_K = 384


class DcpTopKMerge:
    """Exact global top-k merge of decode rows in one kernel; see
    jit/csrc/dsa/dcp_topk_merge.cuh. Built by ``create``. Calls go on one
    stream, never concurrently, with the same rows in the same order on every
    rank."""

    def __init__(self, group, max_rows: int, topk: int, buffer: torch.Tensor, handle):
        device = group.device
        self.dcp_rank = group.rank_in_group
        self.max_rows = max_rows
        self.buffer = buffer
        self.multicast_ptr = handle.multicast_ptr
        self.peer_ptrs = torch.tensor(
            handle.buffer_ptrs, dtype=torch.int64, device=device
        )
        self.counters = torch.zeros(max_rows, dtype=torch.int64, device=device)
        # 1 where a row's candidates could not certify the result (for tests)
        self.fallback = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.union_scratch = torch.empty(
            (max_rows, group.world_size * topk), dtype=torch.int64, device=device
        )

    @classmethod
    def create(
        cls, group, max_rows: int, topk: int, dcp_rank: int
    ) -> Optional[DcpTopKMerge]:
        """Collective over ``group``. None unless it has 8 ranks (16 with the
        megakernel), top-k is 2048, and its GPUs share NVLink multicast."""
        import torch.distributed._symmetric_memory as symm_mem

        # the certificate counts rank q's keys as positions p % world == q
        assert group.rank_in_group == dcp_rank, (group.rank_in_group, dcp_rank)

        world = group.world_size
        worlds = (8, 16) if megakernel_enabled() else (8,)
        if world not in worlds or topk != 2048:
            print_info_once(
                "DSA DCP top-k merge off: it needs 8 or 16 ranks and top-k 2048"
                if 16 in worlds
                else "DSA DCP top-k merge off: it needs 8 ranks and top-k 2048"
            )
            return None
        if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
            group.cpu_group, group.device
        ):
            print_info_once("DSA DCP top-k merge off: ranks span NVLink domains")
            return None
        buffer = symm_mem.empty(
            dcp_topk_merge_buffer_numel(world, max_rows, topk, _DCP_MERGE_CAND_K),
            dtype=torch.int64,
            device=group.device,
        )
        # Zero before the rendezvous: a peer pushes into this buffer as soon as
        # its own rendezvous returns.
        buffer.zero_()
        torch.cuda.synchronize(group.device)
        handle = symm_mem.rendezvous(buffer, group.device_group.group_name)
        if handle.multicast_ptr == 0:
            print_info_once("DSA DCP top-k merge off: no multicast")
            return None
        return cls(group, max_rows, topk, buffer, handle)

    def usable(self, rows: int) -> bool:
        return rows <= self.max_rows

    def __call__(
        self,
        candidates: torch.Tensor,
        global_lens: torch.Tensor,
        page_table: torch.Tensor,
        out_slots: torch.Tensor,
        span: int,
    ) -> None:
        dcp_topk_merge(
            candidates,
            global_lens,
            page_table,
            buffer=self.buffer,
            multicast_ptr=self.multicast_ptr,
            peer_ptrs=self.peer_ptrs,
            counters=self.counters,
            union_scratch=self.union_scratch,
            out_slots=out_slots,
            out_fallback=self.fallback[: out_slots.shape[0]],
            dcp_rank=self.dcp_rank,
            span=span,
            cand_k=_DCP_MERGE_CAND_K,
        )


_TOPK_MERGE: Optional[DcpTopKMerge] = None
_TOPK_MERGE_CREATED = False


def init_dcp_topk_merge(max_rows: int, topk: int) -> bool:
    """Collective over the DCP group; call on every rank before CUDA graph
    capture. Returns whether decode rows merge in one kernel."""
    global _TOPK_MERGE, _TOPK_MERGE_CREATED
    if not _TOPK_MERGE_CREATED:  # one buffer per process, also when create declined
        _TOPK_MERGE_CREATED = True
        parallel = get_parallel()
        _TOPK_MERGE = DcpTopKMerge.create(
            parallel.dcp_group, max_rows, topk, parallel.attn_dcp_rank
        )
    return _TOPK_MERGE is not None


@triton.jit
def _fill_visible_kernel(dense, dense_stride, lens, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.load(lens + row)
    base = dense + row.to(tl.int64) * dense_stride
    for start in range(tl.program_id(1) * BLOCK, n, tl.num_programs(1) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        tl.store(base + offs, float("-inf"), mask=offs < n)


@triton.jit
def _scatter_candidates_kernel(
    gathered,
    num_rows,
    global_lens,
    dense,
    dense_stride,
    WORLD: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPACT_PREFIX: tl.constexpr = False,
):
    src_row = tl.program_id(0)
    row = src_row % num_rows
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # Prefix-only gathers leave the padding undefined, including on graph
    # replays with shorter lengths. Never read it or scatter stale positions.
    n = (tl.load(global_lens + row) + WORLD - 1 - src_row // num_rows) // WORLD
    mask = offs < tl.minimum(n, TOPK)
    base = gathered + src_row.to(tl.int64) * TOPK * 2
    if COMPACT_PREFIX and n <= TOPK:
        score = tl.load(base + offs, mask=mask, other=0).to(tl.float32, bitcast=True)
        pos = offs * WORLD + src_row // num_rows
    else:
        score = tl.load(base + offs * 2, mask=mask, other=0).to(
            tl.float32, bitcast=True
        )
        pos = tl.load(base + offs * 2 + 1, mask=mask, other=-1)
    tl.store(
        dense + row.to(tl.int64) * dense_stride + pos,
        score,
        mask=mask & (pos >= 0),
    )


@triton.jit
def _localize_compact_kernel(
    slots,
    slots_stride,
    out,
    out_stride,
    lengths,
    dcp_size,
    dcp_rank,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, TOPK)
    x = tl.load(slots + row.to(tl.int64) * slots_stride + offs)
    own = (x >= 0) & (x % dcp_size == dcp_rank)
    own_i = own.to(tl.int32)
    n = tl.sum(own_i, axis=0)
    dst = tl.where(
        own, tl.cumsum(own_i, axis=0) - 1, n + tl.cumsum(1 - own_i, axis=0) - 1
    )
    tl.store(
        out + row.to(tl.int64) * out_stride + dst, tl.where(own, x // dcp_size, -1)
    )
    tl.store(lengths + row, n)


def dcp_localize_compact(
    slots: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global top-k slots [rows, topk] -> (this rank's rows compacted to the
    front, -1 after; [rows] counts, into ``lengths`` when given) for FlashMLA's
    ``topk_length``."""
    parallel = get_parallel()
    rows, topk = slots.shape
    out = torch.empty_like(slots)
    if lengths is None:
        lengths = torch.empty(rows, dtype=torch.int32, device=slots.device)
    if rows:
        _localize_compact_kernel[(rows,)](
            slots,
            slots.stride(0),
            out,
            out.stride(0),
            lengths,
            parallel.attn_dcp_size,
            parallel.attn_dcp_rank,
            TOPK=topk,
        )
    return out, lengths


def _local_candidates(
    logits: torch.Tensor, local_lens: torch.Tensor, topk: int
) -> torch.Tensor:
    # [rows, topk, 2] int32 (fp32 score bits, global position or -1); row i of
    # logits scores its request's local positions 0 .. local_lens[i]-1
    parallel = get_parallel()
    rows = local_lens.shape[0]
    out = torch.empty((rows, topk, 2), dtype=torch.int32, device=local_lens.device)
    if rows == 0:
        return out
    if logits.shape[1] == 0:
        out[..., 1] = -1
        return out
    # v2 ragged top-k with the candidate epilogue: local positions ascend
    # with global ones, so its tie order matches the merge's
    topk_transform_candidates_v2(
        logits,
        local_lens,
        out_candidates=out,
        dcp_size=parallel.attn_dcp_size,
        dcp_rank=parallel.attn_dcp_rank,
        deterministic=True,
    )
    return out


def dcp_global_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
    page_size: int,
    plan: torch.Tensor,
) -> torch.Tensor:
    """Decode: global top-k over the DCP group as widened KV slots, -1 padded.

    Row i is request i. ``local_lens`` / ``global_lens``: [rows] visible keys
    of this rank / of the whole sequence. ``page_table``: [rows, pages] this
    rank's physical page ids, which are the widened page ids of the request:
    global position p maps to slot ``page_table[p // span] * span + p % span``
    with span ``page_size * dcp_size``. ``plan``: the v2 top-k plan of
    ``global_lens``. Every rank must call this with the same number of rows,
    also when it has no local keys.

    With ``DcpTopKMerge`` the result holds only this rank's slots, which is all
    ``dcp_localize_compact`` keeps.
    """
    parallel = get_parallel()
    rows = local_lens.shape[0]
    span = page_size * parallel.attn_dcp_size
    global_lens = global_lens.to(torch.int32)
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return out
    if _TOPK_MERGE is not None and _TOPK_MERGE.usable(rows):
        candidates = _local_candidates(logits, local_lens, topk)
        _TOPK_MERGE(candidates, global_lens, page_table, out, span)
        return out
    workspace = _CANDIDATE_ALL_GATHER
    compact_prefix = (
        workspace is not None
        and workspace.prefix_handle is not None
        and rows <= workspace.max_rows
        and topk == workspace.topk
    )
    all_gather = (
        partial(workspace, local_lens=local_lens, compact_prefix=compact_prefix)
        if workspace is not None
        else partial(parallel.dcp_group.all_gather, dim=0)
    )

    # Row i of `dense` holds the candidate scores at their global positions.
    # Its -inf fill depends on the lengths only, so it overlaps the local top-k
    # and the all-gather, which run on a side stream. The fill is captured
    # first so that a CUDA graph continues on this stream after the join.
    width = triton.cdiv(page_table.shape[1] * span, 4) * 4
    dense = (
        workspace.merge_scores(rows, width)
        if workspace is not None
        else torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    )
    stream = torch.cuda.current_stream()
    side = _side_stream(logits.device)
    side.wait_stream(stream)
    _fill_visible_kernel[(rows, 16)](dense, dense.stride(0), global_lens, BLOCK=2048)
    with torch.cuda.stream(side):
        candidates = _local_candidates(logits, local_lens, topk)
        gathered = all_gather(candidates)
    stream.wait_stream(side)
    block = 1024
    _scatter_candidates_kernel[(gathered.shape[0], triton.cdiv(topk, block))](
        gathered,
        rows,
        global_lens,
        dense,
        dense.stride(0),
        WORLD=parallel.attn_dcp_size,
        COMPACT_PREFIX=compact_prefix,
        TOPK=topk,
        BLOCK=block,
    )
    topk_transform_paged_v2(
        dense, global_lens, page_table, out, span, plan, deterministic=True
    )
    return out


def dcp_local_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
    page_size: int,
    plan: torch.Tensor,
    out_len: torch.Tensor,
) -> torch.Tensor:
    """Decode: what DCP decode attention reads, this rank's keys of the global
    top-k as local KV slots compacted to the front (-1 after); their counts go
    to ``out_len`` [rows]. Arguments otherwise as for ``dcp_global_topk``."""
    rows = local_lens.shape[0]
    assert out_len.shape[0] == rows, (out_len.shape, rows)
    if _FUSED_TOPK is not None and 0 < rows and _FUSED_TOPK.usable(rows):
        slots = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
        return _FUSED_TOPK(logits, local_lens, global_lens, page_table, slots, out_len)
    slots = dcp_global_topk(
        logits, local_lens, global_lens, page_table, topk, page_size, plan
    )
    return dcp_localize_compact(slots, out_len)[0]


def dcp_gather_request_rows(
    local_rows: torch.Tensor, seq_lens_cpu: torch.Tensor
) -> torch.Tensor:
    """Prefill: every request's cache rows from all ranks, in position order.

    ``local_rows``: [sum of this rank's shares, W] uint8, request by request
    in local position order. Returns [sum(seq_lens), W]: request after
    request, the rows of positions 0..seq_len-1 (the layout of the ragged
    top-k indices).
    """
    return all_gather_kv_cache_for_dcp(local_rows, local_rows[:, :0], seq_lens_cpu)


class DcpFusedTopK:
    """Local select, exchange and merge of decode rows in one kernel; see
    jit/csrc/dsa/dcp_fused_topk.cuh. Built by ``create``. Rows per call are
    capped by the SM count (every CTA of every rank is resident at once).
    Contract: context length <= 1M (the kernel cuts longer rows). The selected
    values are exact; ties at the threshold are not ordered by position. Calls
    go on one stream, never concurrently, with the same rows in the same order
    on every rank."""

    def __init__(self, group, max_rows: int, buffer: torch.Tensor, handle):
        device = group.device
        self.dcp_rank = group.rank_in_group
        self.max_rows = max_rows
        self.buffer = buffer
        self.multicast_ptr = handle.multicast_ptr
        self.counters = torch.zeros(max_rows, dtype=torch.int32, device=device)
        # exact fallbacks per row slot (for tests)
        self.fallbacks = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self._warm(device)

    @classmethod
    def create(
        cls, group, max_rows: int, topk: int, page_size: int, dcp_rank: int
    ) -> Optional[DcpFusedTopK]:
        """Collective over ``group``. None unless it has 8 ranks, top-k is 2048,
        the page size is 64, the GPUs are SM100 and share NVLink multicast."""
        import torch.distributed._symmetric_memory as symm_mem

        # global position p of a row is local position p // 8 on rank p % 8
        assert group.rank_in_group == dcp_rank, (group.rank_in_group, dcp_rank)

        if (
            group.world_size != DCP_FUSED_TOPK_RANKS
            or topk != DCP_FUSED_TOPK_K
            or page_size != DCP_FUSED_TOPK_PAGE_SIZE
            or not current_arch().is_sm(10)
        ):
            print_info_once(
                "DSA DCP fused top-k off: it needs 8 ranks, top-k 2048, page size 64 and SM100"
            )
            return None
        if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
            group.cpu_group, group.device
        ):
            print_info_once("DSA DCP fused top-k off: ranks span NVLink domains")
            return None
        max_rows = min(
            max_rows,
            torch.cuda.get_device_properties(group.device).multi_processor_count,
        )
        buffer = symm_mem.empty(
            dcp_fused_topk_buffer_numel(max_rows),
            dtype=torch.int32,
            device=group.device,
        )
        # Zero before the rendezvous: a peer pushes into this buffer as soon as
        # its own rendezvous returns.
        buffer.zero_()
        torch.cuda.synchronize(group.device)
        handle = symm_mem.rendezvous(buffer, group.device_group.group_name)
        if handle.multicast_ptr == 0:
            print_info_once("DSA DCP fused top-k off: no multicast")
            return None
        return cls(group, max_rows, buffer, handle)

    def _warm(self, device: torch.device) -> None:
        # The first touch of the peer and multicast pages costs 65-160 us; take
        # it here for both parities of every row slot, not in the first forward.
        rows, n = self.max_rows, 2 * DCP_FUSED_TOPK_K // DCP_FUSED_TOPK_RANKS
        g = torch.Generator(device=device).manual_seed(0)
        logits = torch.randn((rows, n), generator=g, device=device)
        local_lens = torch.full((rows,), n, dtype=torch.int32, device=device)
        page_table = torch.zeros(
            (rows, n // DCP_FUSED_TOPK_PAGE_SIZE), dtype=torch.int32, device=device
        )
        out = torch.empty((rows, DCP_FUSED_TOPK_K), dtype=torch.int32, device=device)
        global_lens = local_lens * DCP_FUSED_TOPK_RANKS
        for _ in range(2):
            self(
                logits,
                local_lens,
                global_lens,
                page_table,
                out,
                torch.empty_like(local_lens),
            )
        torch.cuda.synchronize(device)

    def usable(self, rows: int) -> bool:
        return rows <= self.max_rows

    def __call__(
        self,
        logits: torch.Tensor,
        local_lens: torch.Tensor,
        global_lens: torch.Tensor,
        page_table: torch.Tensor,
        out_slots: torch.Tensor,
        out_len: torch.Tensor,
    ) -> torch.Tensor:
        """Writes this rank's keys of the global top-k into ``out_slots`` as
        widened slots and their counts per row into ``out_len``; returns them
        as local KV slots compacted to the front, -1 after."""
        local = torch.empty_like(out_slots)
        dcp_fused_topk(
            logits,
            local_lens,
            global_lens,
            page_table,
            out_local=local,
            out_slots=out_slots,
            out_len=out_len,
            buffer=self.buffer,
            multicast_ptr=self.multicast_ptr,
            counters=self.counters,
            fallbacks=self.fallbacks,
            dcp_rank=self.dcp_rank,
        )
        return local


_FUSED_TOPK: Optional[DcpFusedTopK] = None
_FUSED_TOPK_CREATED = False


def init_dcp_fused_topk(
    max_rows: int, topk: int, page_size: int, max_context_len: int
) -> bool:
    """Collective over the DCP group; call on every rank before CUDA graph
    capture. Returns whether decode rows select and merge in one kernel."""
    global _FUSED_TOPK, _FUSED_TOPK_CREATED
    if max_context_len > DCP_FUSED_TOPK_MAX_CONTEXT:
        print_info_once("DSA DCP fused top-k off: it needs context length <= 1M")
        return False
    if not _FUSED_TOPK_CREATED:  # one buffer per process, also when create declined
        _FUSED_TOPK_CREATED = True
        parallel = get_parallel()
        _FUSED_TOPK = DcpFusedTopK.create(
            parallel.dcp_group, max_rows, topk, page_size, parallel.attn_dcp_rank
        )
    return _FUSED_TOPK is not None
