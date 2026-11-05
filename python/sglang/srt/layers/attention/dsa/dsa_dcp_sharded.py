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
keys, and the ranks merge them into the global top-k. A key in the global
top-k ranks within the top-k of its own shard, so the merge is exact. With 8
or 16 ranks and multicast, decode rows merge in one kernel (``DcpTopKMerge``).
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.dsa.dcp_topk_merge import (
    dcp_topk_merge,
    dcp_topk_merge_buffer_numel,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_one_nvlink_clique,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled
from sglang.srt.utils import print_info_once

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
