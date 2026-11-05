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
decode rows instead merge in one kernel (``DcpTopKMerge``).
"""

from functools import partial
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import _gather_candidate_prefix_kernel
from sglang.kernels.ops.attention.dsa.dcp_topk_merge import (
    dcp_topk_merge,
    dcp_topk_merge_buffer_numel,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    candidates_cluster_size,
    topk_transform_candidates_v2,
    topk_transform_packed_v2,
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

# Candidates per rank in the merge. Over 109824 decode rows of six real
# workloads (32K and 100K contexts, all 24 DSA layers), a rank held at most 339
# of a row's global top-2048 keys under DCP8, so 384 needs no fallback there.
_DCP_MERGE_CAND_K = 384


class DcpTopKMerge:
    """Exact global top-k merge of decode rows in one kernel; see
    jit/csrc/dsa/dcp_topk_merge.cuh. Needs 8 or 16 ranks (one warp each), top-k
    2048 and multicast; ``usable`` is False otherwise. ``split``: candidates come
    from the context-split verify indexer."""

    def __init__(self, group, max_rows: int, topk: int, split: bool = False):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = group.rank_in_group
        self.max_rows = max_rows
        self.split = split
        self.buffer: Optional[torch.Tensor] = None
        self.multicast_ptr = 0
        self.peer_ptrs: Optional[torch.Tensor] = None
        self.counters: Optional[torch.Tensor] = None
        # 1 where a row's candidates could not certify the result (for tests)
        self.fallback: Optional[torch.Tensor] = None
        self.union_scratch: Optional[torch.Tensor] = None

        world = group.world_size
        # 16 ranks only on megakernel paths; stock DCP16 keeps the all-gather.
        worlds = (8, 16) if split or megakernel_enabled() else (8,)
        if world not in worlds or topk != 2048:
            print_info_once(
                "DSA DCP top-k merge off: it needs 8 or 16 ranks and top-k 2048"
                if 16 in worlds
                else "DSA DCP top-k merge off: it needs 8 ranks and top-k 2048"
            )
            return
        if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
            group.cpu_group, group.device
        ):
            return
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
            return
        self.buffer = buffer
        self.multicast_ptr = handle.multicast_ptr
        self.peer_ptrs = torch.tensor(
            handle.buffer_ptrs, dtype=torch.int64, device=group.device
        )
        self.counters = torch.zeros(max_rows, dtype=torch.int64, device=group.device)
        self.fallback = torch.zeros(max_rows, dtype=torch.int32, device=group.device)
        self.union_scratch = torch.empty(
            (max_rows, world * topk), dtype=torch.int64, device=group.device
        )

    def usable(self, rows: int) -> bool:
        return self.buffer is not None and rows <= self.max_rows

    def __call__(
        self,
        candidates: torch.Tensor,
        global_lens: torch.Tensor,
        page_table: torch.Tensor,
        out: torch.Tensor,
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
            out_slots=out,
            out_fallback=self.fallback[: out.shape[0]],
            rank=self.rank,
            span=span,
            cand_k=_DCP_MERGE_CAND_K,
            split=self.split,
        )


_TOPK_MERGE: Optional[DcpTopKMerge] = None


def init_dcp_topk_merge(max_rows: int, topk: int) -> None:
    """Collective over the DCP group; call on every rank before CUDA graph
    capture."""
    global _TOPK_MERGE
    if _TOPK_MERGE is not None:  # one buffer per process
        return
    parallel = get_parallel()
    group = parallel.dcp_group
    # the kernel's rank must be the one the local candidates encode positions with
    assert (group.world_size, group.rank_in_group) == (
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
    )
    _TOPK_MERGE = DcpTopKMerge(group, max_rows, topk)

