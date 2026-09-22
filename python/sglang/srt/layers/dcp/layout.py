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

"""Pure index math for decode context parallel (DCP): lengths, indices, layouts."""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


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


def dcp_empty_lse_rows(
    local_lens_cpu: torch.Tensor,
    seq_lens_q_cpu: Optional[torch.Tensor],
) -> torch.Tensor:
    """Which query rows must take an -inf LSE into the DCP merge, one flag per row."""
    # CPU tensors only: repeat_interleave sizes its output from the repeat values,
    # so device tensors here would sync once per layer.
    empty = local_lens_cpu == 0
    if seq_lens_q_cpu is None:
        return empty
    return empty.repeat_interleave(seq_lens_q_cpu)


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


def dcp_gather_q_heads(
    q: torch.Tensor,
    group: "GroupCoordinator",
    num_kv_head: int,
    q_per_kv_head: int,
) -> torch.Tensor:
    """All-gather query heads across the DCP group in KV-head-major order."""
    q = group.all_gather(q, dim=1)
    if num_kv_head <= 1 or q_per_kv_head <= 0:
        return q.contiguous()
    tokens, _, head_dim = q.shape
    return (
        q.view(tokens, group.world_size, num_kv_head, q_per_kv_head, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(tokens, -1, head_dim)
        .contiguous()
    )


def dcp_ungather_heads(
    x: torch.Tensor,
    dcp_size: int,
    num_kv_head: int,
    q_per_kv_head: int,
) -> torch.Tensor:
    """Invert ``dcp_gather_q_heads``'s head permutation (KV-major -> rank-major)."""
    if num_kv_head <= 1 or q_per_kv_head <= 0:
        return x
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(-1)
    tokens, _, trailing = x.shape
    x = (
        x.view(tokens, num_kv_head, dcp_size, q_per_kv_head, trailing)
        .permute(0, 2, 1, 3, 4)
        .reshape(tokens, -1, trailing)
    )
    return x.squeeze(-1) if squeeze else x


def dcp_shard_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    max_local_len: int,
) -> Optional[torch.Tensor]:
    """Paged page table over one DCP rank's KV shard, in the kernel's block ids."""
    if max_local_len == 0:
        return None
    num_pages = (max_local_len + page_size - 1) // page_size
    cols = (
        torch.arange(num_pages, device=req_to_token.device, dtype=torch.int64)
        * (page_size * dcp_size)
        + dcp_rank
    )
    slots = req_to_token[req_pool_indices.unsqueeze(1), cols.unsqueeze(0)]
    return (slots // dcp_size // page_size).to(torch.int32)


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
