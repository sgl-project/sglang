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

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

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

    owned = (topk_indices >= 0) & (topk_indices % dcp_size == parallel.attn_dcp_rank)
    local = torch.where(
        owned,
        topk_indices // dcp_size,
        torch.full_like(topk_indices, invalid),
    )

    # Stable compaction without relying on a stable sort: offsetting a
    # non-owned entry's position by K keeps every key distinct, so the
    # permutation is unique and the ordering is exact rather than tie-broken.
    k = topk_indices.shape[-1]
    order = torch.arange(k, device=topk_indices.device, dtype=topk_indices.dtype)
    keys = order + (~owned).to(topk_indices.dtype) * k
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
