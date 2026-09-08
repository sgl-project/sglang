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


def remap_dcp_write_locations_fixed_shape(
    virtual_locs: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    *,
    dummy_loc: int = 0,
) -> torch.Tensor:
    """Map virtual DCP slots to a rank-local pool without compacting tensors.

    Each rank owns ``virtual_loc % dcp_size == dcp_rank`` and stores the slot at
    ``virtual_loc // dcp_size``.  Locations owned by another rank are redirected
    to the allocator-reserved dummy slot.  Keeping the original tensor shape is
    intentional: Ascend boolean indexing lowers to ``aclnnNonzeroV2`` and can
    fail in the 64-rank target-verify path, while the following scatter already
    accepts repeated writes to dummy slot 0.
    """
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(
            f"invalid DCP topology: dcp_size={dcp_size}, dcp_rank={dcp_rank}"
        )
    owner_mask = torch.remainder(virtual_locs, dcp_size) == dcp_rank
    local_locs = torch.div(virtual_locs, dcp_size, rounding_mode="floor")
    return torch.where(
        owner_mask,
        local_locs,
        torch.full_like(local_locs, dummy_loc),
    )


def build_mla_dcp_local_block_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    physical_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    num_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the rank-local MLA page table from widened virtual token ids.

    The allocator exposes virtual pages of ``physical_page_size * dcp_size``
    tokens. Rank ``r`` owns virtual positions ``r, r + dcp_size, ...`` and
    stores them densely at ``virtual_loc // dcp_size``.  Ascend FIA consumes
    physical page ids, so its block table must be derived from those owner
    positions rather than slicing the global virtual table by page size.

    Returns ``(block_tables, local_seq_lens)``.  A one-column dummy table is
    kept when this rank owns no token in the batch because paged-attention
    kernels generally reject a zero-width page table.
    """
    if physical_page_size <= 0:
        raise ValueError(
            f"physical_page_size must be positive, got {physical_page_size}"
        )
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(
            f"invalid DCP topology: dcp_size={dcp_size}, dcp_rank={dcp_rank}"
        )

    local_seq_lens = get_dcp_lens(seq_lens, dcp_size, dcp_rank).to(torch.int32)
    if num_pages is None:
        max_local_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() > 0 else 0
        )
        num_pages = max(
            1, (max_local_len + physical_page_size - 1) // physical_page_size
        )
    elif num_pages <= 0:
        raise ValueError(f"num_pages must be positive, got {num_pages}")

    local_page_offsets = torch.arange(
        num_pages, dtype=torch.long, device=req_to_token.device
    )
    global_positions = dcp_rank + local_page_offsets * physical_page_size * dcp_size
    req_rows = req_pool_indices.to(device=req_to_token.device, dtype=torch.long)
    # A graph table has a fixed maximum width and can include one rounded-up
    # page beyond the request-table width.  Clamp the read itself and mask that
    # page below; this keeps replay shape-static without an out-of-bounds load.
    positions_in_range = global_positions < req_to_token.shape[1]
    safe_global_positions = global_positions.clamp(max=req_to_token.shape[1] - 1)
    virtual_locs = req_to_token[req_rows[:, None], safe_global_positions[None, :]]
    block_tables = (virtual_locs // dcp_size // physical_page_size).to(torch.int32)
    valid_pages = (
        local_page_offsets[None, :] * physical_page_size
        < local_seq_lens.to(req_to_token.device)[:, None]
    ) & positions_in_range[None, :]
    block_tables.masked_fill_(~valid_pages, 0)
    return block_tables.contiguous(), local_seq_lens


def build_mla_dcp_mtp_mask(
    prefix_lens: torch.Tensor,
    query_lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    *,
    max_query_len: int | None = None,
    max_local_kv_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the rank-local causal mask used by speculative MLA decode.

    Queries stay replicated while KV token ``p`` is stored by rank
    ``p % dcp_size``.  Therefore query ``j`` of a request with prefix length
    ``P`` may consume local KV index ``k`` iff ``dcp_rank + k * dcp_size <=
    P + j``.  The returned boolean mask follows FIA semantics (``True`` means
    masked) and is shaped ``[batch, max_q_len, max_local_kv_len]``.

    ``local_seq_lens`` includes the speculative query window because those KV
    rows are written before attention executes.
    """
    if prefix_lens.ndim != 1 or query_lens.ndim != 1:
        raise ValueError("prefix_lens and query_lens must both be rank-1 tensors")
    if prefix_lens.numel() != query_lens.numel():
        raise ValueError(
            "prefix_lens and query_lens must have the same batch size, got "
            f"{prefix_lens.numel()} and {query_lens.numel()}"
        )
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(
            f"invalid DCP topology: dcp_size={dcp_size}, dcp_rank={dcp_rank}"
        )

    prefix_lens = prefix_lens.to(torch.int64)
    query_lens = query_lens.to(device=prefix_lens.device, dtype=torch.int64)
    total_lens = prefix_lens + query_lens
    local_seq_lens = get_dcp_lens(total_lens, dcp_size, dcp_rank).to(torch.int32)

    if max_query_len is None:
        max_q_len = int(query_lens.max().item()) if query_lens.numel() else 0
    else:
        if max_query_len <= 0:
            raise ValueError(f"max_query_len must be positive, got {max_query_len}")
        max_q_len = max_query_len
    if max_local_kv_len is None:
        max_local_kv_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        )
    elif max_local_kv_len <= 0:
        raise ValueError(f"max_local_kv_len must be positive, got {max_local_kv_len}")
    # FIA rejects a zero-width attention mask even for an empty local shard.
    mask = torch.ones(
        (prefix_lens.numel(), max(1, max_q_len), max(1, max_local_kv_len)),
        dtype=torch.bool,
        device=prefix_lens.device,
    )
    if max_q_len == 0 or max_local_kv_len == 0:
        return mask.contiguous(), local_seq_lens

    q_idx = torch.arange(max_q_len, device=prefix_lens.device, dtype=torch.int64)
    k_idx = torch.arange(max_local_kv_len, device=prefix_lens.device, dtype=torch.int64)
    # Local index of the last visible KV token for every global query position.
    last_visible = torch.div(
        prefix_lens[:, None] + q_idx[None, :] - dcp_rank,
        dcp_size,
        rounding_mode="floor",
    )
    mask = k_idx[None, None, :] > last_visible[:, :, None]
    mask |= q_idx[None, :, None] >= query_lens[:, None, None]
    mask |= k_idx[None, None, :] >= local_seq_lens.to(torch.int64)[:, None, None]
    return mask.contiguous(), local_seq_lens


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
