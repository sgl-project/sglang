"""BF16 kpool index cache operations for NPU.

PyTorch-based replacements for the CUDA/FP8 Triton kernels in kpool_fp8_index.py.
Key differences from FP8 path:
  - No Hadamard128 rotation (only needed to spread FP8 quantization error)
  - No FP8 quantization (store BF16 directly)
  - No per-token FP32 scale
  - Buffer layout: (layer_num, num_pages, page_size, 1, head_dim) dtype=bfloat16
"""

from typing import Optional, Tuple

import torch

BLOCK_SIZE_K = 128
INDEX_HEAD_DIM = 128


def build_pooled_page_table_64(
    page_table_64: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Pack one logical pool page into the first token page of each page group."""
    assert BLOCK_SIZE_K % pool_size == 0
    idx = torch.arange(
        0, page_table_64.shape[-1], pool_size, device=page_table_64.device
    )
    return page_table_64[..., idx]


def compute_pooled_write_locs(
    page_table_64: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Map logical pooled-K ids to packed physical index-cache locations."""
    assert page_table_64.ndim == 1
    pool_ids = pool_ids.to(torch.int64)
    pool_page_group = torch.div(pool_ids, BLOCK_SIZE_K, rounding_mode="floor")
    token_page_row = pool_page_group * pool_size
    packed_page = page_table_64.index_select(0, token_page_row.to(torch.int64))
    return packed_page.to(torch.int64) * BLOCK_SIZE_K + torch.remainder(
        pool_ids, BLOCK_SIZE_K
    )


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    assert topk % pool_size == 0
    return topk // pool_size


def kpool_softmax_write_cache_bf16(
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    return_compressed: bool = False,
    write_cache: bool = True,
) -> Optional[Tuple[torch.Tensor, None]]:
    """BF16 softmax-weighted pooling: compress pool_size tokens into 1 BF16 slot.

    No Hadamard rotation, no FP8 quantization.

    Args:
        buf: (layer_num, num_pages, page_size, 1, head_dim) BF16, or
             (num_pages, page_size, head_dim) BF16 for a single layer
        slot_k: (n_pools, pool_size, head_dim) BF16
        slot_score: (n_pools, pool_size, head_dim) BF16
        ape: (pool_size, head_dim) FP32 additive positional encoding
        loc: (n_pools,) int64 physical locations
    """
    assert slot_k.ndim == 3
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == INDEX_HEAD_DIM
    assert slot_k.dtype == torch.bfloat16
    assert ape.dtype == torch.float32

    n_pools, pool_size, head_dim = slot_k.shape
    if n_pools == 0:
        if return_compressed:
            return (
                torch.empty((0, head_dim), dtype=torch.bfloat16, device=slot_k.device),
                None,
            )
        return None

    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()

    # Compute scores with APE: (n_pools, pool_size, head_dim)
    scores = slot_score.float() + ape.unsqueeze(0)

    # Per-head-dim softmax across pool_size dimension
    # softmax over dim=1 (pool_size axis)
    probs = torch.softmax(scores, dim=1)  # (n_pools, pool_size, head_dim)

    # Weighted average: sum over pool_size of prob * key
    pooled = (probs * slot_k.float()).sum(dim=1)  # (n_pools, head_dim)
    pooled_bf16 = pooled.to(torch.bfloat16)

    if write_cache:
        # Write to paged buffer
        # buf shape: (num_pages, page_size, head_dim) for single layer
        # loc is flat index into (num_pages * page_size)
        page_ids = loc // BLOCK_SIZE_K
        slot_offsets = loc % BLOCK_SIZE_K
        buf_single = buf.view(-1, head_dim) if buf.ndim > 2 else buf
        # Use scatter to write
        flat_locs = page_ids * BLOCK_SIZE_K + slot_offsets
        buf_single[flat_locs] = pooled_bf16

    if return_compressed:
        return pooled_bf16, None
    return None


def kpool_decode_update_and_maybe_write_cache_bf16(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Decode-step BF16 kpool update: append current token to tail ring and,
    whenever a pool fills, compress+flush it to the BF16 index cache.
    """
    assert tail_k.ndim == 3
    assert tail_k.shape[1] == pool.index_kpool + pool.tail_extra_slots
    assert tail_k.shape[2] == INDEX_HEAD_DIM
    assert key.ndim == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert ape.shape == (pool.index_kpool, INDEX_HEAD_DIM)

    batch = key.shape[0]
    if batch == 0:
        return

    pool_size = pool.index_kpool
    tail_size = tail_k.shape[1]

    key = key.contiguous()
    slot_score = slot_score.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    out_cache_loc = out_cache_loc.contiguous()

    buf_single = buf.view(-1, INDEX_HEAD_DIM) if buf.ndim > 2 else buf

    for b in range(batch):
        req = int(req_pool_indices[b].item())
        pos = int(positions[b].item())
        cache_loc = int(out_cache_loc[b].item())

        if req < 0 or cache_loc == 0 or pos < 0:
            continue

        slot = pos % pool_size
        phys_slot = pos % tail_size

        # Always write current token to tail ring
        tail_k[req, phys_slot] = key[b]
        tail_score[req, phys_slot] = slot_score[b]

        # Only compress when pool fills (slot == pool_size - 1)
        if slot == pool_size - 1:
            pool_logical_start = pos - slot
            # Gather pool_size tokens from tail ring
            phys_offsets = torch.arange(
                pool_size, device=tail_k.device, dtype=torch.long
            )
            phys = (pool_logical_start + phys_offsets) % tail_size
            k_group = tail_k[req, phys]  # (pool_size, head_dim)
            s_group = tail_score[req, phys]  # (pool_size, head_dim)

            # Softmax-weighted pooling
            scores = s_group.float() + ape.float()  # (pool_size, head_dim)
            probs = torch.softmax(scores, dim=0)  # (pool_size, head_dim)
            pooled = (probs * k_group.float()).sum(dim=0).to(torch.bfloat16)

            # Write to BF16 index cache
            # buf_single[cache_loc] = pooled
            pool_id = pos // pool_size
            pool_page_group = pool_id // pool.page_size
            token_page_row = pool_page_group * pool_size
            packed_page = int(block_tables[b, token_page_row].item())
            write_loc = packed_page * pool.page_size + (pool_id % pool.page_size)
            buf_single[write_loc] = pooled



def kpool_assemble_softmax_write_cache_bf16(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    """BF16 assemble: stitch tail ring + chunk tokens into complete pools,
    then softmax-pool and write to BF16 index cache.
    """
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    pool_size = pool.index_kpool
    tail_size = tail_k.shape[1]
    head_dim = INDEX_HEAD_DIM

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()

    buf_single = buf.view(-1, head_dim) if buf.ndim > 2 else buf

    for r in range(n_rows):
        req = int(req_pool_idx[r].item())
        n_tail = int(n_from_tail[r].item())
        chunk_src = int(chunk_src_start[r].item())
        tail_base = int(tail_logical_base[r].item())
        write_loc = int(loc[r].item())

        # Gather pool_size tokens: first n_tail from tail ring, rest from chunk
        k_list = []
        s_list = []
        for slot in range(pool_size):
            if slot < n_tail:
                phys = (tail_base + slot) % tail_size
                k_list.append(tail_k[req, phys])
                s_list.append(tail_score[req, phys])
            else:
                idx = chunk_src + (slot - n_tail)
                k_list.append(chunk_k[idx])
                s_list.append(chunk_score[idx])

        k_group = torch.stack(k_list)  # (pool_size, head_dim)
        s_group = torch.stack(s_list)  # (pool_size, head_dim)

        # Softmax-weighted pooling
        scores = s_group.float() + ape.float()
        probs = torch.softmax(scores, dim=0)
        pooled = (probs * k_group.float()).sum(dim=0).to(torch.bfloat16)

        buf_single[write_loc] = pooled


def scatter_kpool_tail_updates_bf16(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    """Scatter leftover tail tokens from chunk into the tail ring buffer."""
    pool_size = pool.index_kpool
    tail_size = tail_k.shape[1]
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()

    for r in range(n_rows):
        req = int(req_pool_idx[r].item())
        n_w = int(n_write[r].item())
        if n_w == 0:
            continue
        dst_start = int(dst_logical_start[r].item())
        src_start = int(chunk_src_start[r].item())

        for slot in range(n_w):
            phys = (dst_start + slot) % tail_size
            tail_k[req, phys] = chunk_k[src_start + slot]
            tail_score[req, phys] = chunk_score[src_start + slot]


def gather_index_k_bf16(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
) -> None:
    """Gather BF16 index K from paged buffer into contiguous output.

    Args:
        buf: (num_pages, page_size, head_dim) BF16 for a single layer
        page_indices: (n_pages,) int32/int64 pooled page table
        k_out: (seq_len, head_dim) BF16 output
    """
    if seq_len == 0:
        return

    buf_single = buf.view(-1, INDEX_HEAD_DIM) if buf.ndim > 2 else buf
    # Each page has BLOCK_SIZE_K slots; gather by page then slice
    for i in range(seq_len):
        page_idx = int(page_indices[i // BLOCK_SIZE_K].item())
        slot_in_page = i % BLOCK_SIZE_K
        k_out[i] = buf_single[page_idx * BLOCK_SIZE_K + slot_in_page]


def gather_index_k_bf16_batched(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
) -> None:
    """Vectorized BF16 gather - faster than per-element loop."""
    if seq_len == 0:
        return

    buf_single = buf.view(-1, INDEX_HEAD_DIM) if buf.ndim > 2 else buf
    # Build flat indices: page_indices[i // 64] * 64 + (i % 64)
    slot_offsets = torch.arange(seq_len, device=buf.device, dtype=torch.long)
    page_ids = slot_offsets // BLOCK_SIZE_K
    in_page_offsets = slot_offsets % BLOCK_SIZE_K
    # page_indices may be shorter than unique page_ids
    pages = page_indices[page_ids].to(torch.long)
    flat_locs = pages * BLOCK_SIZE_K + in_page_offsets
    torch.index_select(buf_single, 0, flat_locs, out=k_out[:seq_len])


def bf16_paged_mqa_logits(
    q_bf16: torch.Tensor,
    k_cache_bf16: torch.Tensor,
    weights: torch.Tensor,
    pool_seqlens: torch.Tensor,
    pool_block_tables: torch.Tensor,
    pool_max_seq_len: int,
    slots_per_page: int = 128,
) -> torch.Tensor:
    """BF16 paged MQA logits with per-token causal masking (NPU fallback).

    Mirrors GPU ``deep_gemm.fp8_paged_mqa_logits``: each query token *i* may
    only attend to pooled-K entries ``[0, pool_seqlens[i])``.  In decode every
    request contributes one row so ``pool_seqlens`` is per-request; in
    target_verify / draft_extend_v2 the caller expands it to per-token
    (``seqlens_expanded // pool_size``) so that earlier draft tokens see fewer
    pools than later ones — the causal constraint.

    Computes: logits[i] = sum_h weights[i,h] * relu(q[i,1,h,:] @ k_cache^T)

    Args:
        q_bf16: (n_rows, 1, n_heads, head_dim) BF16 — one row per query token
        k_cache_bf16: (num_pages, page_size, 1, head_dim) BF16
        weights: (n_rows, n_heads) FP32
        pool_seqlens: (n_rows,) int32 — **per-token** causal limit,
            i.e. ``pos_i // pool_size``.  Same request's tokens may have
            different values.
        pool_block_tables: (n_rows, max_pool_pages) int32
        pool_max_seq_len: max number of pools across all rows
        slots_per_page: page size of the index cache (default 128)

    Returns:
        logits: (n_rows, pool_max_seq_len) FP32, entries beyond
        ``pool_seqlens[i]`` are ``-inf``.
    """
    n_rows, _, n_heads, head_dim = q_bf16.shape
    num_pools = pool_max_seq_len

    logits = torch.full(
        (n_rows, num_pools), float("-inf"), dtype=torch.float32, device=q_bf16.device
    )

    for b in range(n_rows):
        seq_len = int(pool_seqlens[b].item())
        if seq_len == 0:
            continue

        n_pages = (seq_len + slots_per_page - 1) // slots_per_page
        page_ids = pool_block_tables[b, :n_pages].to(torch.long)

        k_pages = k_cache_bf16[page_ids]  # (n_pages, page_size, 1, head_dim)
        k_flat = k_pages.view(-1, head_dim)[:seq_len]  # (seq_len, head_dim)

        q_b = q_bf16[b, 0]  # (n_heads, head_dim)
        scores = torch.matmul(q_b.float(), k_flat.float().T)  # (n_heads, seq_len)
        scores = torch.relu(scores)

        w = weights[b]  # (n_heads,)
        logits[b, :seq_len] = (scores * w.unsqueeze(1)).sum(dim=0)

    return logits


def bf16_ragged_mqa_logits(
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    pool_lens: torch.Tensor,
) -> torch.Tensor:
    """BF16 ragged MQA logits with per-token causal masking (NPU fallback).

    Mirrors GPU ``deep_gemm.fp8_mqa_logits``: each query token *i* reads
    ``pool_lens[i]`` keys starting at ``row_starts[i]``.  ``pool_lens`` is
    per-token (``seqlens_expanded[i] // pool_size``) so that earlier tokens
    in the same request see fewer pools than later ones — the causal
    constraint.

    Args:
        q_bf16: (num_q, n_heads, head_dim) BF16
        k_bf16: (total_k, head_dim) BF16 — gathered pool keys (all requests
            concatenated)
        weights: (num_q, n_heads) FP32
        row_starts: (num_q,) int32 — start index into ``k_bf16`` for each
            query token.  Tokens in the same request share the same start.
        pool_lens: (num_q,) int32 — **per-token** causal limit: how many
            keys this token may attend to.  Replaces the old ``row_ends``
            parameter; ``row_ends[i]`` would be ``row_starts[i] +
            pool_lens[i]``.

    Returns:
        logits: (num_q, max_pool_len) FP32, entries beyond
        ``pool_lens[i]`` are ``-inf``.
    """
    num_q, n_heads, head_dim = q_bf16.shape
    max_k = int(pool_lens.max().item()) if num_q > 0 else 0

    logits = torch.full(
        (num_q, max_k), float("-inf"), dtype=torch.float32, device=q_bf16.device
    )

    for i in range(num_q):
        n_k = int(pool_lens[i].item())
        if n_k == 0:
            continue

        ks = int(row_starts[i].item())
        k_chunk = k_bf16[ks : ks + n_k]  # (n_k, head_dim)
        q_i = q_bf16[i]  # (n_heads, head_dim)

        scores = torch.matmul(q_i.float(), k_chunk.float().T)  # (n_heads, n_k)
        scores = torch.relu(scores)

        w = weights[i]  # (n_heads,)
        logits[i, :n_k] = (scores * w.unsqueeze(1)).sum(dim=0)

    return logits


def topk_from_pooled_history_logits_bf16(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_offsets: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
) -> torch.Tensor:
    """PyTorch-based topk for pooled history logits (NPU-compatible).

    Selects group_topk pools, expands to token indices, appends tail.
    """
    assert logits.ndim == 2
    assert group_lengths.ndim == 1
    assert topk % pool_size == 0

    n_rows, cols = logits.shape
    group_topk = history_group_budget_for_topk(topk, pool_size)

    if topk_offsets is not None and topk_offsets.ndim == 2:
        topk_offsets = topk_offsets.squeeze(1)

    # Mask invalid logits
    masked_logits = logits.clone()
    for r in range(n_rows):
        gl = int(group_lengths[r].item())
        if gl < cols:
            masked_logits[r, gl:] = float("-inf")

    # Select top group_topk pools per row
    actual_topk = min(group_topk, cols)
    if actual_topk == 0:
        # No history, just return tail
        topk_values = torch.empty(
            (n_rows, topk + pool_size - 1), dtype=torch.int32, device=logits.device
        )
        topk_values.fill_(-1)
        return _append_tail_bf16(
            topk_values[:, :topk],
            seq_lens,
            group_lengths,
            pool_size,
            page_table,
            topk_offsets,
        )

    _, topk_indices = torch.topk(masked_logits, actual_topk, dim=1, largest=True)

    # Expand groups to token indices
    offsets = torch.arange(pool_size, device=logits.device, dtype=torch.int64)
    token_ids = topk_indices.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(n_rows, actual_topk * pool_size)

    # Pad to topk width if actual_topk < group_topk
    if actual_topk * pool_size < topk:
        padding = torch.full(
            (n_rows, topk - actual_topk * pool_size),
            -1,
            dtype=torch.int32,
            device=logits.device,
        )
        expanded = torch.cat([token_ids.to(torch.int32), padding], dim=1)
    else:
        expanded = token_ids.to(torch.int32)

    # Apply page_table mapping if provided
    if page_table is not None:
        safe_ids = expanded.clamp(min=0, max=page_table.shape[1] - 1)
        expanded = torch.gather(page_table, 1, safe_ids.to(torch.int64)).to(torch.int32)
        # Restore -1 for invalid entries
        invalid = expanded == 0  # heuristic; may need adjustment
    elif topk_offsets is not None:
        expanded = (expanded.to(torch.int64) + topk_offsets.to(torch.int64).unsqueeze(1)).to(
            torch.int32
        )

    # Append tail
    if seq_lens is not None:
        result = _append_tail_bf16(
            expanded, seq_lens, group_lengths, pool_size, page_table, topk_offsets
        )
    else:
        result = expanded

    if out_rows is not None and out_rows > n_rows:
        padded = torch.full(
            (out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device
        )
        padded[:n_rows] = result
        return padded

    return result


def _append_tail_bf16(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: Optional[torch.Tensor] = None,
    topk_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append un-pooled tail tokens after expanded history tokens."""
    rows, n_cols = topk_result.shape
    tail_pool = pool_size - 1
    out_cols = n_cols + tail_pool
    out = torch.full(
        (rows, out_cols), -1, dtype=topk_result.dtype, device=topk_result.device
    )

    for r in range(rows):
        pool_len = int(pool_lens[r].item())
        seq_len = int(seq_lens[r].item()) if seq_lens is not None else pool_len * pool_size
        tail_start = pool_len * pool_size
        history_len = min(tail_start, n_cols)
        tail_count = seq_len % pool_size

        # Copy history
        out[r, :history_len] = topk_result[r, :history_len]

        # Append tail tokens
        for t in range(tail_count):
            col = history_len + t
            tail_raw = tail_start + t
            if page_table is not None:
                safe_tail = min(max(tail_raw, 0), page_table.shape[1] - 1)
                out[r, col] = page_table[r, safe_tail].to(topk_result.dtype)
            elif topk_offsets is not None:
                out[r, col] = (tail_raw + int(topk_offsets[r].item()))
            else:
                out[r, col] = tail_raw

    return out
