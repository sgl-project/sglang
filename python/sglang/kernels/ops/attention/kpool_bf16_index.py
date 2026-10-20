"""BF16 kpool index cache operations for NPU.

PyTorch-based replacements for the CUDA/FP8 Triton kernels in kpool_fp8_index.py.
Key differences from FP8 path:
  - No Hadamard128 rotation (only needed to spread FP8 quantization error)
  - No FP8 quantization (store BF16 directly)
  - No per-token FP32 scale
  - Buffer layout: (layer_num, num_pages, page_size, 1, head_dim) dtype=bfloat16
"""

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
