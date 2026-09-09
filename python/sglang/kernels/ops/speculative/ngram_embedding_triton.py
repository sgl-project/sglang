"""Triton implementation of the n-gram embedding ops.

Drop-in counterpart of the CUDA JIT module in ``ngram_embedding.py``; the
kernels are semantically equivalent to ``ComputeNGramIdsKernel`` and
``UpdateTokenTableKernel`` in ``jit/csrc/speculative/ngram_embedding.cuh``:

* ``update_token_table`` writes each request's new tokens into its token-table
  row at ``column_start + offset``; tokens present in ``ignore_tokens`` are
  stored negated so that later lookups stop at them.
* ``compute_n_gram_ids`` walks, for every (n, k) configuration, the table
  backwards from the current token: the walk stops at the request row start, at
  a negated (ignored) token, or at an eos token (only when looking back, the
  current token itself is allowed); each surviving token contributes
  ``(token * weight_j) % mod`` and the summed residue is offset by the
  configuration's embedder base.

Both kernels read the token *table* (not the token stream) for the lookback, so
context written by previous extends/chunks is visible, exactly as in the CUDA
version.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.kernel_api_logging import debug_kernel_api


@triton.jit
def _update_token_table_kernel(
    tokens_ptr,  # [token_num] int32
    table_ptr,  # [max_running_reqs, max_context_len] int32
    row_indices_ptr,  # [batch] int64
    column_starts_ptr,  # [batch] int32
    req_lens_ptr,  # [batch] int32
    ignore_tokens_ptr,  # [ignore_token_num] int32
    ignore_token_num,
    max_context_len,
    BLOCK_T: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req_id = tl.program_id(0)

    # start = sum(req_lens[:req_id]); req_lens can be large, so chunk the sum.
    # (tl.load of an in-bounds scalar gives the 0-valued int64 scalar the loop
    # carries; 0-d tl.zeros is not portable across Triton versions.)
    start = (tl.load(req_lens_ptr + req_id) * 0).to(tl.int64)
    for b0 in range(0, req_id, BLOCK_B):
        offs_b = b0 + tl.arange(0, BLOCK_B)
        lens = tl.load(req_lens_ptr + offs_b, mask=offs_b < req_id, other=0)
        start += tl.sum(lens.to(tl.int64))
    req_len = tl.load(req_lens_ptr + req_id).to(tl.int64)

    row = tl.load(row_indices_ptr + req_id)  # int64
    col0 = tl.load(column_starts_ptr + req_id).to(tl.int64)
    row_base = row * max_context_len

    for t0 in range(0, req_len, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T)
        m = offs < req_len
        tok = tl.load(tokens_ptr + start + offs, mask=m, other=0)

        # Negate tokens found in ignore_tokens (the loop body never runs when
        # ignore_token_num == 0). -1 is a safe sentinel: valid ids are >= 0.
        hit = tl.zeros([BLOCK_T], dtype=tl.int32)
        for n0 in range(0, ignore_token_num, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nm = offs_n < ignore_token_num
            ig = tl.load(ignore_tokens_ptr + offs_n, mask=nm, other=-1)
            match = (tok[:, None] == ig[None, :]) & nm[None, :]
            hit += tl.sum(match.to(tl.int32), axis=1)
        tok = tl.where(hit > 0, -tok, tok)

        dst = row_base + col0 + offs
        tl.store(table_ptr + dst, tok, mask=m)


@triton.jit
def _compute_n_gram_ids_kernel(
    weights_ptr,  # [ne_n-1, ne_k, ne_n] int32
    mods_ptr,  # [ne_n-1, ne_k] int32
    embed_sums_ptr,  # [(ne_n-1)*ne_k (+1)] int32
    excl_req_ptr,  # [batch+1] int32
    table_ptr,  # [max_running_reqs, max_context_len] int32
    row_indices_ptr,  # [batch] int64
    column_starts_ptr,  # [batch] int32
    out_ptr,  # [token_num, (ne_n-1)*ne_k] int32
    eos_token_id,
    max_context_len,
    ne_n,
    ne_k,
    num_configs,
    JCAP: tl.constexpr,  # >= max n-gram order (next_pow2(ne_n))
    BLOCK_T: tl.constexpr,
):
    config = tl.program_id(0)  # n * ne_k + k
    req_id = tl.program_id(1)
    k_idx = config % ne_k
    n_idx = config // ne_k
    terms = n_idx + 2  # lookback distance j = 0 .. n_idx+1

    mod = tl.load(mods_ptr + config).to(tl.int64)
    esum = tl.load(embed_sums_ptr + config).to(tl.int64)

    # ne_weights layout is [ne_n-1, ne_k, ne_n] and (n*ne_k + k)*ne_n == config*ne_n.
    J = tl.arange(0, JCAP)
    jm = J < terms
    w = tl.load(weights_ptr + config * ne_n + J, mask=jm, other=0).to(tl.int64)

    s = tl.load(excl_req_ptr + req_id).to(tl.int64)
    e = tl.load(excl_req_ptr + req_id + 1).to(tl.int64)
    req_len = e - s

    row = tl.load(row_indices_ptr + req_id)  # int64
    col0 = tl.load(column_starts_ptr + req_id).to(tl.int64)
    row_base = row * max_context_len

    for t0 in range(0, req_len, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T)
        m = offs < req_len

        # Column index of token (col0 + offs) minus lookback j; the walk must
        # stay inside the request's row (col >= 0).
        idx = (col0 + offs)[:, None] - J[None, :]
        in_rng = (idx >= 0) & jm[None, :]
        tok = tl.load(table_ptr + row_base + idx, mask=in_rng, other=-1)

        # A position contributes iff itself and every earlier j are valid; the
        # CUDA loop breaks at the first failure, which is a prefix condition.
        valid = in_rng & (tok >= 0) & (((tok == eos_token_id) & (J[None, :] > 0)) == 0)
        bad_prefix = tl.cumsum((valid == 0).to(tl.int32), axis=1)

        term = (tok.to(tl.int64) * w[None, :]) % mod
        contrib = tl.where(bad_prefix == 0, term, 0)
        acc = tl.sum(contrib, axis=1)

        out = (acc % mod + esum).to(tl.int32)
        gidx = s + offs
        tl.store(out_ptr + gidx * num_configs + config, out, mask=m)


@debug_kernel_api
def update_token_table(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    ignore_tokens: torch.Tensor | None = None,
) -> None:
    """
    Update the token table with new tokens.

    Args:
        tokens: input token ids [token_num] int32
        ne_token_table: token table for all requests [max_running_reqs, max_context_len] int32
        row_indices: row indices for each request [batch] int64
        column_starts: column start positions for each request [batch] int32
        req_lens: request lengths [batch] int32
        ignore_tokens: tokens to be ignored (marked as negative in table) [n] int32
    """
    assert tokens.dtype == torch.int32 and tokens.ndim == 1
    assert ne_token_table.dtype == torch.int32 and ne_token_table.ndim == 2
    assert row_indices.dtype == torch.int64 and row_indices.ndim == 1
    assert column_starts.dtype == torch.int32 and column_starts.ndim == 1
    assert req_lens.dtype == torch.int32 and req_lens.ndim == 1
    batch_size = req_lens.shape[0]
    if batch_size <= 0:
        return
    if ignore_tokens is None:
        ignore_tokens = tokens.new_empty(0, dtype=tokens.dtype)
    assert ignore_tokens.dtype == torch.int32 and ignore_tokens.ndim == 1

    max_context_len = ne_token_table.shape[1]
    _update_token_table_kernel[(batch_size,)](
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
        req_lens,
        ignore_tokens,
        ignore_tokens.numel(),
        max_context_len,
        BLOCK_T=128,
        BLOCK_B=512,
        BLOCK_N=128,
        num_warps=4,
    )


@debug_kernel_api
def compute_n_gram_ids(
    ne_n: int,
    ne_k: int,
    ne_weights: torch.Tensor,
    ne_mods: torch.Tensor,
    exclusive_ne_embedder_size_sums: torch.Tensor,
    tokens: torch.Tensor,  # unused by the kernel; kept for signature parity
    exclusive_req_len_sums: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    n_gram_ids: torch.Tensor,
    eos_token_id: int,
) -> None:
    """
    Compute n-gram IDs for embedding.

    Args:
        ne_n: n value for n-gram
        ne_k: k value for n-gram configurations
        ne_weights: weights tensor with shape [ne_n-1, ne_k, ne_n]
        ne_mods: mods tensor with shape [ne_n-1, ne_k]
        exclusive_ne_embedder_size_sums: exclusive sum of embedder sizes
        tokens: input token ids (lookups read the token table, not this)
        exclusive_req_len_sums: exclusive sum of request lengths [batch+1]
        ne_token_table: token table for all requests [max_running_reqs, max_context_len]
        row_indices: row indices for each request [batch]
        column_starts: column start positions for each request [batch]
        n_gram_ids: output tensor for n-gram ids [token_num, (ne_n-1)*ne_k]
        eos_token_id: tokens before an eos are excluded from the n-gram context
    """
    assert ne_n >= 2 and ne_k >= 1
    assert ne_weights.dtype == torch.int32 and ne_weights.shape == (
        ne_n - 1,
        ne_k,
        ne_n,
    )
    assert ne_mods.dtype == torch.int32 and ne_mods.shape == (ne_n - 1, ne_k)
    assert exclusive_ne_embedder_size_sums.dtype == torch.int32
    assert exclusive_req_len_sums.dtype == torch.int32 and exclusive_req_len_sums.ndim == 1
    assert ne_token_table.dtype == torch.int32 and ne_token_table.ndim == 2
    assert row_indices.dtype == torch.int64 and row_indices.ndim == 1
    assert column_starts.dtype == torch.int32 and column_starts.ndim == 1
    assert n_gram_ids.dtype == torch.int32 and n_gram_ids.ndim == 2

    batch_size = exclusive_req_len_sums.shape[0] - 1
    if batch_size <= 0:
        return
    num_configs = (ne_n - 1) * ne_k
    max_context_len = ne_token_table.shape[1]
    jcap = max(triton.next_power_of_2(int(ne_n)), 2)

    _compute_n_gram_ids_kernel[(num_configs, batch_size)](
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        exclusive_req_len_sums,
        ne_token_table,
        row_indices,
        column_starts,
        n_gram_ids,
        eos_token_id,
        max_context_len,
        ne_n,
        ne_k,
        num_configs,
        JCAP=jcap,
        BLOCK_T=128,
        num_warps=4,
    )
