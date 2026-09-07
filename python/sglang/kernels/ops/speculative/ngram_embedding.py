from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_npu

import torch
if TYPE_CHECKING:
    from tvm_ffi.module import Module

_is_npu = is_npu()


@cache_once
def _jit_ngram_embedding_module() -> Module:
    return load_jit(
        "ngram_embedding",
        cuda_files=["speculative/ngram_embedding.cuh"],
        cuda_wrappers=[
            ("compute_n_gram_ids", "&NgramEmbeddingKernel::compute_n_gram_ids"),
            (
                "compute_n_gram_ids_decode",
                "&NgramEmbeddingKernel::compute_n_gram_ids_decode",
            ),
            ("update_token_table", "&NgramEmbeddingKernel::update_token_table"),
            (
                "update_token_table_decode",
                "&NgramEmbeddingKernel::update_token_table_decode",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Pure-torch fallback implementations (used on NPU where CUDA JIT is
# unavailable).  These mirror the four CUDA kernels in ngram_embedding.cuh
# exactly: same indexing, same break conditions, same modular arithmetic.
# ---------------------------------------------------------------------------


def _torch_update_token_table(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    ignore_tokens: torch.Tensor | None = None,
) -> None:
    max_context_len = ne_token_table.shape[1]
    batch_size = req_lens.shape[0]
    starts = torch.zeros(batch_size + 1, dtype=torch.int32, device=tokens.device)
    starts[1:] = torch.cumsum(req_lens, dim=0)
    has_ignore = ignore_tokens is not None and ignore_tokens.numel() > 0
    for req_id in range(batch_size):
        s = starts[req_id].item()
        e = starts[req_id + 1].item()
        if e <= s:
            continue
        token_slice = tokens[s:e]
        row = row_indices[req_id].item()
        col = column_starts[req_id].item()
        ne_token_table[row, col : col + (e - s)] = token_slice
        if has_ignore:
            mask = torch.isin(token_slice, ignore_tokens)
            if mask.any():
                ne_token_table[row, col : col + (e - s)][mask] = -token_slice[mask]


def _torch_update_token_table_decode(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
) -> None:
    batch_size = tokens.shape[0]
    for req_id in range(batch_size):
        row = row_indices[req_id].item()
        col = column_starts[req_id].item()
        ne_token_table[row, col] = tokens[req_id]


def _torch_compute_n_gram_ids(
    ne_n: int,
    ne_k: int,
    ne_weights: torch.Tensor,
    ne_mods: torch.Tensor,
    exclusive_ne_embedder_size_sums: torch.Tensor,
    tokens: torch.Tensor,
    exclusive_req_len_sums: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    n_gram_ids: torch.Tensor,
    eos_token_id: int,
) -> None:
    batch_size = exclusive_req_len_sums.shape[0] - 1
    for n in range(ne_n - 1):
        for k in range(ne_k):
            mod_val = ne_mods[n, k].item()
            w_row = ne_weights[n, k]
            emb_offset = exclusive_ne_embedder_size_sums[n * ne_k + k].item()
            for req_id in range(batch_size):
                req_start = exclusive_req_len_sums[req_id].item()
                req_end = exclusive_req_len_sums[req_id + 1].item()
                num_tokens = req_end - req_start
                if num_tokens <= 0:
                    continue
                row = row_indices[req_id].item()
                col = column_starts[req_id].item()
                table_row = ne_token_table[row]
                ng = torch.zeros(num_tokens, dtype=torch.int64, device=tokens.device)
                active = torch.ones(num_tokens, dtype=torch.bool, device=tokens.device)
                for j in range(n + 2):
                    valid_start = max(0, j - col)
                    shifted = torch.full(
                        (num_tokens,), -1, dtype=torch.int32, device=tokens.device
                    )
                    if num_tokens > valid_start:
                        src_start = max(0, col - j)
                        cnt = num_tokens - valid_start
                        shifted[valid_start:] = table_row[src_start : src_start + cnt]
                    active = active & (shifted >= 0)
                    if j > 0:
                        active = active & (shifted != eos_token_id)
                    w = w_row[j].item()
                    contrib = (shifted.to(torch.int64) * w) % mod_val
                    ng[active] += contrib[active]
                ng = (ng % mod_val + emb_offset).to(torch.int32)
                n_gram_ids[req_start:req_end, n * ne_k + k] = ng


def _torch_compute_n_gram_ids_decode(
    ne_n: int,
    ne_k: int,
    ne_weights: torch.Tensor,
    ne_mods: torch.Tensor,
    exclusive_ne_embedder_size_sums: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    n_gram_ids: torch.Tensor,
    eos_token_id: int,
) -> None:
    batch_size = row_indices.shape[0]
    if batch_size <= 0:
        return
    for n in range(ne_n - 1):
        for k in range(ne_k):
            mod_val = ne_mods[n, k].item()
            w_row = ne_weights[n, k]
            emb_offset = exclusive_ne_embedder_size_sums[n * ne_k + k].item()
            ng = torch.zeros(batch_size, dtype=torch.int64, device=ne_token_table.device)
            active = torch.ones(batch_size, dtype=torch.bool, device=ne_token_table.device)
            for j in range(n + 2):
                valid = column_starts - j >= 0
                shifted = torch.full(
                    (batch_size,), -1, dtype=torch.int32, device=ne_token_table.device
                )
                if valid.any():
                    vi = valid.nonzero(as_tuple=True)[0]
                    shifted[vi] = ne_token_table[row_indices[vi], column_starts[vi] - j]
                active = active & valid & (shifted >= 0)
                if j > 0:
                    active = active & (shifted != eos_token_id)
                w = w_row[j].item()
                contrib = (shifted.to(torch.int64) * w) % mod_val
                ng[active] += contrib[active]
            ng = (ng % mod_val + emb_offset).to(torch.int32)
            n_gram_ids[:, n * ne_k + k] = ng


@debug_kernel_api
def compute_n_gram_ids(
    ne_n: int,
    ne_k: int,
    ne_weights: torch.Tensor,
    ne_mods: torch.Tensor,
    exclusive_ne_embedder_size_sums: torch.Tensor,
    tokens: torch.Tensor,
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
        tokens: input token ids
        exclusive_req_len_sums: exclusive sum of request lengths
        ne_token_table: token table for all requests
        row_indices: row indices for each request
        column_starts: column start positions for each request
        n_gram_ids: output tensor for n-gram ids
        eos_token_id: tokens before an eos are excluded from the n-gram context
    """
    if _is_npu:
        _torch_compute_n_gram_ids(
            ne_n,
            ne_k,
            ne_weights,
            ne_mods,
            exclusive_ne_embedder_size_sums,
            tokens,
            exclusive_req_len_sums,
            ne_token_table,
            row_indices,
            column_starts,
            n_gram_ids,
            eos_token_id,
        )
        return
    module = _jit_ngram_embedding_module()
    module.compute_n_gram_ids(
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        tokens,
        exclusive_req_len_sums,
        ne_token_table,
        row_indices,
        column_starts,
        n_gram_ids,
        eos_token_id,
    )


@debug_kernel_api
def compute_n_gram_ids_decode(
    ne_n: int,
    ne_k: int,
    ne_weights: torch.Tensor,
    ne_mods: torch.Tensor,
    exclusive_ne_embedder_size_sums: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    n_gram_ids: torch.Tensor,
    eos_token_id: int,
) -> None:
    """
    Compute n-gram IDs for decode, where each request contributes one token.
    """
    if _is_npu:
        _torch_compute_n_gram_ids_decode(
            ne_n,
            ne_k,
            ne_weights,
            ne_mods,
            exclusive_ne_embedder_size_sums,
            ne_token_table,
            row_indices,
            column_starts,
            n_gram_ids,
            eos_token_id,
        )
        return
    module = _jit_ngram_embedding_module()
    module.compute_n_gram_ids_decode(
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        ne_token_table,
        row_indices,
        column_starts,
        n_gram_ids,
        eos_token_id,
    )


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
        tokens: input token ids
        ne_token_table: token table for all requests
        row_indices: row indices for each request
        column_starts: column start positions for each request
        req_lens: request lengths
        ignore_tokens: tokens to be ignored (marked as negative in table)
    """
    if _is_npu:
        if ignore_tokens is None:
            ignore_tokens = tokens.new_empty(0, dtype=tokens.dtype)
        _torch_update_token_table(
            tokens,
            ne_token_table,
            row_indices,
            column_starts,
            req_lens,
            ignore_tokens,
        )
        return
    module = _jit_ngram_embedding_module()
    if ignore_tokens is None:
        # Create an empty tensor for ignore_tokens
        ignore_tokens = tokens.new_empty(0, dtype=tokens.dtype)
    module.update_token_table(
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
        req_lens,
        ignore_tokens,
    )


@debug_kernel_api
def update_token_table_decode(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
) -> None:
    """
    Update one decoded token per request in the ngram embedding token table.

    This is the decode-only fast path for req_lens == 1 and no ignored tokens.
    """
    if _is_npu:
        _torch_update_token_table_decode(
            tokens,
            ne_token_table,
            row_indices,
            column_starts,
        )
        return
    module = _jit_ngram_embedding_module()
    module.update_token_table_decode(
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
    )
