# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's tiled PLE hash using SGLang's row offsets and padded history."""
import torch
import triton
import triton.language as tl


@triton.jit
def _ple_window_ids_kernel(
    contexts_ptr, request_rows, offsets, multipliers_ptr, sizes_ptr, offsets_ptr,
    out_ptr, num_tokens, row_stride, eos_token_id,
    NGRAM_CONTEXT_LEN: tl.constexpr, HEADS_PER_NGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr,
):
    NGRAM_HEADS: tl.constexpr = NGRAM_CONTEXT_LEN * HEADS_PER_NGRAM
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < num_tokens
    req = tl.load(request_rows + token_offsets, mask=token_mask, other=0).to(tl.int64)
    pos = tl.load(offsets + token_offsets, mask=token_mask, other=0).to(tl.int64)
    current = req * row_stride + pos + NGRAM_CONTEXT_LEN
    current_token = tl.load(contexts_ptr + current, mask=token_mask, other=0).to(tl.int64)
    current_multiplier = tl.load(multipliers_ptr)
    mixed = current_token[:, None] * current_multiplier
    g = tl.arange(0, BLOCK_H)
    head_mask = g < NGRAM_HEADS
    ngram_order = g // HEADS_PER_NGRAM + 2

    # Read the padded request row backwards, including history from the preceding chunk.
    crossed = tl.zeros([BLOCK_T], tl.int1)
    for shift in tl.static_range(1, NGRAM_CONTEXT_LEN + 1):
        candidate = tl.load(contexts_ptr + current - shift, mask=token_mask, other=0).to(tl.int64)
        # Older positions remain behind the first EOS boundary.
        candidate = tl.where(crossed, eos_token_id, candidate)
        crossed = crossed | (candidate == eos_token_id)
        multiplier = tl.load(multipliers_ptr + shift)
        term = candidate[:, None] * multiplier
        mixed = mixed ^ tl.where((ngram_order > shift)[None, :], term, 0)

    # Map each hash into its embedding-table partition.
    sizes = tl.load(sizes_ptr + g, mask=head_mask, other=1)[None, :]
    head_offsets = tl.load(offsets_ptr + g, mask=head_mask, other=0)[None, :]
    # Hash products may overflow int64; preserve torch.remainder semantics.
    remainders = mixed % sizes
    remainders = tl.where(remainders < 0, remainders + sizes, remainders)
    ids = remainders + head_offsets
    tl.store(out_ptr + token_offsets[:, None] * NGRAM_HEADS + g[None, :], ids,
             mask=token_mask[:, None] & head_mask[None, :])


def ple_window_ids(contexts, request_rows, offsets, multipliers, sizes, head_offsets, eos):
    tokens = request_rows.numel()
    out = torch.empty((tokens, 16), dtype=torch.long, device=contexts.device)
    if tokens:
        _ple_window_ids_kernel[(triton.cdiv(tokens, 16),)](
            contexts, request_rows, offsets, multipliers, sizes, head_offsets, out, tokens,
            contexts.stride(0), eos, NGRAM_CONTEXT_LEN=2, HEADS_PER_NGRAM=8, BLOCK_H=16, BLOCK_T=16,
            num_warps=4,
        )
    return out
