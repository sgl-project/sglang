"""Triton gather kernel for the kv-canary token-id validator.

Replaces the original host-side D2H/H2D fill of ``ExpectedInputs.tokens``: the
host populates ``pool`` and ``valid_lens`` outside the cuda graph in
``pre_ops_outside_graph``; this kernel runs inside the graph and gathers each
token's expected id from those device tensors, falling back to
``input_ids[i]`` for positions outside the valid range so the canary write
kernel's compare is tautological for speculative drafts / bonus tail / pre-fill
warmup.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.consts import REQ_POOL_IDX_PADDING

# Upper bound on bs covered by one program's per-req reduction. Matches the
# plan kernel's _PLAN_BS_BLOCK_SIZE so the same per-forward capacity flows
# through verify / write / gather.
_GATHER_BS_BLOCK_SIZE: int = 4096

# Per-program token tile width. Chosen empirically; the grid is sized to cover
# ``num_tokens`` exactly so we never iterate further than needed.
_GATHER_TOKEN_BLOCK_SIZE: int = 256


def launch_gather_expected_tokens_kernel(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    pool: torch.Tensor,
    valid_lens: torch.Tensor,
    input_ids: torch.Tensor,
    mode_offset: int,
    out_expected_tokens: torch.Tensor,
    num_tokens: int,
) -> None:
    """Fill ``out_expected_tokens[:num_tokens]`` from ``pool`` / ``valid_lens``.

    Args:
        req_pool_indices: ``[bs_capacity]`` int64; 0 = padding sentinel.
        prefix_lens: ``[bs_capacity]`` int64; per-req prefix already written.
        extend_seq_lens: ``[bs_capacity]`` int64; per-req tokens this forward.
            ``sum(extend_seq_lens[:bs]) == num_tokens``.
        pool: ``[max_reqs, max_context_len]`` int32 source-of-truth tokens
            indexed by ``req_pool_indices`` entries (NOT by row position).
        valid_lens: ``[max_reqs]`` int32; ``valid_lens[req_idx]`` is the live
            length of ``pool[req_idx]``. Rows with 0 fall through to the
            ``input_ids`` tautological fallback for every token.
        input_ids: ``[num_tokens_capacity]`` int64 fallback source.
        mode_offset: ``0`` or ``1``; added to per-token ``pos_in_seq`` to land
            on the right pool row (EAGLE draft-prefill rotates by +1).
        out_expected_tokens: ``[num_tokens_capacity]`` int64, overwritten in
            ``[:num_tokens]`` only; trailing capacity is left untouched.
        num_tokens: Current forward's token count. Captured at cuda-graph
            capture time and baked into the recorded launch.
    """
    if num_tokens <= 0:
        return

    bs_capacity = int(req_pool_indices.shape[0])
    if bs_capacity > _GATHER_BS_BLOCK_SIZE:
        raise ValueError(
            f"kv-canary: gather_expected_tokens supports at most "
            f"bs_capacity={_GATHER_BS_BLOCK_SIZE} reqs per launch, got "
            f"{bs_capacity}. Bump _GATHER_BS_BLOCK_SIZE if real workloads "
            f"need this."
        )
    _validate_inputs(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        pool=pool,
        valid_lens=valid_lens,
        input_ids=input_ids,
        out_expected_tokens=out_expected_tokens,
        bs_capacity=bs_capacity,
        num_tokens=num_tokens,
    )

    pool_stride0 = int(pool.stride(0))
    grid = (triton.cdiv(num_tokens, _GATHER_TOKEN_BLOCK_SIZE),)
    _gather_expected_tokens_kernel[grid](
        req_pool_indices,
        prefix_lens,
        extend_seq_lens,
        pool,
        valid_lens,
        input_ids,
        out_expected_tokens,
        num_tokens,
        bs_capacity,
        pool_stride0,
        int(mode_offset),
        BS_BLOCK=_GATHER_BS_BLOCK_SIZE,
        TOKEN_BLOCK=_GATHER_TOKEN_BLOCK_SIZE,
        REQ_POOL_IDX_PADDING=REQ_POOL_IDX_PADDING,
    )


def _validate_inputs(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    pool: torch.Tensor,
    valid_lens: torch.Tensor,
    input_ids: torch.Tensor,
    out_expected_tokens: torch.Tensor,
    bs_capacity: int,
    num_tokens: int,
) -> None:
    if req_pool_indices.dtype != torch.int64:
        raise ValueError(
            f"kv-canary: req_pool_indices must be int64, got {req_pool_indices.dtype}"
        )
    if prefix_lens.dtype != torch.int64:
        raise ValueError(
            f"kv-canary: prefix_lens must be int64, got {prefix_lens.dtype}"
        )
    if extend_seq_lens.dtype != torch.int64:
        raise ValueError(
            f"kv-canary: extend_seq_lens must be int64, got {extend_seq_lens.dtype}"
        )
    if pool.dtype != torch.int32:
        raise ValueError(f"kv-canary: pool must be int32, got {pool.dtype}")
    if pool.ndim != 2:
        raise ValueError(f"kv-canary: pool must be 2-D, got shape {tuple(pool.shape)}")
    if valid_lens.dtype != torch.int32:
        raise ValueError(f"kv-canary: valid_lens must be int32, got {valid_lens.dtype}")
    if input_ids.dtype != torch.int64:
        raise ValueError(f"kv-canary: input_ids must be int64, got {input_ids.dtype}")
    if out_expected_tokens.dtype != torch.int64:
        raise ValueError(
            f"kv-canary: out_expected_tokens must be int64, got "
            f"{out_expected_tokens.dtype}"
        )
    if prefix_lens.shape[0] != bs_capacity or extend_seq_lens.shape[0] != bs_capacity:
        raise ValueError(
            f"kv-canary: prefix_lens/extend_seq_lens length must match "
            f"req_pool_indices length {bs_capacity}, got "
            f"{prefix_lens.shape[0]} / {extend_seq_lens.shape[0]}"
        )
    if int(out_expected_tokens.shape[0]) < num_tokens:
        raise ValueError(
            f"kv-canary: out_expected_tokens length "
            f"{int(out_expected_tokens.shape[0])} < num_tokens {num_tokens}"
        )
    if int(input_ids.shape[0]) < num_tokens:
        raise ValueError(
            f"kv-canary: input_ids length {int(input_ids.shape[0])} < "
            f"num_tokens {num_tokens}"
        )


@triton.jit
def _gather_expected_tokens_kernel(
    req_pool_indices_ptr,
    prefix_lens_ptr,
    extend_seq_lens_ptr,
    pool_ptr,
    valid_lens_ptr,
    input_ids_ptr,
    out_expected_tokens_ptr,
    num_tokens,
    bs_capacity,
    pool_stride0,
    mode_offset,
    BS_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    REQ_POOL_IDX_PADDING: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offs = pid * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)  # [TOKEN_BLOCK]
    token_mask = token_offs < num_tokens  # [TOKEN_BLOCK]

    # Load per-req metadata for all bs slots in this program. Padding rows
    # (rpi == REQ_POOL_IDX_PADDING) contribute extend_lens=0 below so they
    # never own a token in the exclusive cumsum.
    bs_offs = tl.arange(0, BS_BLOCK)  # [BS_BLOCK]
    bs_mask = bs_offs < bs_capacity  # [BS_BLOCK]
    rpi = tl.load(
        req_pool_indices_ptr + bs_offs, mask=bs_mask, other=REQ_POOL_IDX_PADDING
    )
    prefix_lens = tl.load(prefix_lens_ptr + bs_offs, mask=bs_mask, other=0)
    extend_lens = tl.load(extend_seq_lens_ptr + bs_offs, mask=bs_mask, other=0)

    is_active = (rpi != REQ_POOL_IDX_PADDING) & bs_mask
    extend_lens = tl.where(is_active, extend_lens, 0)
    inclusive = tl.cumsum(extend_lens, axis=0)  # [BS_BLOCK]
    exclusive = inclusive - extend_lens  # [BS_BLOCK]

    # Per-token req lookup: req_id = sum(exclusive <= i) - 1, where the sum
    # counts the number of req slots whose run starts at or before token i.
    # Inactive slots have extend_lens=0 so their inclusive==exclusive — they
    # cannot satisfy ``inclusive > i`` for any live i, so they are excluded
    # from the count by clamping with ``extend_lens > 0`` below.
    i_col = token_offs[:, None]  # [TOKEN_BLOCK, 1]
    excl_row = exclusive[None, :]  # [1, BS_BLOCK]
    incl_row = inclusive[None, :]  # [1, BS_BLOCK]
    has_run = (extend_lens > 0)[None, :]  # [1, BS_BLOCK]
    owns = has_run & (excl_row <= i_col) & (i_col < incl_row)  # [TOKEN_BLOCK, BS_BLOCK]

    # Reduce: locate req_id per token. Each row of ``owns`` has at most one
    # True entry; multiply by bs index and sum to extract it. Tokens past
    # num_tokens get bogus req_id=0 but are masked out at the final store.
    bs_idx_row = bs_offs[None, :]  # [1, BS_BLOCK]
    req_id = tl.sum(tl.where(owns, bs_idx_row, 0), axis=1)  # [TOKEN_BLOCK]
    req_start = tl.sum(tl.where(owns, excl_row, 0), axis=1)  # [TOKEN_BLOCK]

    # Per-token req-level data via gather. Out-of-range tokens read slot 0
    # but their result is dropped by token_mask at store time.
    safe_req_id = tl.where(token_mask, req_id, 0)
    rpi_per_token = tl.load(
        req_pool_indices_ptr + safe_req_id, mask=token_mask, other=REQ_POOL_IDX_PADDING
    )  # [TOKEN_BLOCK]
    prefix_per_token = tl.load(
        prefix_lens_ptr + safe_req_id, mask=token_mask, other=0
    )  # [TOKEN_BLOCK]

    pos_in_seq = prefix_per_token + (token_offs - req_start)
    logical_pos = pos_in_seq + mode_offset
    vlen = tl.load(valid_lens_ptr + rpi_per_token, mask=token_mask, other=0).to(
        tl.int64
    )  # [TOKEN_BLOCK]
    in_range = (logical_pos >= 0) & (logical_pos < vlen)

    safe_pos = tl.where(in_range, logical_pos, 0)
    pool_gather = tl.load(
        pool_ptr + rpi_per_token * pool_stride0 + safe_pos,
        mask=token_mask & in_range,
        other=0,
    ).to(
        tl.int64
    )  # [TOKEN_BLOCK]

    fallback = tl.load(input_ids_ptr + token_offs, mask=token_mask, other=0)
    out = tl.where(in_range, pool_gather, fallback)
    tl.store(out_expected_tokens_ptr + token_offs, out, mask=token_mask)
