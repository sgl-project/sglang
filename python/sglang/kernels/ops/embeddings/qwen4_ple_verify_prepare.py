"""Qwen4 PLE target-verify layout in one launch, at any batch.

``_prepare_ple_batch`` builds a target-verify PLE layout from about 20 one-CTA torch ops
(arange, floor-div, index_select, compare, ne/any, where, full, index_put, index_select of
the N-gram history, cat) and ``compute_ngram_ids`` adds one more (unfold + advanced index).
In a captured verify graph they are about 20 launches at the graph head. This kernel writes
the same tensors from the same inputs with integer copies and compares only, so every output
is bit-identical; the hash, the PLE gather and the commit consume them unchanged.

Rows are ``n`` requests of ``W`` draft tokens (``processed = n * W``). For request ``i`` and
column ``t``:

* ``lengths[i]`` = ``extend_seq_lens[i]`` (int64) when the batch carries them, else ``W``;
* ``req_indices[i*W+t]`` = ``i``, ``token_offsets[i*W+t]`` = ``t``,
  ``valid_tokens[i*W+t]`` = ``t < lengths[i]``;
* ``state_indices[i]`` = ``mamba_map[req_pool_indices[i]]`` (int64) when
  ``lengths[i] != 0`` and any of the request's ``out_cache_loc`` is non-zero, else 0 (the
  state pools' reserved dummy slot);
* ``ngram_context[i]`` = ``history[state_indices[i]]`` (``C`` columns) followed by the ``W``
  tokens, each token replaced by EOS where ``valid_tokens`` is false;
* ``contexts[i*W+t]`` = ``ngram_context[i, t : t + C + 1]`` (``unfold(1, C + 1, 1)``).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_REQUESTS = 256
_logged = False


class VerifyPLELayout(NamedTuple):
    lengths: torch.Tensor
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: torch.Tensor
    contexts: torch.Tensor


@triton.jit
def _qwen4_ple_verify_prepare(
    Tokens,
    Lengths,
    Requests,
    Locations,
    SlotMap,
    History,
    OutLengths,
    OutRequests,
    OutOffsets,
    OutValid,
    OutSlots,
    OutContext,
    OutContexts,
    EOS: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    HAS_LOCATIONS: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    in_row = col < W
    p = i * W + col
    if HAS_LENGTHS:
        length = tl.load(Lengths + i).to(tl.int64)
    else:
        length = tl.full((), W, tl.int64)
    tl.store(OutLengths + i, length)
    tl.store(OutRequests + p, tl.zeros((BLOCK,), tl.int64) + i, in_row)
    tl.store(OutOffsets + p, col.to(tl.int64), in_row)
    valid = col.to(tl.int64) < length
    tl.store(OutValid + p, valid, in_row)

    live = length != 0
    if HAS_LOCATIONS:
        loc = tl.load(Locations + p, in_row, other=0)
        live = live & (tl.sum((loc != 0).to(tl.int32), 0) != 0)
    req = tl.load(Requests + i).to(tl.int64)
    slot = tl.load(SlotMap + req).to(tl.int64)
    slot = tl.where(live, slot, 0)
    tl.store(OutSlots + i, slot)

    # ngram_context[i] = [history[slot, :C], padded tokens]
    token = tl.load(Tokens + p, in_row, other=0)
    padded = tl.where(valid, token, EOS)
    hist_col = tl.arange(0, BLOCK)
    hist_in = hist_col < C
    history = tl.load(History + slot * C + hist_col, hist_in, other=0)
    tl.store(OutContext + i * (C + W) + hist_col, history, hist_in)
    tl.store(OutContext + i * (C + W) + C + col, padded, in_row)

    # contexts[i*W+t, k] = ngram_context[i, t + k], k in [0, C]
    k = tl.arange(0, BLOCK)
    j = col[:, None] + k[None, :]
    use = in_row[:, None] & (k[None, :] <= C)
    from_history = use & (j < C)
    from_tokens = use & (j >= C)
    tj = j - C
    hist_val = tl.load(History + slot * C + j, from_history, other=0)
    tok_val = tl.load(Tokens + i * W + tj, from_tokens, other=0)
    tok_val = tl.where(tj.to(tl.int64) < length, tok_val, EOS)
    value = tl.where(j < C, hist_val, tok_val)
    tl.store(OutContexts + p[:, None] * (C + 1) + k[None, :], value, use)


def can_prepare(tokens, lengths, requests, locations, slot_map, history, width, ngram_size):
    n = requests.numel()
    return (
        ngram_size == 3
        and 0 < width <= 8
        and tokens.is_cuda
        and tokens.dtype == torch.int64
        and tokens.ndim == 1
        and tokens.is_contiguous()
        and tokens.numel() > 0
        and tokens.numel() % width == 0
        and n == tokens.numel() // width
        and 0 < n <= _MAX_REQUESTS
        and requests.dtype in (torch.int32, torch.int64)
        and requests.is_contiguous()
        and (
            lengths is None
            or (
                lengths.numel() == n
                and lengths.dtype in (torch.int32, torch.int64)
                and lengths.is_contiguous()
            )
        )
        and (
            locations is None
            or (locations.numel() == tokens.numel() and locations.is_contiguous())
        )
        and slot_map.dtype in (torch.int32, torch.int64)
        and slot_map.is_contiguous()
        and history is not None
        and history.dtype == torch.int64
        and history.ndim == 2
        and history.shape[1] == ngram_size - 1
        and history.is_contiguous()
        and all(
            t.device == tokens.device
            for t in (requests, slot_map, history)
            + ((lengths,) if lengths is not None else ())
            + ((locations,) if locations is not None else ())
        )
    )


def prepare_verify_layout(tokens, lengths, requests, locations, slot_map, history, width, eos):
    """Caller checked ``can_prepare``; ``tokens`` are the ``n * width`` processed tokens and
    ``locations`` (or None) their ``out_cache_loc``."""
    global _logged
    device = tokens.device
    n = requests.numel()
    processed = tokens.numel()
    ctx = history.shape[1]
    out = VerifyPLELayout(
        torch.empty((n,), dtype=torch.int64, device=device),
        torch.empty((processed,), dtype=torch.int64, device=device),
        torch.empty((processed,), dtype=torch.int64, device=device),
        torch.empty((processed,), dtype=torch.bool, device=device),
        torch.empty((n,), dtype=torch.int64, device=device),
        torch.empty((n, ctx + width), dtype=torch.int64, device=device),
        torch.empty((processed, ctx + 1), dtype=torch.int64, device=device),
    )
    _qwen4_ple_verify_prepare[(n,)](
        tokens,
        lengths if lengths is not None else requests,
        requests,
        locations if locations is not None else tokens,
        slot_map,
        history,
        *out,
        EOS=int(eos),
        W=int(width),
        C=int(ctx),
        BLOCK=max(4, triton.next_power_of_2(max(int(width), int(ctx) + 1))),
        HAS_LENGTHS=lengths is not None,
        HAS_LOCATIONS=locations is not None,
        num_warps=1,
    )
    if not _logged:
        _logged = True
        logger.info(
            "Qwen4 PLE verify prepare: one launch at batch %d x %d "
            "(SGLANG_ENABLE_Q38_PLE_VERIFY_PREPARE_BATCHED)",
            n,
            width,
        )
    return out
