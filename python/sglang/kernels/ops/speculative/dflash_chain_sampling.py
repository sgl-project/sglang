"""ROCm implementation of DFLASH's chain-specialised target-only verifier.

Drop-in replacement for the CUDA-only sgl_kernel operator
``tree_speculative_sampling_target_only`` when it is used by DFLASH, whose
proposals are linear (``topk == 1``). It keeps the operator's keyword signature
so the call site in ``dflash_utils.py`` is unchanged.

Two observations make the chain case far simpler than the general tree kernel
it replaces:

*   ``retrive_next_sibling`` is always ``-1`` and ``retrive_next_token[i] == i+1``,
    so the inner sibling loop of the CUDA kernel executes exactly once per level
    and the walk stops at the first rejection.
*   ``prob_acc`` is reset to zero on every acceptance, so on a chain it never
    accumulates across levels. The acceptance test at level ``j`` reduces to a
    purely local predicate on that level's target probability::

        tps_j  = target_probs[row(j-1), candidates[j]]
        accept = (coin[j-1] <= tps_j / threshold_acc) or (tps_j >= threshold_single)

    which removes the sequential dependence entirely: the whole walk becomes a
    gather plus a leading-run count.

Only the residual draw needs a kernel. After the walk, the CUDA code samples
from ``relu(target_probs - draft_probs)`` on the surviving row. ``draft_probs``
enters this call zeroed and the walk writes exactly one element of it -- the
target probability of the rejected candidate -- so the residual is the target
row with the rejected token's mass removed. The Triton kernel below takes that
token directly instead of re-reading a vocabulary-sized ``draft_probs`` row,
which is arithmetically identical and halves the memory traffic. ``draft_probs``
is still written so the tensor the caller allocated keeps the same contents as
on CUDA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_sample_kernel(
    target_probs_ptr,
    row_ptr,
    banned_ptr,
    coin_ptr,
    out_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
):
    """One program per sequence: sample from the surviving target row.

    Mirrors flashinfer's DeviceSamplingFromProb contract, including its two
    fallbacks: if no index satisfies the strict ``cumulative > u`` test the last
    index with non-zero mass is used, and if there is none the last vocabulary
    entry is used.
    """
    b = tl.program_id(0)
    row = tl.load(row_ptr + b).to(tl.int64)
    banned = tl.load(banned_ptr + b).to(tl.int64)
    base = target_probs_ptr + row * vocab_size
    num_blocks = tl.cdiv(vocab_size, BLOCK)

    total = tl.zeros((), dtype=tl.float32)
    for i in range(num_blocks):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < vocab_size
        probs = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        probs = tl.where(offs == banned, 0.0, probs)
        probs = tl.maximum(probs, 0.0)
        total += tl.sum(probs, axis=0)

    threshold = tl.load(coin_ptr + b).to(tl.float32) * total

    running = tl.zeros((), dtype=tl.float32)
    sampled = vocab_size
    last_valid = -1
    for i in range(num_blocks):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < vocab_size
        probs = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        probs = tl.where(offs == banned, 0.0, probs)
        probs = tl.maximum(probs, 0.0)

        positive = (probs > 0.0) & mask
        inclusive = tl.cumsum(probs, axis=0) + running
        hit = tl.where((inclusive > threshold) & positive, offs, vocab_size)
        first_hit = tl.min(hit, axis=0)
        sampled = tl.where(sampled == vocab_size, first_hit, sampled)

        here = tl.where(positive, offs, -1)
        last_valid = tl.maximum(last_valid, tl.max(here, axis=0))
        running += tl.sum(probs, axis=0)

    fallback = tl.where(last_valid == -1, vocab_size - 1, last_valid)
    sampled = tl.where(sampled == vocab_size, fallback, sampled)
    tl.store(out_ptr + b, sampled.to(tl.int32))


def chain_speculative_sampling_target_only(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool = True,
) -> None:
    batch, draft_len = candidates.shape
    vocab = target_probs.shape[-1]
    device = target_probs.device
    # build_speculative_verify_target_probs returns a contiguous (bs, draft_len, vocab)
    # tensor; the row arithmetic below is flat, so view it as (bs * draft_len,
    # vocab). reshape keeps a plain view for the contiguous case and still works
    # if a caller passes the already-flat layout.
    target_flat = target_probs.reshape(-1, vocab)
    draft_flat = draft_probs.reshape(-1, vocab)
    rows = torch.arange(batch, dtype=torch.int64, device=device)
    base_row = rows * draft_len

    # accept_index[b, k] == retrive_index[b, k] == b * draft_len + k for every k
    # the caller may read; the CUDA kernel leaves entries past accept_token_num
    # undefined, so filling the whole row is compatible and cheaper than masking.
    accept_index.copy_(
        (base_row[:, None] + torch.arange(draft_len, device=device)[None, :]).to(
            accept_index.dtype
        )
    )

    if draft_len > 1:
        level = torch.arange(draft_len - 1, dtype=torch.int64, device=device)
        # Level j is verified against the row of level j-1.
        probe_rows = base_row[:, None] + level[None, :]
        probe_tokens = candidates[:, 1:].to(torch.int64)
        tps = target_flat[probe_rows.reshape(-1), probe_tokens.reshape(-1)]
        tps = tps.view(batch, draft_len - 1).to(torch.float32)

        coins = uniform_samples[:, : draft_len - 1].to(torch.float32)
        accepted = (coins <= tps / threshold_acc) | (tps >= threshold_single)
        # Leading run of acceptances; the walk stops at the first rejection.
        leading = torch.cumprod(accepted.to(torch.int32), dim=1)
        num_accepted = leading.sum(dim=1)

        # predicts[b * draft_len + j - 1] = candidates[b, j] for accepted levels.
        predict_slots = (base_row[:, None] + level[None, :]).reshape(-1)
        predict_vals = probe_tokens.reshape(-1).to(predicts.dtype)
        keep = leading.reshape(-1).bool()
        predicts[predict_slots[keep]] = predict_vals[keep]
    else:
        num_accepted = torch.zeros(batch, dtype=torch.int64, device=device)

    accept_token_num.copy_(num_accepted.to(accept_token_num.dtype))

    # Surviving row and the slot the residual draw must fill.
    final_row = base_row + num_accepted
    rejected_level = num_accepted + 1
    has_rejection = num_accepted != (draft_len - 1)
    safe_level = torch.where(
        has_rejection, rejected_level, torch.zeros_like(rejected_level)
    ).clamp_(0, draft_len - 1)
    rejected_token = candidates.to(torch.int64)[rows, safe_level]
    banned = torch.where(
        has_rejection, rejected_token, torch.full_like(rejected_token, -1)
    )

    # Keep draft_probs byte-identical to the CUDA path for the caller.
    if bool(has_rejection.any()):
        hit = has_rejection.nonzero(as_tuple=True)[0]
        hit_rows = final_row[hit]
        hit_tokens = rejected_token[hit]
        draft_flat[hit_rows, hit_tokens] = target_flat[hit_rows, hit_tokens]

    sampled = torch.empty(batch, dtype=torch.int32, device=device)
    block = 8192 if vocab >= 8192 else triton.next_power_of_2(max(vocab, 1))
    _residual_sample_kernel[(batch,)](
        target_flat,
        final_row.to(torch.int32),
        banned,
        uniform_samples_for_final_sampling.to(torch.float32),
        sampled,
        vocab,
        BLOCK=block,
        num_warps=8,
    )
    predicts[final_row] = sampled.to(predicts.dtype)
