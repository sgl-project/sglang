"""Fused ROCm verifier for DFLASH non-greedy (temperature > 0) speculative decoding.

The stock ROCm path costs far more than the greedy path even though it does
almost no extra arithmetic. Measured on Qwen3.5-397B-A17B-FP8 (vocab 248320,
draft_len 8, top_k 20, top_p 0.95), per decode step:

    greedy  t=0    621 CPU ops,  5.12 ms CPU,  GPU idle  8.5%
    sampling t=0.7 965 CPU ops, 38.61 ms CPU,  GPU idle 24.1%

The op *count* only grows 1.55x while the op *time* grows 7.5x, because three
of the added ops block on the device. The extra GPU compute is under 0.2 ms.

Two things cause that gap, and this module removes both.

1.  Density. `build_speculative_verify_target_probs` takes the sparse top-k result
    -- k=20 values per row -- and scatters it into a dense (bs*draft_len,
    vocab) tensor, then `chain_speculative_sampling_target_only` allocates a
    second dense tensor of the same size for `draft_probs`. That is two 7.9 MB
    zero-fills plus a scatter per step to carry 160 useful numbers, and it
    forces the residual sampler to scan 248320 entries per row with a single
    workgroup.

2.  Host stalls. The chain verifier is ~30 separate torch ops, three of which
    (`predicts[mask] = ...`, `if bool(has_rejection.any())`, `.nonzero()`)
    require the mask on the host and therefore drain the stream. That collapses
    overlap scheduling and leaves the GPU idle until the CPU catches up.

The observation that makes fusing easy: the caller only ever consumes two
values, `accept_len` and `bonus`. Everything else -- `predicts`, `accept_index`,
`draft_probs` -- is scaffolding the CUDA operator's signature demands. And the
accept test needs a *single* probability per level (that of the proposed token),
not a distribution. So the verify runs on the k-sized top-k output and never
materialises anything vocab-shaped.

Probabilities are produced by the *same torch calls the stock path makes*,
applied to the k-sized top-k result rather than to a dense vocab-sized tensor:
the same temperature division, the same `torch.topk`, the same rank mask, the
same `F.softmax`, the same `top_p_renorm_prob`. They are therefore bit-identical
to stock by construction rather than by a rounding argument. Two earlier
versions of this file re-derived those steps and diverged from stock by one
float32 ULP in two different places -- once by selecting the top-k on unscaled
logits, once by hand-rolling the softmax -- and in both cases a single ULP was
enough to flip `coin <= tps` and change the emitted token. Anything that decides
an acceptance must be computed the way stock computes it.

What the Triton kernel does, and all it does, is the part that has no torch
equivalent without host syncs: the chain accept walk and the residual draw.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# The residual walk is O(K^2) in registers, one program per sequence. 64 keeps
# the comparison tile at 4096 elements; larger top_k falls back to stock.
MAX_FUSED_TOP_K = 64
MAX_FUSED_DRAFT_LEN = 32

_warned = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


@triton.jit
def _fused_verify_kernel(
    probs_ptr,  # (bs*L, K) float32, renormalised, descending by logit
    probs_acc_ptr,  # (bs*L, K) float32, probs / threshold_acc (aliased if 1.0)
    idx_ptr,  # (bs*L, K) int64 vocab indices
    cand_ptr,  # (bs, L)   int64 proposed tokens
    unif_ptr,  # (bs, L)   float32 acceptance coins
    unif_final_ptr,  # (bs,)     float32 residual-draw coin
    accept_out,  # (bs,)     int32
    bonus_out,  # (bs,)     int64
    K,
    L,
    VOCAB,
    THRESH_SINGLE,
    BLOCK_K: tl.constexpr,
    SAME_ACC: tl.constexpr,
):
    b = tl.program_id(0)
    base = b * L
    ar = tl.arange(0, BLOCK_K)
    inb = ar < K

    # --- chain accept walk ---------------------------------------------------
    # Level j is verified against row j-1. prob_acc resets on every acceptance,
    # so on a chain the test is local to the level and the walk is the leading
    # run of acceptances.
    num_acc = 0
    still = True
    for l in range(0, L - 1):
        off = (base + l) * K + ar
        p = tl.load(probs_ptr + off, mask=inb, other=0.0).to(tl.float32)
        idx = tl.load(idx_ptr + off, mask=inb, other=-1).to(tl.int64)
        cand = tl.load(cand_ptr + base + l + 1).to(tl.int64)
        # Tokens outside the top-k support have probability 0, which is the
        # same value the dense path would have read.
        tps = tl.sum(tl.where(idx == cand, p, 0.0), axis=0)
        # `tps / threshold_acc` is done in torch, not here: a Triton divide and
        # a PyTorch divide can round differently, and `coin <= .` is a discrete
        # decision, so one ULP flips an acceptance. With threshold_acc == 1.0
        # stock divides by one, which is exact, and the second tensor is an
        # alias -- the branch below is compiled away.
        if SAME_ACC:
            tps_acc = tps
        else:
            pa = tl.load(probs_acc_ptr + off, mask=inb, other=0.0).to(tl.float32)
            tps_acc = tl.sum(tl.where(idx == cand, pa, 0.0), axis=0)
        coin = tl.load(unif_ptr + base + l).to(tl.float32)
        acc = (coin <= tps_acc) | (tps >= THRESH_SINGLE)
        still = still & acc
        num_acc += tl.where(still, 1, 0)

    # --- residual draw on the surviving row ----------------------------------
    final_row = base + num_acc
    off = final_row * K + ar
    p = tl.load(probs_ptr + off, mask=inb, other=0.0).to(tl.float32)
    idx = tl.load(idx_ptr + off, mask=inb, other=-1).to(tl.int64)

    has_rej = num_acc != (L - 1)
    safe = tl.where(has_rej, num_acc + 1, 0)
    rej = tl.load(cand_ptr + base + safe).to(tl.int64)
    banned = tl.where(has_rej, rej, -1)
    # draft_probs holds exactly the rejected token's target mass, so the
    # residual relu(target - draft) is the row with that token removed.
    p = tl.where(idx == banned, 0.0, p)

    total = tl.sum(p, axis=0)
    thr = tl.load(unif_final_ptr + b).to(tl.float32) * total

    pos = p > 0.0
    # The dense sampler walks the vocabulary in index order; zeros do not move
    # the running sum, so restricting the walk to the support is equivalent.
    prefix = tl.sum(
        tl.where((idx[None, :] < idx[:, None]) & pos[None, :], p[None, :], 0.0), axis=1
    )
    incl = prefix + p
    hit = (incl > thr) & pos

    # Sentinels built from idx so they inherit its int64 type without needing
    # a constexpr fill value.
    sent = idx * 0 + VOCAB
    none_ = idx * 0 - 1
    sampled = tl.min(tl.where(hit, idx, sent), axis=0)
    # Same two fallbacks as the dense sampler: the last index carrying mass,
    # then the last vocabulary entry.
    last_valid = tl.max(tl.where(pos, idx, none_), axis=0)
    fallback = tl.where(last_valid == -1, VOCAB - 1, last_valid)
    out = tl.where(sampled == VOCAB, fallback, sampled)

    tl.store(accept_out + b, num_acc)
    tl.store(bonus_out + b, out)


def can_use_fused(
    *,
    sampling_info: Any,
    max_top_k: Optional[int],
    vocab_size: int,
    draft_token_num: int,
    use_sparse_topk: bool,
) -> bool:
    """Whether the fused path covers this batch's sampling configuration."""
    if not use_sparse_topk:
        return False
    if not bool(getattr(sampling_info, "need_top_k_sampling", True)):
        # Without top-k the support is the whole vocabulary and there is no
        # sparse representation to exploit.
        _warn_once("no_topk", "DFLASH fused verify: top-k disabled, using stock path.")
        return False
    if max_top_k is None:
        return False
    k = int(max_top_k)
    if not 0 < k < vocab_size:
        return False
    if k > MAX_FUSED_TOP_K:
        _warn_once(
            "big_topk",
            f"DFLASH fused verify: top_k={k} exceeds {MAX_FUSED_TOP_K}, using stock path.",
        )
        return False
    if not 0 < draft_token_num <= MAX_FUSED_DRAFT_LEN:
        return False
    # One line per process, so a server log states unambiguously which verify
    # path ran instead of leaving it to be inferred from throughput.
    _warn_once(
        "active",
        f"DFLASH fused verify ACTIVE (top_k={k}, draft_len={draft_token_num}, "
        f"vocab={vocab_size}).",
    )
    return True


def _sparse_target_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_len: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The stock probability pipeline, kept sparse.

    Every step here is the call `build_speculative_verify_target_probs` makes, in
    the same order, with the same operators -- only the dense scatter at the
    end is dropped. That is what makes the result bit-identical to stock
    instead of merely close to it.

    Returns (probs, indices), both (bs*draft_len, k).
    """
    from sglang.srt.speculative.dflash_utils import top_p_renorm_prob

    device = next_token_logits.device
    vocab = int(next_token_logits.shape[-1])

    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_len, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    # The top-k must be selected on the *scaled* logits. Temperature scaling is
    # order-preserving in real arithmetic but not injective in float32: it can
    # round two adjacent values onto the same float, after which topk's
    # tie-break keeps a different token and the support set differs.
    topk_logits, topk_indices = torch.topk(scaled_logits, k=k, dim=-1, sorted=True)

    repeated_top_ks = torch.repeat_interleave(
        sampling_info.top_ks, draft_len, dim=0
    ).to(dtype=torch.int64)
    repeated_top_ks.clamp_(min=1, max=vocab)
    ranks = torch.arange(k, device=device, dtype=torch.int64)[None, :]
    valid = ranks < repeated_top_ks.unsqueeze(1)
    topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

    topk_probs = F.softmax(topk_logits, dim=-1)
    if bool(getattr(sampling_info, "need_top_p_sampling", False)):
        # Also validates that top_ps has one value per row, exactly as stock.
        repeated_top_ps = torch.repeat_interleave(
            sampling_info.top_ps, draft_len, dim=0
        )
        topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

    return topk_probs.contiguous(), topk_indices.contiguous()


def fused_dflash_sampling_verify(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    threshold_single: float,
    threshold_acc: float,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accept lengths and bonus tokens, without any vocab-sized intermediate.

    Returns (accept_len int32 [bs], bonus int64 [bs]), matching
    `compute_dflash_sampling_correct_drafts_and_bonus`.
    """
    bs, draft_len = candidates.shape
    vocab = int(next_token_logits.shape[-1])
    device = next_token_logits.device
    k = max(1, min(int(max_top_k), vocab))

    # The kernel indexes the per-request tensors by program id without bounds
    # checking. The stock path fails loudly on a size mismatch (its
    # repeat_interleave broadcast); make that failure explicit here rather than
    # reading out of bounds. top_ps is checked inside top_p_renorm_prob.
    n_temp = sampling_info.temperatures.numel()
    n_topk = sampling_info.top_ks.numel()
    if n_temp != bs or n_topk != bs:
        raise ValueError(
            "sampling_info does not match the batch: expected "
            f"{bs} rows, got temperatures={n_temp}, top_ks={n_topk}."
        )

    probs, indices = _sparse_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_len=draft_len,
        k=k,
    )

    # The acceptance threshold is divided in torch so the comparison operand is
    # bit-identical to stock's. Only the elements the walk actually reads
    # matter, but an elementwise divide on (bs*L, K) is cheaper than gathering
    # them first, and at threshold_acc == 1.0 it is skipped entirely.
    same_acc = float(threshold_acc) == 1.0
    probs_acc = probs if same_acc else torch.div(probs, float(threshold_acc))

    accept_len = torch.empty(bs, dtype=torch.int32, device=device)
    bonus = torch.empty(bs, dtype=torch.int64, device=device)

    _fused_verify_kernel[(bs,)](
        probs,
        probs_acc,
        indices,
        candidates.to(torch.int64).contiguous(),
        uniform_samples.contiguous(),
        uniform_samples_for_final_sampling.contiguous(),
        accept_len,
        bonus,
        k,
        draft_len,
        vocab,
        float(threshold_single),
        BLOCK_K=triton.next_power_of_2(k),
        SAME_ACC=same_acc,
        num_warps=4,
    )
    return accept_len, bonus
