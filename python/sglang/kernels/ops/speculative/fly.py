"""FLy: entropy-gated deferred verification for speculative decoding.

Native verification stops at the first rejected draft token: for a greedy request one
that differs from the target argmax, for a sampled one a token whose draw fails the
chain rule u * q(x) < p(x). FLy (https://arxiv.org/abs/2511.22972, vLLM #53987)
accepts such a rejected draft token instead when all of these hold:

* the target top-k entropy at that position is at least ``entropy_threshold``: the
  ``entropy_top_k`` largest processed target probabilities, renormalized over
  themselves, give ``-sum(p * log p) / log(entropy_top_k)`` in [0, 1] (0 for a top-k
  of 1), as in vLLM #66593;
* the target does not mask the draft token (a finite logit, p(x) > 0) and, for a
  sampled request, the draft proposed it with q(x) > 0;
* the next ``window_size`` draft tokens pass native verification, and all of them lie
  inside the row's verified drafts.

A rejection inside that window rejects the original position, and a position without
a full window after it is never deferred. The pre-passes encode each decision in an
input the unchanged accept reads: the greedy one overwrites the target argmax with the
draft token, the sampling ones set the position's draw u to 0, which the chain rule
then accepts. The accept still takes its bonus at the first rejection that is not
deferred. FLy is lossy: it does not preserve the target distribution.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.srt.speculative.fly_config import FlyConfig


def compute_fly_entropy(logits: torch.Tensor, entropy_top_k: int) -> torch.Tensor:
    """FLy's normalized top-k entropy of each ``logits`` [N, V] row, float32 [N]."""
    top_k = min(entropy_top_k, logits.shape[-1])
    if top_k <= 1:
        return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    top_values = torch.topk(logits.float(), k=top_k, dim=-1).values
    return _normalized_entropy(torch.softmax(top_values, dim=-1))


def compute_fly_entropy_from_probs(
    probs: torch.Tensor, entropy_top_k: int
) -> torch.Tensor:
    """FLy's normalized top-k entropy of each ``probs`` [N, V] row, float32 [N]."""
    top_k = min(entropy_top_k, probs.shape[-1])
    if top_k <= 1:
        return torch.zeros(probs.shape[0], dtype=torch.float32, device=probs.device)
    top_values = torch.topk(probs.float(), k=top_k, dim=-1).values
    mass = top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return _normalized_entropy(top_values / mass)


def _normalized_entropy(top_probs: torch.Tensor) -> torch.Tensor:
    terms = torch.where(
        top_probs > 0, top_probs * top_probs.log(), torch.zeros_like(top_probs)
    )
    return -terms.sum(dim=-1) / math.log(top_probs.shape[-1])


class FlyGreedyPrepass:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(cls, **kwargs) -> torch.Tensor:
        return apply_fly_greedy(**kwargs)

    @classmethod
    def triton(cls, **kwargs) -> torch.Tensor:
        return apply_fly_greedy_triton(**kwargs)


def apply_fly_greedy(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    target_logits: torch.Tensor,
    fly: FlyConfig,
    verify_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference pre-pass. ``candidates`` [bs, T] (anchor + drafts), ``target_predict``
    [bs, T] or [bs * T] argmax ids, ``target_logits`` [bs * T, V], ``verify_lens``
    [bs] verified rows per request (None: all T). Returns ``target_predict`` with the
    deferred positions replaced by their draft tokens, in its input shape."""
    bs, num_rows = candidates.shape
    gamma = num_rows - 1
    vocab = target_logits.shape[-1]
    target = target_predict.reshape(bs, num_rows)
    drafts = candidates[:, 1:].to(torch.int64)
    native = drafts == target[:, :gamma].to(torch.int64)

    draft_rows = target_logits.reshape(bs, num_rows, vocab)[:, :gamma].float()
    valid = (drafts >= 0) & (drafts < vocab)
    draft_logit = draft_rows.gather(-1, drafts.clamp(0, vocab - 1)[..., None])[..., 0]
    entropy = compute_fly_entropy(
        draft_rows.reshape(bs * gamma, vocab), fly.entropy_top_k
    ).view(bs, gamma)
    deferrable = (
        _window_ok(native, fly.window_size, verify_lens)
        & valid
        & (draft_logit > float("-inf"))
        & (entropy >= fly.entropy_threshold)
    )

    deferred = _deferred_positions(native, deferrable)
    out = target.clone()
    out[:, :gamma] = torch.where(deferred, drafts.to(out.dtype), out[:, :gamma])
    return out.view_as(target_predict)


def _window_ok(
    native: torch.Tensor, window: int, verify_lens: Optional[torch.Tensor]
) -> torch.Tensor:
    """[bs, gamma]: the next ``window`` drafts pass natively, inside the verified ones."""
    bs, gamma = native.shape
    if verify_lens is None:
        num_drafts = torch.full((bs,), gamma, dtype=torch.int64, device=native.device)
    else:
        num_drafts = verify_lens.to(device=native.device, dtype=torch.int64) - 1
    positions = torch.arange(gamma, device=native.device)
    window_ok = (positions[None, :] + window) < num_drafts[:, None]
    padded_native = F.pad(native, (0, window), value=False)
    for offset in range(1, window + 1):
        window_ok &= padded_native[:, offset : offset + gamma]
    return window_ok


def _deferred_positions(native: torch.Tensor, deferrable: torch.Tensor) -> torch.Tensor:
    """[bs, gamma]: the rejections deferred before the walk stops."""
    deferred = torch.zeros_like(native)
    alive = torch.ones(native.shape[0], dtype=torch.bool, device=native.device)
    for pos in range(native.shape[1]):
        deferred[:, pos] = alive & ~native[:, pos] & deferrable[:, pos]
        alive &= native[:, pos] | deferred[:, pos]
    return deferred


@triton.jit
def _row_topk_entropy(
    row_ptr,
    vocab,
    scale,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One pass over a logits row times ``scale`` (1 / temperature) for its running
    top-k values, softmaxed over themselves for the normalized entropy."""
    kidx = tl.arange(0, TOP_K_PAD)
    # Pad slots hold +inf so the min-replacement below never picks them.
    top = tl.where(kidx < TOP_K, float("-inf"), float("inf"))
    voffs = tl.arange(0, BLOCK_V)
    for start in range(0, vocab, BLOCK_V):
        x = (
            tl.load(
                row_ptr + start + voffs,
                mask=start + voffs < vocab,
                other=float("-inf"),
            ).to(tl.float32)
            * scale
        )
        for _ in tl.static_range(TOP_K):
            value, index = tl.max(x, axis=0, return_indices=True)
            x = tl.where(voffs == index, float("-inf"), x)
            lowest, slot = tl.min(top, axis=0, return_indices=True)
            top = tl.where((kidx == slot) & (value > lowest), value, top)
    if TOP_K == 1:
        entropy = tl.zeros((), tl.float32)
    else:
        top = tl.where(kidx < TOP_K, top, float("-inf"))
        exps = tl.where(top == float("-inf"), 0.0, tl.exp(top - tl.max(top, axis=0)))
        probs = exps / tl.sum(exps, axis=0)
        terms = tl.where(probs > 0, probs * tl.log(probs), 0.0)
        entropy = -tl.sum(terms, axis=0) / tl.log(tl.full((), TOP_K, tl.float32))
    return entropy


@triton.jit
def _row_topk_prob_entropy(
    row_ptr,
    vocab,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One pass over a probability row for its running top-k values, renormalized
    over themselves for the normalized entropy."""
    kidx = tl.arange(0, TOP_K_PAD)
    # Probabilities lie in [0, 1]: pad slots hold 2 so the min-replacement skips them.
    top = tl.where(kidx < TOP_K, -1.0, 2.0)
    voffs = tl.arange(0, BLOCK_V)
    for start in range(0, vocab, BLOCK_V):
        x = tl.load(row_ptr + start + voffs, mask=start + voffs < vocab, other=-1.0).to(
            tl.float32
        )
        for _ in tl.static_range(TOP_K):
            value, index = tl.max(x, axis=0, return_indices=True)
            x = tl.where(voffs == index, -1.0, x)
            lowest, slot = tl.min(top, axis=0, return_indices=True)
            top = tl.where((kidx == slot) & (value > lowest), value, top)
    if TOP_K == 1:
        entropy = tl.zeros((), tl.float32)
    else:
        top = tl.where((kidx < TOP_K) & (top > 0), top, 0.0)
        probs = top / tl.maximum(tl.sum(top, axis=0), 1e-12)
        terms = tl.where(probs > 0, probs * tl.log(probs), 0.0)
        entropy = -tl.sum(terms, axis=0) / tl.log(tl.full((), TOP_K, tl.float32))
    return entropy


@triton.jit
def _fly_greedy_prepass_kernel(
    candidates_ptr,
    cand_stride,
    target_ptr,
    logits_ptr,
    logits_stride,
    verify_lens_ptr,
    vocab,
    entropy_threshold,
    NUM_ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    WINDOW: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_VERIFY_LENS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    gamma = NUM_ROWS - 1
    offs = tl.arange(0, BLOCK_T)
    in_draft = offs < gamma
    cand_base = candidates_ptr + row * cand_stride + 1
    target_base = target_ptr + row * NUM_ROWS
    if HAS_VERIFY_LENS:
        num_drafts = tl.load(verify_lens_ptr + row).to(tl.int32) - 1
    else:
        num_drafts = gamma
    drafts = tl.load(cand_base + offs, mask=in_draft, other=-1).to(tl.int64)
    targets = tl.load(target_base + offs, mask=in_draft, other=-2).to(tl.int64)
    native = in_draft & (drafts == targets)
    # Every read precedes the stores below, so the window sees native decisions.
    window_ok = (offs + WINDOW) < num_drafts
    for offset in tl.static_range(1, WINDOW + 1):
        in_future = (offs + offset) < gamma
        future_draft = tl.load(cand_base + offs + offset, mask=in_future, other=-1)
        future_target = tl.load(target_base + offs + offset, mask=in_future, other=-2)
        window_ok &= future_draft.to(tl.int64) == future_target.to(tl.int64)
    deferrable = in_draft & window_ok & (drafts >= 0) & (drafts < vocab)

    alive = tl.full((), 1, tl.int32)
    for pos in tl.static_range(NUM_ROWS - 1):
        at_pos = offs == pos
        is_native = tl.sum((at_pos & native).to(tl.int32), axis=0)
        can_try = tl.sum((at_pos & deferrable).to(tl.int32), axis=0)
        if (alive != 0) & (is_native == 0):
            deferred = tl.full((), 0, tl.int32)
            if can_try != 0:
                draft = tl.sum(tl.where(at_pos, drafts, 0), axis=0)
                row_ptr = logits_ptr + (row * NUM_ROWS + pos) * logits_stride
                draft_logit = tl.load(row_ptr + draft).to(tl.float32)
                if draft_logit > float("-inf"):
                    entropy = _row_topk_entropy(
                        row_ptr,
                        vocab,
                        1.0,
                        TOP_K=TOP_K,
                        TOP_K_PAD=TOP_K_PAD,
                        BLOCK_V=BLOCK_V,
                    )
                    if entropy >= entropy_threshold:
                        tl.store(
                            target_base + pos, draft.to(target_ptr.dtype.element_ty)
                        )
                        deferred = tl.full((), 1, tl.int32)
            alive = deferred


def apply_fly_greedy_triton(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    target_logits: torch.Tensor,
    fly: FlyConfig,
    verify_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Same contract as apply_fly_greedy."""
    bs, num_rows = candidates.shape
    assert target_predict.numel() == bs * num_rows
    assert target_logits.dim() == 2
    assert candidates.stride(1) == 1 and target_logits.stride(-1) == 1
    out = target_predict.clone(memory_format=torch.contiguous_format)
    vocab = target_logits.shape[-1]
    _fly_greedy_prepass_kernel[(bs,)](
        candidates,
        candidates.stride(0),
        out,
        target_logits,
        target_logits.stride(0),
        verify_lens if verify_lens is not None else out,
        vocab,
        float(fly.entropy_threshold),
        NUM_ROWS=num_rows,
        BLOCK_T=triton.next_power_of_2(num_rows),
        WINDOW=fly.window_size,
        TOP_K=min(fly.entropy_top_k, vocab),
        TOP_K_PAD=triton.next_power_of_2(min(fly.entropy_top_k, vocab)),
        BLOCK_V=2048,
        HAS_VERIFY_LENS=verify_lens is not None,
        num_warps=4,
    )
    return out


class FlySamplingPrepass:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(cls, **kwargs) -> torch.Tensor:
        return apply_fly_sampling(**kwargs)

    @classmethod
    def triton(cls, **kwargs) -> torch.Tensor:
        return apply_fly_sampling_triton(**kwargs)


def apply_fly_sampling(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    fly: FlyConfig,
    verify_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference pre-pass for the chain rule accept (draft token x accepted when
    u * q(x) < p(x)). ``candidates`` [bs, T], ``target_probs`` [bs, T, V] processed
    target probabilities, ``draft_probs`` [bs, T - 1, V], ``uniform_samples`` [bs, T - 1]
    the draws. Returns the draws with each deferred position's set to 0."""
    bs, num_rows = candidates.shape
    gamma = num_rows - 1
    vocab = target_probs.shape[-1]
    drafts = candidates[:, 1:].to(torch.int64)
    valid = (drafts >= 0) & (drafts < vocab)
    index = drafts.clamp(0, vocab - 1)[..., None]
    p = target_probs[:, :gamma].float().gather(-1, index)[..., 0]
    q = draft_probs[:, :gamma].float().gather(-1, index)[..., 0]
    q_is_prob = valid & (q > 0) & (q <= 1)
    native = q_is_prob & (uniform_samples[:, :gamma] * q < p)
    entropy = compute_fly_entropy_from_probs(
        target_probs[:, :gamma].reshape(bs * gamma, vocab), fly.entropy_top_k
    ).view(bs, gamma)
    deferrable = (
        _window_ok(native, fly.window_size, verify_lens)
        & q_is_prob
        & (p > 0)
        & (entropy >= fly.entropy_threshold)
    )
    deferred = _deferred_positions(native, deferrable)
    out = uniform_samples.clone()
    out[:, :gamma] = torch.where(deferred, 0.0, out[:, :gamma])
    return out


@triton.jit
def _fly_sampling_prepass_kernel(
    candidates_ptr,
    cand_stride,
    uniform_ptr,
    uniform_stride,
    target_ptr,
    target_stride_b,
    target_stride_s,
    draft_ptr,
    draft_stride_b,
    draft_stride_s,
    verify_lens_ptr,
    vocab,
    entropy_threshold,
    NUM_ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_VERIFY_LENS: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    if HAS_VERIFY_LENS:
        num_drafts = tl.load(verify_lens_ptr + b).to(tl.int32) - 1
    else:
        num_drafts = NUM_ROWS - 1
    # Bit pos: the chain rule accepts draft pos; the draft is a deferral candidate.
    native_bits = tl.zeros((), tl.int32)
    usable_bits = tl.zeros((), tl.int32)
    for pos in tl.static_range(NUM_ROWS - 1):
        token = tl.load(candidates_ptr + b * cand_stride + 1 + pos).to(tl.int64)
        valid = (token >= 0) & (token < vocab)
        safe = tl.where(valid, token, 0)
        p = tl.load(target_ptr + b * target_stride_b + pos * target_stride_s + safe)
        q = tl.load(draft_ptr + b * draft_stride_b + pos * draft_stride_s + safe)
        u = tl.load(uniform_ptr + b * uniform_stride + pos)
        q_is_prob = valid & (q > 0.0) & (q <= 1.0)
        native_bits |= (q_is_prob & (u * q < p)).to(tl.int32) << pos
        usable_bits |= (q_is_prob & (p > 0.0)).to(tl.int32) << pos

    window_mask = (1 << WINDOW) - 1
    alive = tl.full((), 1, tl.int32)
    for pos in tl.static_range(NUM_ROWS - 1):
        is_native = (native_bits >> pos) & 1
        can_try = (
            (((usable_bits >> pos) & 1) != 0)
            & (((native_bits >> (pos + 1)) & window_mask) == window_mask)
            & (pos + WINDOW < num_drafts)
        )
        if (alive != 0) & (is_native == 0):
            deferred = tl.full((), 0, tl.int32)
            if can_try:
                entropy = _row_topk_prob_entropy(
                    target_ptr + b * target_stride_b + pos * target_stride_s,
                    vocab,
                    TOP_K=TOP_K,
                    TOP_K_PAD=TOP_K_PAD,
                    BLOCK_V=BLOCK_V,
                )
                if entropy >= entropy_threshold:
                    tl.store(uniform_ptr + b * uniform_stride + pos, 0.0)
                    deferred = tl.full((), 1, tl.int32)
            alive = deferred


def apply_fly_sampling_triton(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    fly: FlyConfig,
    verify_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Same contract as apply_fly_sampling."""
    bs, num_rows = candidates.shape
    assert target_probs.dim() == 3 and draft_probs.dim() == 3
    assert target_probs.stride(-1) == 1 and draft_probs.stride(-1) == 1
    out = uniform_samples.clone(memory_format=torch.contiguous_format)
    vocab = target_probs.shape[-1]
    top_k = min(fly.entropy_top_k, vocab)
    _fly_sampling_prepass_kernel[(bs,)](
        candidates,
        candidates.stride(0),
        out,
        out.stride(0),
        target_probs,
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        verify_lens if verify_lens is not None else out,
        vocab,
        float(fly.entropy_threshold),
        NUM_ROWS=num_rows,
        WINDOW=fly.window_size,
        TOP_K=top_k,
        TOP_K_PAD=triton.next_power_of_2(top_k),
        BLOCK_V=2048,
        HAS_VERIFY_LENS=verify_lens is not None,
        num_warps=4,
    )
    return out
