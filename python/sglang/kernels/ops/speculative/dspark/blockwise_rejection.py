"""Block-wise rejection sampling of a DSpark verify chain (SGLANG_OPT_DSV41_BLOCKWISE_REJECTION).

Classic chain verification only. It replaces the draft / target softmax writes and the
single-CTA ``speculative_sampling_classic_kernel``:

1. per-vocab-block target / draft max and sum-exp (rows x 8,192-token blocks), no
   probability tensors;
2. a one-warp accept per request that reads only those stats plus the draft token's
   logits: accept iff ``log p(x) > log u + log q(x)``, i.e. ``u * q < p`` as before;
3. a block-parallel Gumbel-max resample of the rejected row's residual
   ``max(p - q, 0)`` (or of ``p`` at the bonus row), reduced by a one-program insert.

The draws are counter-based: a request row's accept coins and its resample noise are
functions of (server key, a device step counter, row, token). Every TP rank holds the
same key, counter and logits, so ranks agree without a broadcast, and a CUDA graph
replay advances the counter like an eager call. Distributions are those of the classic
kernel; only the random stream differs.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

# XORed into the server seed so the rejection draws differ from other samplers'.
_REJECTION_NOISE_DOMAIN = 0x5B1E3C7A9D2F4E61
# Odd multiplier mixing the step counter into the per-step seed.
_STEP_SEED_MULT = tl.constexpr(0x632BE59BD9B4E019)
# Smallest positive fp32 value of tl.rand; coins stay in (0, 1).
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10)
_STATS_BLOCK = 8192
_RESAMPLE_BLOCK = 1024


class RejectionNoise(msgspec.Struct, frozen=True):
    """Key of the rejection draws, and a device step counter each call advances."""

    key: int
    counter: torch.Tensor


def rejection_noise_key(random_seed: int) -> int:
    return (int(random_seed) ^ _REJECTION_NOISE_DOMAIN) & 0x7FFFFFFFFFFFFFFF


def make_rejection_noise(random_seed: int, device) -> RejectionNoise:
    return RejectionNoise(
        key=rejection_noise_key(random_seed),
        counter=torch.zeros(1, dtype=torch.int64, device=device),
    )


@triton.jit
def _step_seed(key, counter_ptr):
    return key + tl.load(counter_ptr) * _STEP_SEED_MULT


@triton.jit
def _rotl32(value, shift: tl.constexpr):
    return (value << shift) | (value >> (32 - shift))


@triton.jit
def _murmur3_mix(h, key):
    key *= 0xCC9E2D51
    key = _rotl32(key, 15)
    key *= 0x1B873593
    h ^= key
    h = _rotl32(h, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _murmur3_uniform32(seed, pos, offset):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    offset = offset.to(tl.uint32)
    h = (seed ^ seed).to(tl.uint32)
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, offset)
    h ^= 16
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    # Both 16-bit halves convert exactly: (h + 0.5) * 2**-32 keeps the u -> 0 tail.
    hi16 = (h >> 16).to(tl.int32)
    lo16 = (h & 0xFFFF).to(tl.int32)
    return (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


@triton.jit
def _log1p_neg_stable(value):
    # log(1 - value): series below 0.25 (abs error < 6e-7), direct above.
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial
    direct = tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08))
    return tl.where(value < 0.25, series, direct)


@triton.jit
def _global_lse(
    max_ptr, sumexp_ptr, row, NUM_BLOCKS: tl.constexpr, PADDED: tl.constexpr
):
    blocks = tl.arange(0, PADDED)
    mask = blocks < NUM_BLOCKS
    maxes = tl.load(max_ptr + row * NUM_BLOCKS + blocks, mask=mask, other=float("-inf"))
    sumexps = tl.load(sumexp_ptr + row * NUM_BLOCKS + blocks, mask=mask, other=0.0)
    global_max = tl.max(maxes, axis=0)
    return global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))


@triton.jit
def _stats_kernel(
    t_max_ptr,
    t_sumexp_ptr,
    d_max_ptr,
    d_sumexp_ptr,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = row // STRIDE
    step = row % STRIDE
    if step == STRIDE - 1:
        # Bonus row: its resample reads raw logits, no stats.
        return
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    t = (
        tl.load(
            target_logits_ptr + row * target_stride + offs,
            mask=offs < target_vocab,
            other=float("-inf"),
        ).to(tl.float32)
        / t_temp
    )
    t_max = tl.max(t, axis=0)
    tl.store(t_max_ptr + row * NUM_BLOCKS + block_idx, t_max)
    tl.store(
        t_sumexp_ptr + row * NUM_BLOCKS + block_idx,
        tl.where(t_max > float("-inf"), tl.sum(tl.exp(t - t_max)), 0.0),
    )
    if HAS_DRAFT_LOGITS:
        d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
        d = (
            tl.load(
                draft_logits_ptr + req * draft_stride_0 + step * draft_stride_1 + offs,
                mask=offs < draft_vocab,
                other=float("-inf"),
            ).to(tl.float32)
            / d_temp
        )
        d_max = tl.max(d, axis=0)
        tl.store(d_max_ptr + row * NUM_BLOCKS + block_idx, d_max)
        tl.store(
            d_sumexp_ptr + row * NUM_BLOCKS + block_idx,
            tl.where(d_max > float("-inf"), tl.sum(tl.exp(d - d_max)), 0.0),
        )


@triton.jit
def _rejection_kernel(
    correct_len_ptr,
    raw_len_ptr,
    trim_ptr,
    rej_lse_ptr,
    candidates_ptr,
    candidates_stride,
    verify_lens_ptr,
    t_max_ptr,
    t_sumexp_ptr,
    d_max_ptr,
    d_sumexp_ptr,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    bs,
    key,
    counter_ptr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_CUTOFF: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    seed = _step_seed(key, counter_ptr)
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    d_temp = 1.0
    if HAS_DRAFT_LOGITS:
        d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
    accepted = tl.zeros((), tl.int32)
    verifying = True
    for i in range(STRIDE - 1):
        if verifying:
            row = req * STRIDE + i
            token = tl.load(candidates_ptr + req * candidates_stride + i + 1).to(
                tl.int64
            )
            valid = (token >= 0) & (token < target_vocab)
            token = tl.minimum(tl.maximum(token, 0), target_vocab - 1)
            t_lse = _global_lse(t_max_ptr, t_sumexp_ptr, row, NUM_BLOCKS, PADDED_BLOCKS)
            log_p = (
                tl.load(target_logits_ptr + row * target_stride + token).to(tl.float32)
                / t_temp
                - t_lse
            )
            log_q = 0.0
            if HAS_DRAFT_LOGITS:
                d_lse = _global_lse(
                    d_max_ptr, d_sumexp_ptr, row, NUM_BLOCKS, PADDED_BLOCKS
                )
                log_q = (
                    tl.load(
                        draft_logits_ptr
                        + req * draft_stride_0
                        + i * draft_stride_1
                        + token,
                        mask=token < draft_vocab,
                        other=float("-inf"),
                    ).to(tl.float32)
                    / d_temp
                    - d_lse
                )
            u = tl.maximum(tl.rand(seed, (req * STRIDE + i).to(tl.int32)), _TL_RAND_MIN)
            # A proposal must have positive draft probability (classic kernel's guard).
            ok = valid & (log_q > float("-inf")) & (log_p > tl.log(u) + log_q)
            verifying = ok
            accepted += ok.to(tl.int32)
    tl.store(raw_len_ptr + req, accepted)
    capped = accepted
    if HAS_CUTOFF:
        cap = tl.load(verify_lens_ptr + req).to(tl.int32) - 1
        capped = tl.minimum(accepted, cap)
    tl.store(correct_len_ptr + req, capped)
    tl.store(trim_ptr + req, accepted - capped)
    t_lse_rej = 0.0
    d_lse_rej = 0.0
    if accepted < STRIDE - 1:
        rej_row = req * STRIDE + accepted
        t_lse_rej = _global_lse(
            t_max_ptr, t_sumexp_ptr, rej_row, NUM_BLOCKS, PADDED_BLOCKS
        )
        if HAS_DRAFT_LOGITS:
            d_lse_rej = _global_lse(
                d_max_ptr, d_sumexp_ptr, rej_row, NUM_BLOCKS, PADDED_BLOCKS
            )
    tl.store(rej_lse_ptr + req, t_lse_rej)
    tl.store(rej_lse_ptr + bs + req, d_lse_rej)


@triton.jit
def _resample_kernel(
    res_max_ptr,
    res_arg_ptr,
    raw_len_ptr,
    rej_lse_ptr,
    candidates_ptr,
    candidates_stride,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    bs,
    key,
    counter_ptr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    seed = _step_seed(key, counter_ptr)
    r = tl.load(raw_len_ptr + req).to(tl.int64)
    row = req * STRIDE + r
    is_bonus = r == STRIDE - 1
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < target_vocab
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    t = (
        tl.load(
            target_logits_ptr + row * target_stride + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / t_temp
    )
    if is_bonus:
        residual = t
    else:
        rejected = tl.load(candidates_ptr + req * candidates_stride + r + 1)
        if HAS_DRAFT_LOGITS:
            d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
            log_p = t - tl.load(rej_lse_ptr + req)
            log_q = tl.load(
                draft_logits_ptr + req * draft_stride_0 + r * draft_stride_1 + offs,
                mask=offs < draft_vocab,
                other=float("-inf"),
            ).to(tl.float32) / d_temp - tl.load(rej_lse_ptr + bs + req)
            # log max(p - q, 0) = log p + log1p(-q / p), -inf where q >= p.
            ratio = tl.exp(log_q - log_p)
            residual = tl.where(
                ratio < 1.0, log_p + libdevice.log1p(-ratio), float("-inf")
            ).to(tl.float32)
        else:
            # One-hot draft: the residual is p without the rejected token.
            residual = tl.where(offs != rejected, t, float("-inf"))
    u = _murmur3_uniform32(seed, req, offs)
    noised = tl.where(mask, residual - tl.log(-_log1p_neg_stable(u)), float("-inf"))
    value, idx = tl.max(noised, axis=0, return_indices=True)
    tl.store(res_max_ptr + req * NUM_BLOCKS + block_idx, value)
    tl.store(res_arg_ptr + req * NUM_BLOCKS + block_idx, block_idx * BLOCK + idx)


@triton.jit
def _insert_kernel(
    bonus_ptr,
    correct_len_ptr,
    raw_len_ptr,
    res_max_ptr,
    res_arg_ptr,
    candidates_ptr,
    candidates_stride,
    counter_ptr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    capped = tl.load(correct_len_ptr + req)
    raw = tl.load(raw_len_ptr + req)
    if capped < raw:
        # Capped by the verify budget: the bonus is the accepted draft after the cap.
        bonus = tl.load(candidates_ptr + req * candidates_stride + capped + 1).to(
            tl.int64
        )
    else:
        blocks = tl.arange(0, PADDED_BLOCKS)
        local_max = tl.load(
            res_max_ptr + req * NUM_BLOCKS + blocks,
            mask=blocks < NUM_BLOCKS,
            other=float("-inf"),
        )
        # NaN would make argmax land in the padded tail.
        local_max = tl.where(local_max != local_max, float("-inf"), local_max)
        best = tl.argmax(local_max, axis=0)
        bonus = tl.load(res_arg_ptr + req * NUM_BLOCKS + best)
    tl.store(bonus_ptr + req, bonus)
    if req == 0:
        # The other kernels of this call have read the counter (stream order).
        tl.store(counter_ptr, tl.load(counter_ptr) + 1)


class BlockwiseRejection:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return blockwise_rejection_triton(*args, **kwargs)
        return blockwise_rejection_torch(*args, **kwargs)


def blockwise_rejection_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    target_temperatures: torch.Tensor,
    draft_temperatures: torch.Tensor,
    noise: RejectionNoise,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``draft_logits`` is pre-temperature, or None for a one-hot draft. Returns
    (correct_len int32, bonus int64, cap_trim_lens int32), each [bs]."""
    bs, stride = candidates.shape
    target_vocab = target_logits.shape[-1]
    assert target_logits.stride(-1) == 1 and target_logits.shape[0] == bs * stride
    has_draft = draft_logits is not None
    draft_vocab = draft_logits.shape[-1] if has_draft else target_vocab
    if has_draft:
        assert draft_logits.stride(-1) == 1 and draft_logits.shape[1] >= stride - 1
    device = candidates.device
    target_temperatures = target_temperatures.reshape(-1).contiguous()
    draft_temperatures = draft_temperatures.reshape(-1).contiguous()
    num_blocks = triton.cdiv(max(target_vocab, draft_vocab), _STATS_BLOCK)
    padded_blocks = triton.next_power_of_2(num_blocks)
    stats = torch.empty(
        (4, bs * stride, num_blocks), dtype=torch.float32, device=device
    )
    lens = torch.empty((3, bs), dtype=torch.int32, device=device)
    rej_lse = torch.empty((2, bs), dtype=torch.float32, device=device)
    res_blocks = triton.cdiv(target_vocab, _RESAMPLE_BLOCK)
    res_max = torch.empty((bs, res_blocks), dtype=torch.float32, device=device)
    res_arg = torch.empty((bs, res_blocks), dtype=torch.int64, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    correct_len, raw_len, trim = lens[0], lens[1], lens[2]
    t_max, t_sumexp, d_max, d_sumexp = stats[0], stats[1], stats[2], stats[3]
    d_ptr = draft_logits if has_draft else target_logits
    d_s0 = draft_logits.stride(0) if has_draft else 0
    d_s1 = draft_logits.stride(1) if has_draft else 0
    _stats_kernel[(bs * stride, num_blocks)](
        t_max, t_sumexp, d_max, d_sumexp,
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab,
        STRIDE=stride, NUM_BLOCKS=num_blocks, BLOCK=_STATS_BLOCK,
        HAS_DRAFT_LOGITS=has_draft,
    )  # fmt: skip
    has_cutoff = cutoff_verify_lens is not None
    verify_lens = (
        cutoff_verify_lens.to(device=device).contiguous() if has_cutoff else correct_len
    )
    _rejection_kernel[(bs,)](
        correct_len, raw_len, trim, rej_lse,
        candidates, candidates.stride(0), verify_lens,
        t_max, t_sumexp, d_max, d_sumexp,
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        noise.key, noise.counter,
        STRIDE=stride, NUM_BLOCKS=num_blocks, PADDED_BLOCKS=padded_blocks,
        HAS_DRAFT_LOGITS=has_draft, HAS_CUTOFF=has_cutoff,
        num_warps=1,
    )  # fmt: skip
    _resample_kernel[(bs, res_blocks)](
        res_max, res_arg, raw_len, rej_lse,
        candidates, candidates.stride(0),
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        noise.key, noise.counter,
        STRIDE=stride, NUM_BLOCKS=res_blocks, BLOCK=_RESAMPLE_BLOCK,
        HAS_DRAFT_LOGITS=has_draft,
    )  # fmt: skip
    _insert_kernel[(bs,)](
        bonus, correct_len, raw_len, res_max, res_arg,
        candidates, candidates.stride(0), noise.counter,
        NUM_BLOCKS=res_blocks, PADDED_BLOCKS=triton.next_power_of_2(res_blocks),
        num_warps=1,
    )  # fmt: skip
    return correct_len, bonus, trim


def blockwise_rejection_torch(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    target_temperatures: torch.Tensor,
    draft_temperatures: torch.Tensor,
    noise: RejectionNoise,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference with the same math; draws from a generator seeded by (key, counter)."""
    bs, stride = candidates.shape
    gamma = stride - 1
    vocab = target_logits.shape[-1]
    gen = torch.Generator(device=candidates.device)
    gen.manual_seed((noise.key + int(noise.counter.item()) * 1_000_003) % (2**63))
    t_temp = target_temperatures.reshape(bs, 1, 1).float()
    log_p = torch.log_softmax(
        target_logits.float().view(bs, stride, vocab) / t_temp, -1
    )
    if draft_logits is not None:
        d_temp = draft_temperatures.reshape(bs, 1, 1).float()
        d = draft_logits[:bs, :gamma].float() / d_temp
        log_q = torch.full((bs, gamma, vocab), float("-inf"), device=d.device)
        log_q[..., : d.shape[-1]] = torch.log_softmax(d, -1)[..., :vocab]
    else:
        log_q = torch.full((bs, gamma, vocab), float("-inf"), device=log_p.device)
        log_q.scatter_(-1, candidates[:, 1:].unsqueeze(-1), 0.0)
    coins = torch.rand((bs, gamma), generator=gen, device=candidates.device)
    gumbel = -torch.log(
        -torch.log(torch.rand((bs, vocab), generator=gen, device=candidates.device))
    )
    tokens = candidates[:, 1:].long()
    lp = log_p[:, :gamma].gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    lq = log_q.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    ok = (lq > float("-inf")) & (lp > torch.log(coins.clamp_min(4.6566127342e-10)) + lq)
    raw = torch.cumprod(ok.int(), dim=-1).sum(-1).int()
    rows = torch.arange(bs, device=candidates.device)
    lp_rej = log_p[rows, raw.long()]
    lq_rej = torch.cat([log_q, log_q[:, :1]], 1)[rows, raw.long()]
    residual = torch.where(
        raw.unsqueeze(-1) == gamma,
        lp_rej,
        torch.where(
            lq_rej < lp_rej,
            lp_rej + torch.log1p(-torch.exp(lq_rej - lp_rej)),
            float("-inf"),
        ),
    )
    bonus = (residual + gumbel).argmax(-1)
    if cutoff_verify_lens is not None:
        capped = torch.minimum(raw, cutoff_verify_lens.to(raw.device).int() - 1)
    else:
        capped = raw
    capped_bonus = candidates.gather(
        1, (capped.long() + 1).clamp_max(gamma).unsqueeze(-1)
    ).squeeze(-1)
    bonus = torch.where(capped < raw, capped_bonus, bonus)
    noise.counter.add_(1)
    return capped, bonus.long(), raw - capped
