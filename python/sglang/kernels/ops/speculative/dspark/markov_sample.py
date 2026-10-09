"""One Markov draft step: add the step bias to the base logits and pick one token
per row, the argmax at temperature 0 and a Gumbel-max sample otherwise.

The vocab may be TP-sharded: each rank reduces its shard to one (value,
global-index-bits) pair per block, and only those pairs are all-gathered.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 4096
_NO_INDEX = 2147483647
# XORed into the seed so this stream never reuses another consumer's Philox key.
_NOISE_DOMAIN = 0x2D2A9C317E44B1F3
_PHILOX_ROUNDS = 10
_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def draft_noise_key(seed: int) -> int:
    """The Philox key of the draft noise for a server's random seed."""
    return (int(seed) ^ _NOISE_DOMAIN) & 0x7FFFFFFFFFFFFFFF


@triton.jit
def _round_to_step_dtype(value, ROUND: tl.constexpr):
    if ROUND == 1:
        value = value.to(tl.bfloat16).to(tl.float32)
    if ROUND == 2:
        value = value.to(tl.float16).to(tl.float32)
    return value


@triton.jit
def _markov_sample_partial_kernel(
    B,
    X,
    T,
    OFFSET,
    L,
    P,
    BS,
    XS,
    LS,
    WIDTH,
    PADDED,
    VOCAB_START,
    PARTS,
    KEY,
    STEP,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    STORE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    in_shard = i < PADDED
    bias = tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    base = tl.load(X + row * XS + i, in_shard, 0).to(tl.float32)
    value = _round_to_step_dtype(base + bias, ROUND)
    if STORE:
        tl.store(L + row * LS + i, value.to(L.dtype.element_ty), in_shard)
    key = value
    temperature = tl.load(T + row)
    if temperature != 0.0:
        # One Philox call yields four words; word k goes to token 4 * g + k.
        quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
        lane = tl.zeros([BLOCK // 4], tl.int64)
        w0, w1, w2, w3 = tl.philox(
            KEY,
            quad.to(tl.uint32),
            (lane + row).to(tl.uint32),
            (lane + STEP).to(tl.uint32),
            ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
            PHILOX_ROUNDS,
        )
        bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
        # Uniform in (0, 1), dense near 0 so the winning tail of the noise keeps
        # its resolution; exp_noise ~ Exp(1), and value/T - log(exp_noise) is
        # value/T plus Gumbel noise.
        uniform = (bits.to(tl.float32) + 0.5) * 2.3283064365386963e-10
        exp_noise = -libdevice.log1p(-uniform)
        key = libdevice.div_rn(value, temperature) - libdevice.log(exp_noise)
    nan = valid & (key != key)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, key, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (key == maximum))
    idx = tl.min(tl.where(wins, i + VOCAB_START, 2147483647), 0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(P + (row * PARTS + part) * 2, maximum)
    tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


@triton.jit
def _markov_sample_finish_kernel(
    P,
    OUT,
    BS: tl.constexpr,
    PARTS: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < PARTS * WORLD
    rank, part = i // PARTS, i % PARTS
    offset = ((rank * BS + row) * PARTS + part) * 2
    value = tl.load(P + offset, valid, -float("inf"))
    idx = tl.load(P + offset + 1, valid, 0).to(tl.int32, bitcast=True)
    # The NVLink push transport changes +0 to -0 as its Lamport sentinel.
    # Indices are nonnegative int32 bits, so strip that sign bit to recover ID 0.
    idx &= 2147483647
    valid &= idx != 2147483647
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (value == maximum))
    result = tl.min(tl.where(wins, idx, 2147483647), 0)
    tl.store(OUT + row, result.to(tl.int64))


def _mul32(a: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    # (high, low) 32-bit words of a * b for a < 2**32 held in int64, without
    # overflowing int64: split b into 16-bit halves.
    product_lo = a * (b & 0xFFFF)
    product_hi = a * (b >> 16)
    high = (product_hi + (product_lo >> 16)) >> 16
    low = (((product_hi & 0xFFFF) << 16) + product_lo) & 0xFFFFFFFF
    return high, low


def _philox(key: int, c0, c1, c2, c3) -> tuple[torch.Tensor, ...]:
    """Philox4x32-10, the same function as ``tl.philox``."""
    k0, k1 = key & 0xFFFFFFFF, (key >> 32) & 0xFFFFFFFF
    for _ in range(_PHILOX_ROUNDS):
        hi_b, lo_b = _mul32(c2, 0xCD9E8D57)
        hi_a, lo_a = _mul32(c0, 0xD2511F53)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def _reduce_partials(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Lowest index among the NaNs if any, else among the maxima; _NO_INDEX
    # marks an empty candidate.
    valid = indices != _NO_INDEX
    nan = valid & values.isnan()
    has_nan = nan.any(dim=-1, keepdim=True)
    finite = torch.where(valid & ~nan, values, -torch.inf)
    maximum = finite.amax(dim=-1, keepdim=True)
    wins = torch.where(has_nan, nan, valid & (finite == maximum))
    return torch.where(wins, indices, _NO_INDEX).amin(dim=-1)


class MarkovSampleStep:
    """``base_logits`` is the rank's padded vocab shard, ``bias`` its first
    ``width`` columns (the rest is padding and never wins). The step logits are
    their sum rounded to ``step_dtype``, the dtype the head's own add produces.
    A temperature of 0 takes the argmax. ``logits_out``, when given, receives
    the step logits of the whole padded shard. Returns global token ids."""

    @classmethod
    def execute(cls, **kwargs) -> torch.Tensor:
        if kwargs["base_logits"].is_cuda:
            return cls.triton(**kwargs)
        return cls.torch(**kwargs)

    @staticmethod
    def _check(
        base_logits, bias, temperatures, offset, step_dtype, vocab_start, logits_out
    ):
        assert base_logits.ndim == bias.ndim == 2
        assert bias.shape[0] == base_logits.shape[0]
        assert bias.shape[1] <= base_logits.shape[1]
        assert bias.stride(1) == base_logits.stride(1) == 1
        assert temperatures.shape == (base_logits.shape[0],)
        assert temperatures.dtype == torch.float32
        assert offset.dtype == torch.int64 and offset.numel() == 1
        assert step_dtype in _ROUND_CODE, step_dtype
        # Philox words are dealt to tokens in groups of four from a global id.
        assert vocab_start % 4 == 0
        if logits_out is not None:
            assert logits_out.shape == base_logits.shape
            assert logits_out.dtype == step_dtype and logits_out.stride(1) == 1

    @classmethod
    def triton(
        cls,
        *,
        base_logits: torch.Tensor,
        bias: torch.Tensor,
        temperatures: torch.Tensor,
        key: int,
        offset: torch.Tensor,
        step: int,
        step_dtype: torch.dtype,
        vocab_start: int = 0,
        group=None,
        gather=None,
        logits_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cls._check(
            base_logits, bias, temperatures, offset, step_dtype, vocab_start, logits_out
        )
        rows, width = bias.shape
        padded = base_logits.shape[1]
        parts = triton.cdiv(padded, _BLOCK)
        assert parts > 0
        partial = torch.empty(
            (rows, parts, 2), device=base_logits.device, dtype=torch.float32
        )
        _markov_sample_partial_kernel[(rows, parts)](
            bias,
            base_logits,
            temperatures,
            offset,
            base_logits if logits_out is None else logits_out,
            partial,
            bias.stride(0),
            base_logits.stride(0),
            0 if logits_out is None else logits_out.stride(0),
            width,
            padded,
            vocab_start,
            parts,
            key,
            step,
            _BLOCK,
            _ROUND_CODE[step_dtype],
            logits_out is not None,
            _PHILOX_ROUNDS,
            num_warps=4,
        )
        world = 1 if group is None else group.world_size
        # The padded shard width fixes the transport shape on all ranks;
        # WIDTH masks real entries, including a completely empty final shard.
        if gather is not None:
            gathered = gather(partial.view(rows, parts * 2))
        elif world > 1:
            gathered = group.all_gather(partial, dim=0)
        else:
            gathered = partial
        result = torch.empty(rows, device=base_logits.device, dtype=torch.int64)
        _markov_sample_finish_kernel[(rows,)](
            gathered,
            result,
            rows,
            parts,
            world,
            triton.next_power_of_2(parts * world),
            num_warps=4,
        )
        return result

    @classmethod
    def torch(
        cls,
        *,
        base_logits: torch.Tensor,
        bias: torch.Tensor,
        temperatures: torch.Tensor,
        key: int,
        offset: torch.Tensor,
        step: int,
        step_dtype: torch.dtype,
        vocab_start: int = 0,
        group=None,
        gather=None,
        logits_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del gather
        cls._check(
            base_logits, bias, temperatures, offset, step_dtype, vocab_start, logits_out
        )
        rows, width = bias.shape
        padded = base_logits.shape[1]
        device = base_logits.device
        value = base_logits.to(torch.float32, copy=True)
        value[:, :width] += bias.float()
        value = value.to(step_dtype).float()
        if logits_out is not None:
            logits_out.copy_(value)

        cols = torch.arange(padded, device=device, dtype=torch.int64)
        keys = value
        sampled = temperatures != 0
        if bool(sampled.any()):
            tokens = (cols + vocab_start)[None, :]
            words = _philox(
                key,
                tokens // 4,
                torch.arange(rows, device=device, dtype=torch.int64)[:, None],
                torch.full((1, 1), step, device=device, dtype=torch.int64),
                offset.reshape(1, 1) & 0xFFFFFFFF,
            )
            bits = torch.stack(torch.broadcast_tensors(*words), dim=-1)
            bits = bits.gather(-1, (tokens % 4).expand(rows, -1)[..., None])[..., 0]
            uniform = (bits.float() + 0.5) * 2.3283064365386963e-10
            exp_noise = -torch.log1p(-uniform)
            safe_temperatures = torch.where(sampled, temperatures, 1.0)[:, None]
            noised = value / safe_temperatures - torch.log(exp_noise)
            keys = torch.where(sampled[:, None], noised, value)
        keys = torch.where(cols[None, :] < width, keys, -torch.inf)

        if width == 0:
            local_values = torch.full((rows,), -torch.inf, device=device)
            local_indices = torch.full((rows,), _NO_INDEX, device=device)
        else:
            local = keys.argmax(dim=1)
            local_values = keys.gather(1, local[:, None]).squeeze(1)
            local_indices = local + vocab_start
        pairs = torch.stack([local_values.double(), local_indices.double()], dim=1)
        if group is not None and group.world_size > 1:
            pairs = group.all_gather(pairs[None], dim=0)
        else:
            pairs = pairs[None]
        values = pairs[..., 0].transpose(0, 1)
        indices = pairs[..., 1].transpose(0, 1).long()
        return _reduce_partials(values, indices)
