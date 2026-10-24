"""One Markov draft step: add the step bias and pick one token per row, the
argmax at temperature 0 and a Gumbel-max sample otherwise."""

import functools
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.srt.utils import is_npu

if is_npu():
    # Ascend does not map the generic libdevice stubs to CANN implementations.
    from triton.language.extra.cann import libdevice
else:
    from triton.language.extra import libdevice

# Measured fastest for 1-64 rows, where the per-block Philox work sets latency.
_BLOCK = 1024
_NO_INDEX = tl.constexpr(2147483647)
# 2**-32: maps a 32-bit Philox word to a uniform in (0, 1].
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_PHILOX_ROUNDS = 10
_ROUND_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


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
    HAS_BIAS: tl.constexpr,
    STORE: tl.constexpr,
    SELECT: tl.constexpr,
    SAMPLE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    in_shard = i < PADDED
    value = tl.load(X + row * XS + i, in_shard, 0).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    value = _round_to_step_dtype(value, ROUND)
    if STORE:
        tl.store(L + row * LS + i, value.to(L.dtype.element_ty), in_shard)
    if SELECT:
        _select_block(
            value,
            T,
            OFFSET,
            P,
            row,
            part,
            i,
            valid,
            PARTS,
            KEY,
            STEP,
            VOCAB_START,
            BLOCK,
            SAMPLE,
            PHILOX_ROUNDS,
        )


@triton.jit
def _select_block(
    value,
    T,
    OFFSET,
    P,
    row,
    part,
    i,
    valid,
    PARTS,
    KEY,
    STEP,
    VOCAB_START,
    BLOCK: tl.constexpr,
    SAMPLE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    key = value
    if SAMPLE:
        temperature = tl.load(T + row)
    else:
        temperature = 0.0
    if temperature != 0.0:
        # One Philox call yields four words; word k goes to token 4 * q + k.
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
        # Small words convert to FP32 exactly, so the winning small noise stays precise.
        uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
        exp_noise = -libdevice.log1p(-uniform)
        key = value / temperature - tl.log(exp_noise)
    nan = valid & (key != key)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, key, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (key == maximum))
    idx = tl.min(tl.where(wins, i + VOCAB_START, _NO_INDEX), 0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(P + (row * PARTS + part) * 2, maximum)
    tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


@triton.jit
def _markov_sample_finish_kernel(
    P,
    OUT,
    BS,
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
    idx &= _NO_INDEX
    valid &= idx != _NO_INDEX
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (value == maximum))
    result = tl.min(tl.where(wins, idx, _NO_INDEX), 0)
    tl.store(OUT + row, result.to(tl.int64))


def _mul32(a: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    # (high, low) 32-bit words of a * b; b splits in halves so int64 never overflows.
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


def _launch_partial(
    base_logits,
    bias,
    *,
    width,
    store,
    temperatures,
    key,
    offset,
    step,
    step_dtype,
    has_sampled_rows,
    vocab_start=0,
    select=True,
):
    # Returns the [rows, parts * 2] (value, index bits) candidates of each block.
    rows, padded = base_logits.shape
    # The padded width fixes the transport shape on every rank; WIDTH masks padding.
    parts = triton.cdiv(padded, _BLOCK)
    assert parts > 0
    partial = torch.empty(
        (rows, parts, 2), device=base_logits.device, dtype=torch.float32
    )
    _markov_sample_partial_kernel[(rows, parts)](
        base_logits if bias is None else bias,
        base_logits,
        temperatures,
        offset,
        base_logits if store is None else store,
        partial,
        0 if bias is None else bias.stride(0),
        base_logits.stride(0),
        0 if store is None else store.stride(0),
        width,
        padded,
        vocab_start,
        parts,
        key,
        step,
        _BLOCK,
        _ROUND_CODE[step_dtype],
        bias is not None,
        store is not None,
        select,
        has_sampled_rows is not False,
        _PHILOX_ROUNDS,
        num_warps=4,
    )
    return partial.view(rows, parts * 2)


class MarkovSampleStep:
    """Returns global token ids. ``bias`` covers the first ``width`` columns of
    the rank's padded shard ``base_logits``. Sharded, ``gather`` exchanges
    per-block candidates, or ``gather_logits`` the kept ``logits_out`` whole.
    ``has_sampled_rows=False`` argmaxes every row."""

    @classmethod
    def execute(cls, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(**kwargs):
            return cls.triton(**kwargs)
        return cls.torch(**kwargs)

    @staticmethod
    def _check(
        base_logits,
        bias,
        temperatures,
        offset,
        step_dtype,
        vocab_start,
        gather,
        gather_logits,
        logits_out,
    ):
        assert base_logits.ndim == bias.ndim == 2
        assert bias.shape[0] == base_logits.shape[0]
        assert bias.shape[1] <= base_logits.shape[1]
        assert bias.stride(1) == base_logits.stride(1) == 1
        assert temperatures.shape == (base_logits.shape[0],)
        assert temperatures.dtype == torch.float32 and temperatures.stride(0) == 1
        assert offset.dtype == torch.int64 and offset.numel() == 1
        assert step_dtype in _ROUND_CODE, step_dtype
        # Philox words go to tokens in quads of global ids.
        assert vocab_start % 4 == 0, vocab_start
        if logits_out is not None:
            assert logits_out.dtype == step_dtype and logits_out.stride(1) == 1
            if gather is None:
                assert logits_out.shape == base_logits.shape
            else:
                assert gather_logits is not None
                assert logits_out.shape[0] == base_logits.shape[0]

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
        gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        gather_logits: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        logits_out: Optional[torch.Tensor] = None,
        has_sampled_rows: Optional[bool] = None,
    ) -> torch.Tensor:
        cls._check(
            base_logits,
            bias,
            temperatures,
            offset,
            step_dtype,
            vocab_start,
            gather,
            gather_logits,
            logits_out,
        )
        rows, width = bias.shape
        launch = functools.partial(
            _launch_partial,
            temperatures=temperatures,
            key=key,
            offset=offset,
            step=step,
            step_dtype=step_dtype,
            has_sampled_rows=has_sampled_rows,
        )
        if gather is not None and logits_out is not None:
            local = torch.empty(
                base_logits.shape, device=base_logits.device, dtype=step_dtype
            )
            launch(base_logits, bias, width=width, store=local, select=False)
            full = gather_logits(local)[:, : logits_out.shape[1]]
            gathered = launch(full, None, width=full.shape[1], store=logits_out)
        else:
            partial = launch(
                base_logits,
                bias,
                width=width,
                store=logits_out,
                vocab_start=vocab_start,
            )
            gathered = partial if gather is None else gather(partial)
        parts = gathered.shape[1] // 2
        world = gathered.shape[0] // rows
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
        gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        gather_logits: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        logits_out: Optional[torch.Tensor] = None,
        has_sampled_rows: Optional[bool] = None,
    ) -> torch.Tensor:
        # None draws the noise for every row: reading the temperatures on the
        # host would break a graph capture.
        cls._check(
            base_logits,
            bias,
            temperatures,
            offset,
            step_dtype,
            vocab_start,
            gather,
            gather_logits,
            logits_out,
        )
        rows, width = bias.shape
        value = base_logits.to(torch.float32, copy=True)
        value[:, :width] += bias.float()
        value = value.to(step_dtype).float()
        if gather is not None and logits_out is not None:
            full = gather_logits(value.to(step_dtype))[:, : logits_out.shape[1]]
            return cls.torch(
                base_logits=full,
                bias=torch.zeros_like(full),
                temperatures=temperatures,
                key=key,
                offset=offset,
                step=step,
                step_dtype=step_dtype,
                logits_out=logits_out,
                has_sampled_rows=has_sampled_rows,
            )
        padded = base_logits.shape[1]
        device = base_logits.device
        if logits_out is not None:
            logits_out.copy_(value)

        keys = value
        if has_sampled_rows is not False:
            quads = vocab_start // 4 + torch.arange(
                triton.cdiv(padded, 4), device=device, dtype=torch.int64
            )
            words = _philox(
                key,
                quads[None, :],
                torch.arange(rows, device=device, dtype=torch.int64)[:, None],
                torch.full((1, 1), step, device=device, dtype=torch.int64),
                offset.reshape(1, 1) & 0xFFFFFFFF,
            )
            words = torch.broadcast_tensors(*words)
            bits = torch.stack(words, dim=-1).reshape(rows, -1)[:, :padded]
            uniform = (bits.float() + 0.5) * _UNIFORM_SCALE.value
            exp_noise = -torch.log1p(-uniform)
            sampled = temperatures != 0
            safe_temperatures = torch.where(sampled, temperatures, 1.0)[:, None]
            noised = value / safe_temperatures - torch.log(exp_noise)
            keys = torch.where(sampled[:, None], noised, value)
        cols = torch.arange(padded, device=device)
        keys = torch.where(cols[None, :] < width, keys, -torch.inf)

        if width == 0:
            local_values = torch.full((rows,), -torch.inf, device=device)
            local_indices = torch.full(
                (rows,), _NO_INDEX.value, dtype=torch.int32, device=device
            )
        else:
            local = keys.argmax(dim=1)
            local_values = keys.gather(1, local[:, None]).squeeze(1)
            local_indices = (local + vocab_start).to(torch.int32)
        partial = torch.stack(
            [local_values.float(), local_indices.view(torch.float32)], dim=1
        )
        gathered = partial if gather is None else gather(partial)
        gathered = gathered.view(-1, rows, 2)
        values = gathered[..., 0].transpose(0, 1)
        indices = gathered[..., 1].contiguous().view(torch.int32) & _NO_INDEX.value
        indices = indices.transpose(0, 1).long()
        # Lowest index among the NaNs if any, else among the maxima.
        valid = indices != _NO_INDEX.value
        nan = valid & values.isnan()
        has_nan = nan.any(dim=-1, keepdim=True)
        finite = torch.where(valid & ~nan, values, -torch.inf)
        maximum = finite.amax(dim=-1, keepdim=True)
        wins = torch.where(has_nan, nan, valid & (finite == maximum))
        return torch.where(wins, indices, _NO_INDEX.value).amin(dim=-1)
