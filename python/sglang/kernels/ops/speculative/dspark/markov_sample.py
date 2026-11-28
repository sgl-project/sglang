"""One Markov draft step: add the step bias and pick one token per row, the
argmax at temperature 0 and a Gumbel-max sample otherwise."""

import functools
from typing import Callable, Optional

import msgspec
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.srt.utils import is_hip, is_npu

_is_hip = is_hip()
_is_npu = is_npu()

# Measured fastest for 1-64 rows, where the per-block Philox work sets latency.
_BLOCK = 1024
# block width when the kernel computes the step bias (fold): the fold weight dominates
_FOLD_BLOCK = 128
_FOLD_BLOCK_R = 64
_FOLD_MAX_ROWS = 16
_FOLD_L2_SHARE = 0.8
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
    softcap,
    vocab_limit,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE: tl.constexpr,
    SELECT: tl.constexpr,
    SAMPLE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    SOFTCAP: tl.constexpr = False,
    LIMIT: tl.constexpr = False,
    FW=None,
    FB=None,
    FOLD: tl.constexpr = False,
    RANK: tl.constexpr = 1,
):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    in_shard = i < PADDED
    value = tl.load(X + row * XS + i, in_shard, 0).to(tl.float32)
    if FOLD:
        # The bias as torch.addmm(FB, B, FW.T) gives it (B: the rows' latents), in
        # FW's dtype; the sum order is not the GEMM's.
        k = tl.arange(0, RANK)
        latent = tl.load(B + row * BS + k).to(FW.dtype.element_ty).to(tl.float32)
        w = tl.load(FW + i[:, None] * RANK + k[None, :], valid[:, None], 0)
        bias = tl.sum(w.to(tl.float32) * latent[None, :], 1)
        bias += tl.load(FB + i, valid, 0).to(tl.float32)
        value += bias.to(FW.dtype.element_ty).to(tl.float32)
    elif HAS_BIAS:
        value += tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    value = _round_to_step_dtype(value, ROUND)
    if SOFTCAP:
        # softcap_inplace_logits on the rounded step logits.
        value = _round_to_step_dtype(libdevice.tanh(value / softcap) * softcap, ROUND)
    if LIMIT:
        value = tl.where(i + VOCAB_START < vocab_limit, value, -float("inf"))
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


@triton.jit
def _markov_fold_partial_kernel(
    Latent,
    W,
    FoldBias,
    X,
    T,
    OFFSET,
    L,
    P,
    LS,
    XS,
    OS,
    WIDTH,
    PARTS,
    resident_parts,
    KEY,
    STEP,
    softcap,
    vocab_limit,
    ROWS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROUND: tl.constexpr,
    STORE: tl.constexpr,
    SAMPLE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    SOFTCAP: tl.constexpr,
    LIMIT: tl.constexpr,
    KEEP: tl.constexpr,
):
    part = tl.program_id(0)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    rows = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK), tl.float32)
    for r0 in tl.static_range(0, RANK, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        latent = tl.load(
            Latent + rows[:, None] * LS + r[None, :], rows[:, None] < ROWS, other=0
        )
        w_ptr = W + i[None, :].to(tl.int64) * RANK + r[:, None]
        if KEEP and part < resident_parts:
            w = tl.load(w_ptr, valid[None, :], other=0, eviction_policy="evict_last")
        else:
            w = tl.load(w_ptr, valid[None, :], other=0, eviction_policy="evict_first")
        acc = tl.dot(latent, w, acc)
    bias = acc + tl.load(FoldBias + i, valid, other=0).to(tl.float32)[None, :]
    bias = bias.to(tl.bfloat16).to(tl.float32)
    for row in tl.static_range(ROWS):
        value = tl.load(X + row * XS + i, valid, 0).to(tl.float32)
        value += tl.sum(tl.where(rows[:, None] == row, bias, 0.0), 0)
        value = _round_to_step_dtype(value, ROUND)
        if SOFTCAP:
            value = _round_to_step_dtype(
                libdevice.tanh(value / softcap) * softcap, ROUND
            )
        if LIMIT:
            value = tl.where(i < vocab_limit, value, -float("inf"))
        if STORE:
            tl.store(L + row * OS + i, value.to(L.dtype.element_ty), valid)
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
            0,
            BLOCK,
            SAMPLE,
            PHILOX_ROUNDS,
        )


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
    softcap=None,
    vocab_limit=None,
    parts_multiple=1,
    fold=None,
):
    # Returns the [rows, parts * 2] (value, index bits) candidates of each block.
    # fold=(fold weight, fold bias): bias holds the rows' latents and the kernel
    # computes the step bias, in narrower blocks that spread that GEMV's reads.
    rows, padded = base_logits.shape
    block = _BLOCK if fold is None else _FOLD_BLOCK
    # The padded width fixes the transport shape on every rank; WIDTH masks padding.
    # Blocks past the shard (parts_multiple) write (-inf, no index) candidates.
    parts = triton.cdiv(triton.cdiv(padded, block), parts_multiple) * parts_multiple
    assert parts > 0
    fold_args = {}
    if fold is not None:
        fold_weight, fold_bias = fold
        fold_args = dict(
            FW=fold_weight, FB=fold_bias, FOLD=True, RANK=fold_weight.shape[1]
        )
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
        float(softcap or 0.0),
        int(vocab_limit or 0),
        block,
        _ROUND_CODE[step_dtype],
        bias is not None,
        store is not None,
        select,
        has_sampled_rows is not False,
        _PHILOX_ROUNDS,
        softcap is not None,
        vocab_limit is not None,
        **fold_args,
        num_warps=4,
    )
    return partial.view(rows, parts * 2)


class MarkovFold(msgspec.Struct, frozen=True):
    """A step bias addmm(bias, latent, weight^T) for the step kernel to compute:
    BF16 latent [rows, rank], weight [>= width, rank] and bias [>= width]."""

    latent: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


def use_fold(rows: int, weight: torch.Tensor) -> bool:
    if rows > _FOLD_MAX_ROWS or weight.shape[1] % _FOLD_BLOCK_R:
        return False
    return rows <= 2 or weight.nbytes > _FOLD_L2_SHARE * current_arch().l2_cache_bytes


def _launch_fold_partial(
    base_logits,
    fold,
    *,
    store,
    temperatures,
    key,
    offset,
    step,
    step_dtype,
    has_sampled_rows,
    softcap,
    vocab_limit,
    keep_in_l2,
):
    rows, width = base_logits.shape
    rank = fold.weight.shape[1]
    parts = triton.cdiv(width, _FOLD_BLOCK)
    partial = torch.empty(
        (rows, parts, 2), device=base_logits.device, dtype=torch.float32
    )
    block_bytes = _FOLD_BLOCK * rank * fold.weight.element_size()
    resident_parts = int(_FOLD_L2_SHARE * current_arch().l2_cache_bytes) // block_bytes
    _markov_fold_partial_kernel[(parts,)](
        fold.latent,
        fold.weight,
        fold.bias,
        base_logits,
        temperatures,
        offset,
        base_logits if store is None else store,
        partial,
        fold.latent.stride(0),
        base_logits.stride(0),
        0 if store is None else store.stride(0),
        width,
        parts,
        resident_parts,
        key,
        step,
        float(softcap or 0.0),
        int(vocab_limit or 0),
        rows,
        rank,
        _FOLD_BLOCK,
        _FOLD_BLOCK_R,
        _ROUND_CODE[step_dtype],
        store is not None,
        has_sampled_rows is not False,
        _PHILOX_ROUNDS,
        softcap is not None,
        vocab_limit is not None,
        keep_in_l2,
        num_warps=4,
    )
    return partial.view(rows, parts * 2)


def markov_fold_step(
    *,
    base_logits: torch.Tensor,
    fold: MarkovFold,
    temperatures: torch.Tensor,
    key: int,
    offset: torch.Tensor,
    step: int,
    step_dtype: torch.dtype,
    logits_out: Optional[torch.Tensor] = None,
    has_sampled_rows: Optional[bool] = None,
    softcap: Optional[float] = None,
    vocab_limit: Optional[int] = None,
    keep_in_l2: bool = False,
) -> torch.Tensor:
    """MarkovSampleStep.triton with an unsharded fold in place of the bias, for at
    most _FOLD_MAX_ROWS rows; keep_in_l2 keeps the fold weight in L2 for a later
    step."""
    rows, width = base_logits.shape
    assert base_logits.stride(1) == 1 and rows <= _FOLD_MAX_ROWS
    assert fold.latent.shape == (rows, fold.weight.shape[1])
    assert fold.weight.dtype == fold.latent.dtype == torch.bfloat16
    assert fold.weight.is_contiguous() and fold.weight.shape[0] >= width
    assert fold.weight.shape[1] % _FOLD_BLOCK_R == 0
    assert fold.latent.stride(1) == 1 and fold.bias.stride(0) == 1
    assert fold.bias.ndim == 1 and fold.bias.dtype == torch.bfloat16
    assert fold.bias.shape[0] >= width
    assert temperatures.shape == (rows,)
    assert temperatures.dtype == torch.float32 and temperatures.stride(0) == 1
    assert offset.dtype == torch.int64 and offset.numel() == 1
    assert step_dtype in _ROUND_CODE, step_dtype
    if logits_out is not None:
        assert logits_out.dtype == step_dtype and logits_out.stride(1) == 1
        assert logits_out.shape == base_logits.shape
    partial = _launch_fold_partial(
        base_logits,
        fold,
        store=logits_out,
        temperatures=temperatures,
        key=key,
        offset=offset,
        step=step,
        step_dtype=step_dtype,
        has_sampled_rows=has_sampled_rows,
        softcap=softcap,
        vocab_limit=vocab_limit,
        keep_in_l2=keep_in_l2,
    )
    parts = partial.shape[1] // 2
    result = torch.empty(rows, device=base_logits.device, dtype=torch.int64)
    _markov_sample_finish_kernel[(rows,)](
        partial,
        result,
        rows,
        parts,
        1,
        triton.next_power_of_2(parts),
        num_warps=4,
    )
    return result


class MarkovSampleStep:
    """Returns global token ids. ``bias`` covers the first ``width`` columns of
    the rank's padded shard ``base_logits``. Sharded, ``gather`` exchanges
    per-block candidates, or ``gather_logits`` the kept ``logits_out`` whole;
    without ``gather_logits`` the kept logits stay on the shard.
    ``has_sampled_rows=False`` argmaxes every row. ``fold=(weight, bias)``:
    ``bias`` holds the rows' latents and the step bias is
    ``torch.addmm(fold bias, latents, weight.T)`` over ``weight``'s rows, which the
    Triton path computes in the candidate kernel."""

    @classmethod
    def execute(cls, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(**kwargs):
            if _is_npu:
                from sgl_kernel_npu.operators.sample import markov_sample_step_triton

                return markov_sample_step_triton(**kwargs)
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
        fold=None,
    ):
        assert base_logits.ndim == bias.ndim == 2
        assert bias.shape[0] == base_logits.shape[0]
        assert bias.stride(1) == base_logits.stride(1) == 1
        if fold is None:
            assert bias.shape[1] <= base_logits.shape[1]
        else:
            assert gather_logits is None
            weight, fold_bias = fold
            assert weight.is_contiguous() and fold_bias.is_contiguous()
            assert weight.shape == (fold_bias.shape[0], bias.shape[1])
            assert weight.shape[0] <= base_logits.shape[1]
            assert triton.next_power_of_2(weight.shape[1]) == weight.shape[1]
        assert temperatures.shape == (base_logits.shape[0],)
        assert temperatures.dtype == torch.float32 and temperatures.stride(0) == 1
        assert offset.dtype == torch.int64 and offset.numel() == 1
        assert step_dtype in _ROUND_CODE, step_dtype
        # Philox words go to tokens in quads of global ids.
        assert vocab_start % 4 == 0, vocab_start
        if logits_out is not None:
            assert logits_out.dtype == step_dtype and logits_out.stride(1) == 1
            if gather_logits is None:
                assert logits_out.shape == base_logits.shape
            else:
                assert gather is not None
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
        softcap: Optional[float] = None,
        vocab_limit: Optional[int] = None,
        fold: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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
            fold,
        )
        rows, width = bias.shape if fold is None else (bias.shape[0], fold[0].shape[0])
        launch = functools.partial(
            _launch_partial,
            temperatures=temperatures,
            key=key,
            offset=offset,
            step=step,
            step_dtype=step_dtype,
            has_sampled_rows=has_sampled_rows,
        )
        if gather_logits is not None and logits_out is not None:
            local = torch.empty(
                base_logits.shape, device=base_logits.device, dtype=step_dtype
            )
            launch(
                base_logits,
                bias,
                width=width,
                store=local,
                select=False,
                vocab_start=vocab_start,
                softcap=softcap,
                vocab_limit=vocab_limit,
            )
            full = gather_logits(local)[:, : logits_out.shape[1]]
            gathered = launch(full, None, width=full.shape[1], store=logits_out)
        else:
            # Kept shards on ROCm: candidate rows of 16-byte multiples, which aiter's
            # all-gather requires.
            keeps_shard = _is_hip and gather is not None and logits_out is not None
            partial = launch(
                base_logits,
                bias,
                width=width,
                store=logits_out,
                vocab_start=vocab_start,
                softcap=softcap,
                vocab_limit=vocab_limit,
                parts_multiple=2 if keeps_shard else 1,
                fold=fold,
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
        softcap: Optional[float] = None,
        vocab_limit: Optional[int] = None,
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
        if softcap is not None:
            value = (torch.tanh(value / softcap) * softcap).to(step_dtype).float()
        if vocab_limit is not None:
            columns = vocab_start + torch.arange(value.shape[1], device=value.device)
            value[:, columns >= vocab_limit] = float("-inf")
        if gather_logits is not None and logits_out is not None:
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
