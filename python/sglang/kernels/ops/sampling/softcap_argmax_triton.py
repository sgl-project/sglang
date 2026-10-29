import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _partial(
    X,
    Values,
    Indices,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    CHUNKS: tl.constexpr,
    CAP,
    HAS_CAP: tl.constexpr,
    SANITIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    col = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + row * STRIDE + col, valid, other=0).to(tl.float32)
    if HAS_CAP:
        x = libdevice.tanh(x / CAP) * CAP
    if SANITIZE:
        x = tl.where(x != x, -1.0e30, x)
        x = tl.where(x == float("inf"), 1.0e30, x)
        x = tl.where(x == -float("inf"), -1.0e30, x)
    # torch.argmax unsanitized NaN selects first NaN, ahead of finite/Inf.
    nan = valid & (x != x)
    any_nan = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(tl.where(valid & ~nan, x, -float("inf")), 0)
    selected = valid & tl.where(any_nan, nan, x == best)
    index = tl.min(tl.where(selected, col, 2147483647), 0)
    value = tl.where(any_nan, float("nan"), best)
    tl.store(Values + row * CHUNKS + chunk, value)
    tl.store(Indices + row * CHUNKS + chunk, index)


@triton.jit
def _finish(Values, Indices, Out, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < CHUNKS
    x = tl.load(Values + row * CHUNKS + col, valid, other=-float("inf"))
    ids = tl.load(Indices + row * CHUNKS + col, valid, other=2147483647)
    nan = valid & (x != x)
    any_nan = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(tl.where(valid & ~nan, x, -float("inf")), 0)
    selected = valid & tl.where(any_nan, nan, x == best)
    result = tl.min(tl.where(selected, ids, 2147483647), 0)
    tl.store(Out + row, result)


@triton.jit
def _rank(value, Table):
    bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
    return tl.load(Table + bits).to(tl.uint32, bitcast=True)


@triton.jit
def _rank_partial(
    X,
    Table,
    Ranks,
    Indices,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    value = tl.load(X + row * STRIDE + col, valid, other=0)
    ranks = _rank(value, Table)
    best = tl.max(tl.where(valid, ranks, 0), 0)
    index = tl.min(tl.where(valid & (ranks == best), col, 2147483647), 0)
    tl.store(Ranks + row * CHUNKS + tl.program_id(1), best.to(tl.int32, bitcast=True))
    tl.store(Indices + row * CHUNKS + tl.program_id(1), index)


@triton.jit
def _rank_finish(Ranks, Indices, Out, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < CHUNKS
    ranks = tl.load(Ranks + row * CHUNKS + col, valid, other=0).to(
        tl.uint32, bitcast=True
    )
    ids = tl.load(Indices + row * CHUNKS + col, valid, other=2147483647)
    best = tl.max(tl.where(valid, ranks, 0), 0)
    result = tl.min(tl.where(valid & (ranks == best), ids, 2147483647), 0)
    tl.store(Out + row, result)


def softcap_argmax(logits, softcap=None, sanitize=False, comparison_ranks=None):
    """Reduce BF16 logits after FP32 softcap, retaining first-token ties/NaNs."""
    if logits.ndim != 2 or logits.dtype != torch.bfloat16:
        raise ValueError("softcap_argmax expects rank-two BF16 logits")
    if logits.stride(1) != 1 or logits.shape[1] == 0:
        raise ValueError("softcap_argmax expects nonempty contiguous vocabulary rows")
    if comparison_ranks is not None:
        if sanitize:
            raise ValueError("comparison ranks support unsanitized logits only")
        if (
            comparison_ranks.dtype != torch.int32
            or comparison_ranks.shape != (65536,)
            or not comparison_ranks.is_contiguous()
            or comparison_ranks.device != logits.device
        ):
            raise ValueError(
                "comparison ranks must be contiguous int32[65536] on the logits device"
            )
    if logits.device.type == "meta":
        return torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    if logits.device.type != "xpu":
        raise ValueError("softcap_argmax requires XPU")
    if comparison_ranks is not None:
        m, n = logits.shape
        chunks = triton.cdiv(n, 4096)
        ranks = torch.empty((m, chunks), device=logits.device, dtype=torch.int32)
        indices = torch.empty_like(ranks)
        out = torch.empty((m,), device=logits.device, dtype=torch.int64)
        _rank_partial[(m, chunks)](
            logits,
            comparison_ranks,
            ranks,
            indices,
            n,
            logits.stride(0),
            chunks,
            4096,
            num_warps=4,
        )
        _rank_finish[(m,)](
            ranks, indices, out, chunks, triton.next_power_of_2(chunks), num_warps=4
        )
        return out
    cap = softcap if softcap else None

    m, n = logits.shape
    chunks = triton.cdiv(n, 4096)
    v = torch.empty((m, chunks), device=logits.device, dtype=torch.float32)
    i = torch.empty((m, chunks), device=logits.device, dtype=torch.int32)
    out = torch.empty((m,), device=logits.device, dtype=torch.int64)
    _partial[(m, chunks)](
        logits,
        v,
        i,
        n,
        logits.stride(0),
        chunks,
        cap,
        cap is not None,
        sanitize,
        4096,
        num_warps=4,
    )
    _finish[(m,)](v, i, out, chunks, triton.next_power_of_2(chunks), num_warps=4)
    return out
