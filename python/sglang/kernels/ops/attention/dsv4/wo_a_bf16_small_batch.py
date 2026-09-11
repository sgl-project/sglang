"""Split-K grouped BF16 projection for small speculative batches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wo_a_partial(X, W, P, M: tl.constexpr, SX: tl.constexpr):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        w = tl.load(W + (group * 1024 + n[None, :]) * 4096 + offsets[:, None])
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


@triton.jit
def _wo_a_reduce(P, Y, E: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    tl.store(Y + i, tl.sum(values, 0), i < E)


def wo_a_bf16_small_batch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for the TP4 WO-A shape.

    Eight K partitions expose more CTAs than the small-M batched GEMM.
    Partial sums stay in FP32 until the final BF16, token-major store.
    """
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    _wo_a_reduce[(triton.cdiv(m * 2048, 256),)](partial, result, m * 2048, num_warps=4)
    return result
