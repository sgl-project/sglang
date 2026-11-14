"""Materialize FlashInfer KDA's strided inputs in one copy launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_strided(
    X,
    Y,
    block_id,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = block_id * BLOCK + tl.arange(0, BLOCK)
    remaining = linear
    source = tl.full((BLOCK,), 0, tl.int32)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        source += (remaining % SHAPE[dim]) * STRIDES[dim]
        remaining //= SHAPE[dim]
    # Copy the representation, including BF16 NaN payloads and signed zero.
    if X.dtype.element_ty == tl.bfloat16:
        X = X.to(tl.pointer_type(tl.uint16))
        Y = Y.to(tl.pointer_type(tl.uint16))
    else:
        X = X.to(tl.pointer_type(tl.uint32))
        Y = Y.to(tl.pointer_type(tl.uint32))
    value = tl.load(X + source, linear < SIZE, other=0)
    tl.store(Y + linear, value, linear < SIZE)


@triton.jit
def _pack_kda_inputs(
    sources,
    destinations,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    SIZES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    start = 0
    for i in tl.static_range(len(SIZES)):
        if SIZES[i] > 0:
            blocks = tl.cdiv(SIZES[i], BLOCK)
            if (block_id >= start) & (block_id < start + blocks):
                _copy_strided(
                    sources[i],
                    destinations[i],
                    block_id - start,
                    SHAPES[i],
                    STRIDES[i],
                    SIZES[i],
                    BLOCK,
                )
            start += blocks


def contiguous_kda_inputs(q, k, v, g, beta):
    """Preserve each input's values, dtype, shape, and the current CUDA stream.

    Convolution outputs that already provide contiguous Q/K/V groups need at
    most the beta copy. Keep those on the ordinary path; this kernel targets
    the multiple strided views produced by token-major QKV convolution.
    """
    tensors = (q, k, v, g, beta)
    sizes = tuple(0 if t.is_contiguous() else t.numel() for t in tensors)
    if not any(sizes):
        return tensors
    if sum(n > 0 for n in sizes) < 2 or not all(
        t.is_cuda
        and t.device == q.device
        and t.dtype in (torch.bfloat16, torch.float32)
        and not t.requires_grad
        and 1 <= t.ndim <= 4
        and t.numel() < 2**31
        and all(s >= 0 for s in t.stride())
        and sum((d - 1) * s for d, s in zip(t.shape, t.stride())) < 2**31
        for t in tensors
    ):
        return tuple(t.contiguous() for t in tensors)
    outputs = tuple(
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if n else t
        for t, n in zip(tensors, sizes)
    )
    block = 4096
    with torch.cuda.device(q.device):
        _pack_kda_inputs[(sum(triton.cdiv(n, block) for n in sizes),)](
            tensors,
            outputs,
            tuple(tuple(t.shape) for t in tensors),
            tuple(tuple(t.stride()) for t in tensors),
            sizes,
            block,
            num_warps=8,
        )
    return outputs
