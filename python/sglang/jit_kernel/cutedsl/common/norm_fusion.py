from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from einops import rearrange

from sglang.jit_kernel.cutedsl.common.reduce import cta_reduce_sum, warp_reduce_sum

NUM_WARP_THREADS = 32


@cute.jit
def apply_norm(
    norm_type: cutlass.Constexpr,
    num_threads: cutlass.Int32 | cutlass.Constexpr,
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    tWrW: Optional[cute.Tensor],
    tBrB: Optional[cute.Tensor],
    D: cutlass.Int32 | cutlass.Constexpr,
    eps: cutlass.Float32 | cutlass.Constexpr,
) -> cute.Tensor:
    if cutlass.const_expr(norm_type == "rms"):
        return apply_rmsnorm(num_threads, tidx, tXrX, tWrW, D, eps)
    else:
        return apply_layernorm(num_threads, tidx, tXrX, tWrW, tBrB, D, eps)


@cute.jit
def apply_rmsnorm(
    num_threads: cutlass.Int32 | cutlass.Constexpr,
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    tWrW: Optional[cute.Tensor],
    D: cutlass.Int32 | cutlass.Constexpr,
    eps: cutlass.Float32 | cutlass.Constexpr,
) -> cute.Tensor:
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        val += tXrX[idx] * tXrX[idx]
    val = warp_reduce_sum(val)
    acc_sq = cta_reduce_sum(val, num_threads // NUM_WARP_THREADS, tidx)
    acc_sq = cute.rsqrt(acc_sq / D + eps)

    tNrN = cute.make_fragment_like(tXrX)
    if cutlass.const_expr(tWrW is not None):
        for i in range(cute.size(tXrX)):
            tNrN[i] = (tXrX[i] * acc_sq * tWrW[i]).to(tNrN.element_type)
    else:
        for i in range(cute.size(tXrX)):
            tNrN[i] = (tXrX[i] * acc_sq).to(tNrN.element_type)
    return tNrN


@cute.jit
def apply_layernorm(
    num_threads: cutlass.Int32 | cutlass.Constexpr,
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    tWrW: Optional[cute.Tensor],
    tBrB: Optional[cute.Tensor],
    D: cutlass.Int32 | cutlass.Constexpr,
    eps: cutlass.Float32 | cutlass.Constexpr,
) -> cute.Tensor:
    # Reduce mean
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        val += tXrX[idx]
    val = warp_reduce_sum(val)
    val = cta_reduce_sum(val, num_threads // NUM_WARP_THREADS, tidx)
    mean = val / D
    # Reduce variance
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        val += (tXrX[idx] - mean) * (tXrX[idx] - mean)
    val = warp_reduce_sum(val)
    val = cta_reduce_sum(val, num_threads // NUM_WARP_THREADS, tidx)
    factor = cute.rsqrt(val / D + eps)
    # Normalize
    tNrN = cute.make_fragment_like(tXrX)
    if cutlass.const_expr(tWrW is not None):
        for i in range(cute.size(tXrX)):
            tNrN[i] = ((tXrX[i] - mean) * factor * tWrW[i] + tBrB[i]).to(
                tNrN.element_type
            )
    else:
        for i in range(cute.size(tXrX)):
            tNrN[i] = ((tXrX[i] - mean) * factor).to(tNrN.element_type)
    return tNrN


def preprocess_tensor(
    tensor: Optional[torch.Tensor] | int,
    B: int,
    S: int,
    D: int,
) -> Union[Optional[torch.Tensor], int]:
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 1:
            if tensor.numel() == 1:
                return tensor
            return rearrange(tensor, "d -> 1 1 d").expand(B, S, D)
        if tensor.ndim == 2:
            return rearrange(tensor, "b d -> b 1 d").expand(B, S, D)
        if tensor.ndim == 3:
            return tensor.expand(B, S, D)
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"Unsupported tensor ndim: {tensor.ndim}.")
    elif isinstance(tensor, int):
        return tensor
    elif tensor is None:
        return None
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}.")


@cute.jit
def tensor_slice(
    mV: cute.Tensor,
    thr_copy: cute.ThrCopy,
    batch_id: cutlass.Int32,
    seq_id: cutlass.Int32,
    D: Union[cutlass.Int32, cutlass.Constexpr],
    len_f: Union[cutlass.Int32, cutlass.Constexpr],
) -> Tuple[cute.Tensor, cute.Tensor]:
    gV: cute.Tensor
    if cutlass.const_expr(cute.is_static(mV.layout) and cute.size(mV.layout) == 1):
        layout = cute.make_layout(shape=((1, 1), (1,)))
        tVgV = cute.make_tensor(mV.iterator, layout)
        tVrV = cute.make_rmem_tensor(layout, mV.element_type)
        return tVgV, tVrV
    if cutlass.const_expr(len(mV.shape) == 1):
        gV = mV
    elif cutlass.const_expr(len(mV.shape) == 3):
        # NOTE: Use `local_tile`` instead of direct indexing to preserve
        #       base pointer alignment for vectorized gmem loads.
        gV = cute.local_tile(mV, tiler=(1, 1, D), coord=(batch_id, seq_id, 0))[
            0, 0, None
        ]
    elif cutlass.const_expr(len(mV.shape) == 4):
        # Same as above
        gV = cute.local_tile(
            mV,
            tiler=(1, 1, 1, D),
            coord=(batch_id, seq_id // len_f, 0, 0),
        )[0, 0, 0, None]
    else:
        raise NotImplementedError(f"Tensor shape {mV.shape} not supported.")
    tVgV = thr_copy.partition_S(gV)
    tVrV = cute.make_fragment_like(tVgV, tVgV.element_type)
    return tVgV, tVrV
