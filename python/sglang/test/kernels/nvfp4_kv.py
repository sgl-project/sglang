"""NVFP4 KV quantize in `NVFP4KVQuantizeUtil`'s layout, in Triton: the gfx950 M3.1 kernel
tests build their packed caches with it (the pool stores through the fused qk-norm kernel)."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import nvfp4_f32_to_e2m1

_ROWS_PER_PROGRAM = 16


@triton.jit
def _nvfp4_quantize_rows(x, N: tl.constexpr, D: tl.constexpr, global_scale=1.0):
    """Quantize ``[N, D]`` fp32 rows; returns (payload ``[N, D//2]`` uint8,
    scales ``[N, D//16]`` uint8 holding E4M3 bits)."""
    NBLK: tl.constexpr = D // 16
    blocks = tl.reshape(x, (N, NBLK, 16))
    amax = tl.max(tl.abs(blocks), axis=2)
    scale_e4m3 = (amax / (6.0 * global_scale)).to(tl.float8e4nv)
    scale = scale_e4m3.to(tl.float32)
    inv = tl.where(amax != 0.0, 1.0 / (scale * global_scale), 0.0)
    codes = nvfp4_f32_to_e2m1(blocks * inv[:, :, None])
    lo, hi = tl.split(tl.reshape(codes, (N, D // 2, 2)))
    payload = lo | (hi << 4)
    return payload, scale_e4m3.to(tl.uint8, bitcast=True)


@triton.jit
def _nvfp4_kv_quantize_kernel(
    x_ptr,
    payload_ptr,
    scale_ptr,
    global_scale_ptr,
    num_rows,
    stride_x,
    N: tl.constexpr,
    D: tl.constexpr,
):
    rows = tl.program_id(0) * N + tl.arange(0, N)
    row_mask = rows < num_rows
    rows = rows.to(tl.int64)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x + tl.arange(0, D)[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    payload, scales = _nvfp4_quantize_rows(x, N, D, tl.load(global_scale_ptr))
    tl.store(
        payload_ptr + rows[:, None] * (D // 2) + tl.arange(0, D // 2)[None, :],
        payload,
        mask=row_mask[:, None],
    )
    tl.store(
        scale_ptr + rows[:, None] * (D // 16) + tl.arange(0, D // 16)[None, :],
        scales,
        mask=row_mask[:, None],
    )


def nvfp4_kv_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, D]`` bf16/fp16 -> (``[R, D//2]`` uint8 payload, ``[R, D//16]`` uint8 E4M3 scales)."""
    num_rows, dim = x.shape
    assert dim % 16 == 0 and x.stride(1) == 1
    payload = torch.empty((num_rows, dim // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((num_rows, dim // 16), dtype=torch.uint8, device=x.device)
    if num_rows:
        _nvfp4_kv_quantize_kernel[(triton.cdiv(num_rows, _ROWS_PER_PROGRAM),)](
            x,
            payload,
            scales,
            global_scale,
            num_rows,
            x.stride(0),
            N=_ROWS_PER_PROGRAM,
            D=dim,
        )
    return payload, scales
