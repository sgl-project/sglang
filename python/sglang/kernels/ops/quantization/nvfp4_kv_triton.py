"""Portable (Triton) NVFP4 KV quantize, the layout of `NVFP4KVQuantizeUtil`."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import nvfp4_quantize_rows

_ROWS_PER_PROGRAM = 16


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
    payload, scales = nvfp4_quantize_rows(x, N, D, tl.load(global_scale_ptr))
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
