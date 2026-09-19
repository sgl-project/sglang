"""SM90 FP8 Tensor Core indexer with request-grouped K reuse."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("sm90_fp4_grouped_indexer requires an SM90 GPU")
    return load_jit(
        "sm90_fp4_grouped_indexer",
        cuda_files=["sm90_fp4_grouped_indexer/entry.cuh"],
        cuda_wrappers=[("dispatch", "sm90_fp4_grouped_indexer_dispatch")],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


_get_current_stream_raw = torch._C._cuda_getCurrentRawStream


@register_custom_op(
    op_name="sm90_fp4_grouped_indexer",
    mutates_args=["out"],
)
def _sm90_fp4_grouped_indexer_op(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    out: torch.Tensor,
    width: int,
    group_size: int,
    page_size: int,
    ratio: int,
) -> None:
    _jit_module().dispatch(
        q,
        q_scale,
        weights,
        req_to_token,
        req,
        lens,
        table,
        out,
        q.shape[0],
        width,
        group_size,
        page_size,
        ratio,
        q.stride(0),
        q.stride(1),
        q_scale.stride(0),
        weights.stride(0),
        req_to_token.stride(0),
        table.stride(0),
        out.stride(0),
        _get_current_stream_raw(q.device.index),
    )


@debug_kernel_api
def fp4_index_logits_grouped_sm90(
    q: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    width: int,
    group_size: int,
) -> torch.Tensor:
    """Score uniform request-major query groups while reusing each K tile."""
    assert q.dtype == torch.bfloat16 and q.shape[1:] == (64, 128)
    assert weights.dtype == torch.bfloat16 and weights.shape == q.shape[:2]
    assert req_to_token.dtype == torch.int32 and req_to_token.dim() == 2
    assert req.dtype == torch.int64 and req.shape == (q.shape[0],)
    assert lens.dtype == torch.int64 and lens.shape == req.shape
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert q.is_contiguous() and weights.is_contiguous()
    assert req.is_contiguous() and lens.is_contiguous()
    assert req_to_token.stride(1) == 1 and table.stride(1) == 1
    assert ratio in (1, 2) and group_size > 1

    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )

    q_fp4, q_scale = quantize_fp4_indexer_tensor(q, rne=True)
    q_fp4 = q_fp4.view(q.shape[0], q.shape[1], 64).view(torch.uint8)
    q_scale = q_scale.view(q.shape[0], q.shape[1])
    storage_width = (width + 3) // 4 * 4
    out_storage = torch.empty(
        (q.shape[0], storage_width), dtype=torch.float32, device=q.device
    )
    out = out_storage[:, :width]
    if width:
        _sm90_fp4_grouped_indexer_op(
            q_fp4,
            q_scale,
            weights,
            req_to_token,
            req,
            lens,
            table,
            out,
            width,
            group_size,
            page_size,
            ratio,
        )
    return out
