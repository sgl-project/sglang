"""SM90 grouped FP8 indexer, adapted from sgl-project/sglang#40160.

Logical H32/H64 inputs share a 64-row WGMMA tile. H32 padding lives only in
shared memory; no padded model projections or global scratch are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(heads: int) -> Module:
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("sm90_fp4_grouped_indexer requires an SM90 GPU")
    return load_jit(
        f"sm90_fp4_grouped_indexer_h{heads}",
        cuda_files=["sm90_fp4_grouped_indexer/entry.cuh"],
        cuda_wrappers=[("dispatch", f"sm90_fp4_grouped_indexer_dispatch<{heads}>")],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


@register_custom_op(op_name="sm90_fp4_grouped_indexer", mutates_args=["out"])
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
    _jit_module(q.shape[1]).dispatch(
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
        torch.cuda.current_stream(q.device).cuda_stream,
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
    """Score consecutive query groups sharing one request's FP4 index-K cache.

    The caller guarantees equal request IDs within each group; the last group
    may be partial. Q must already be FP4-representable, as produced by the
    V4.1 indexer. Each row keeps its own device-side causal length. Only the
    upstream rectangular grid is ported; CTAs never wait for other CTAs.
    """
    assert q.is_cuda and q.dtype == torch.bfloat16 and q.ndim == 3
    rows, heads, dim = q.shape
    assert heads in (32, 64) and dim == 128
    assert weights.dtype == torch.bfloat16 and weights.shape == (rows, heads)
    assert req_to_token.dtype == torch.int32 and req_to_token.ndim == 2
    assert req.shape == lens.shape == (rows,)
    assert req.dtype in (torch.int32, torch.int64)
    assert lens.dtype in (torch.int32, torch.int64)
    assert table.dtype == torch.uint8 and table.ndim == 2
    assert page_size > 0 and table.shape[1] >= page_size * 68
    assert req_to_token.stride(1) == table.stride(1) == 1
    assert ratio in (1, 2) and 0 <= width <= req_to_token.shape[1] // ratio
    assert 2 <= group_size <= 6
    assert all(t.device == q.device for t in (weights, req_to_token, req, lens, table))
    assert torch.cuda.current_device() == q.device.index

    # Candidate-source amax consumes eight scores per vector. Keep the same
    # row alignment as fp4_index_logits_paged, including odd replay capacities.
    out = torch.empty(
        (rows, (width + 7) // 8 * 8), dtype=torch.float32, device=q.device
    )[:, :width]
    if not rows or not width:
        return out

    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )

    q_fp4, q_scale = quantize_fp4_indexer_tensor(q, rne=True)
    _sm90_fp4_grouped_indexer_op(
        q_fp4.view(rows, heads, 64).view(torch.uint8),
        q_scale.view(rows, heads),
        weights.contiguous(),
        req_to_token,
        req.to(torch.int64).contiguous(),
        lens.to(torch.int64).contiguous(),
        table,
        out,
        width,
        group_size,
        page_size,
        ratio,
    )
    return out
