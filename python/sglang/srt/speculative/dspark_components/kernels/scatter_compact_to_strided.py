from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    compact_row_index,
)
from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class ScatterCompactToStrided:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided_triton(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )


def scatter_compact_to_strided(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    compact = compact[: layout.graph_num_tokens]
    strided = torch.full(
        (bs * stride + 1, dim), fill_value, dtype=compact.dtype, device=device
    )
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    sink = bs * stride
    strided_pos = torch.where(
        valid,
        req_id.clamp(max=bs - 1) * stride + within,
        torch.full_like(within, sink),
    )
    strided.index_copy_(0, strided_pos, compact)
    return strided[: bs * stride]


@triton.jit
def _scatter_compact_to_strided_kernel(
    compact_ptr,
    verify_lens_ptr,
    start_ptr,
    out_ptr,
    stride,
    dim,
    fill_value,
    BLOCK_D: tl.constexpr,
):
    o = tl.program_id(0).to(tl.int64)
    dblk = tl.program_id(1)
    i = o // stride
    w = o % stride
    vl_i = tl.load(verify_lens_ptr + i)
    start_i = tl.load(start_ptr + i)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < dim
    in_range = w < vl_i
    src = tl.where(in_range, start_i + w, 0)
    val = tl.load(compact_ptr + src * dim + d, mask=dmask & in_range, other=0)
    val = tl.where(in_range, val, fill_value)
    tl.store(out_ptr + o * dim + d, val, mask=dmask)


def scatter_compact_to_strided_into(
    *,
    compact: torch.Tensor,
    verify_lens: torch.Tensor,
    out: torch.Tensor,
    stride: int,
    fill_value: float,
) -> torch.Tensor:
    dim = compact.shape[1]
    fill_value = float(fill_value) if out.dtype.is_floating_point else int(fill_value)
    compact = compact.contiguous()
    verify_lens = verify_lens.to(dtype=torch.int64).contiguous()
    start = (torch.cumsum(verify_lens, dim=0) - verify_lens).contiguous()
    n_out = out.shape[0]
    BLOCK_D = 1024
    grid = (n_out, triton.cdiv(dim, BLOCK_D))
    _scatter_compact_to_strided_kernel[grid](
        compact,
        verify_lens,
        start,
        out,
        stride,
        dim,
        fill_value,
        BLOCK_D=BLOCK_D,
    )
    return out


def scatter_compact_to_strided_triton(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    out = torch.empty((bs * stride, dim), dtype=compact.dtype, device=device)
    return scatter_compact_to_strided_into(
        compact=compact,
        verify_lens=layout.verify_lens.to(device=device),
        out=out,
        stride=stride,
        fill_value=fill_value,
    )
