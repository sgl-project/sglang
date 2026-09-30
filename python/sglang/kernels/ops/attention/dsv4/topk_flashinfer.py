"""FlashInfer row selection followed by the sparse indexer's physical remap."""

from typing import Callable

import torch
import triton
import triton.language as tl

from .candidate_table import CANDIDATE_BLOCK_SIZE


@triton.jit
def _remap_sparse_topk(
    raw_indices,
    blocks,
    out_indices,
    raw_stride: tl.constexpr,
    block_stride: tl.constexpr,
    out_stride: tl.constexpr,
    top_k: tl.constexpr,
    score_width: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(raw_indices + row * raw_stride + col, col < top_k, -1)
    valid = (col < top_k) & (selected >= 0) & (selected < score_width)
    physical = tl.load(
        blocks + row * block_stride + selected // block_size,
        valid,
        0,
    )
    mapped = physical * block_size + selected % block_size
    tl.store(
        out_indices + row * out_stride + col, tl.where(valid, mapped, -1), col < top_k
    )


def topk_transform_sparse_flashinfer(
    logits: torch.Tensor,
    valid_lens: torch.Tensor,
    blocks: torch.Tensor,
    out_indices: torch.Tensor,
    topk_op: Callable,
) -> bool:
    """Return False before allocation/launch when the input layout is unsupported."""
    if logits.ndim != 2:
        return False
    rows, width = logits.shape
    if (
        rows == 0
        or width == 0
        or logits.dtype != torch.bfloat16
        or not logits.is_contiguous()
        or logits.data_ptr() % 16 != 0
        or valid_lens.dtype != torch.int32
        or valid_lens.shape != (rows,)
        or not valid_lens.is_contiguous()
        or blocks.ndim != 2
        or blocks.dtype != torch.int32
        or blocks.shape[0] != rows
        or blocks.shape[1] * CANDIDATE_BLOCK_SIZE < width
        or blocks.stride(1) != 1
        or out_indices.ndim != 2
        or out_indices.dtype != torch.int32
        or out_indices.shape[0] != rows
        or out_indices.shape[1] == 0
        or out_indices.stride(1) != 1
        or out_indices.stride(0) < out_indices.shape[1]
    ):
        return False
    if not logits.is_cuda or any(
        tensor.device != logits.device for tensor in (valid_lens, blocks, out_indices)
    ):
        return False
    if any(
        tensor.is_neg() or tensor.is_conj()
        for tensor in (logits, valid_lens, blocks, out_indices)
    ):
        return False
    # This invocation owns its temporary. CUDA graph capture retains the
    # allocation, without a shared tensor cache across graphs or streams.
    with torch.cuda.device(logits.device):
        raw_indices = torch.empty(
            out_indices.shape, dtype=torch.int32, device=out_indices.device
        )
        selected, values = topk_op(
            logits,
            valid_lens,
            out_indices.shape[1],
            compress_ratio=1,
            next_n=1,
            return_values=False,
            out_indices=raw_indices,
            backend="auto",
        )
        if selected is not raw_indices or values is not None:
            raise RuntimeError(
                "FlashInfer varlen top-k violated its caller-output contract"
            )
        _remap_sparse_topk[(rows, triton.cdiv(out_indices.shape[1], 256))](
            raw_indices,
            blocks,
            out_indices,
            raw_indices.stride(0),
            blocks.stride(0),
            out_indices.stride(0),
            out_indices.shape[1],
            width,
            CANDIDATE_BLOCK_SIZE,
            BLOCK=256,
        )
    return True
