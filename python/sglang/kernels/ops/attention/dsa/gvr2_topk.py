"""DSA decode top-k with FlashInfer's self-sampling GVR V2 kernel.

``flashinfer.top_k_varlen(backend="gvr_2")`` selects row-local token positions,
which :func:`gvr2_topk` returns (the unfused top-k contract);
:func:`gvr2_topk_transform_paged` maps them through the indexer's page table to
physical KV slots, the contract of ``fast_topk_transform_fused``. FlashInfer
builds with ``top_k_varlen(page_table=...)`` write the slots from the kernel
itself; older builds get a separate Triton lookup.

CUDA graphs: each geometry must run eagerly on the capturing stream before
capture (compiles the launcher and creates that stream's workspace slab); the
graph runners' warm-up passes do exactly that.
"""

from __future__ import annotations

import functools
from typing import Optional, Union

import torch
import triton
import triton.language as tl

GVR2_TOPK_SIZES = (512, 1024, 2048)
_GVR2_DEVICE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


@functools.cache
def gvr2_device_supported(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in _GVR2_DEVICE_CAPABILITIES


def gvr2_check_config(device: Union[str, torch.device], topk: int) -> None:
    """Raise unless GVR V2 can serve ``topk`` on ``device`` (checked at startup)."""
    device = torch.device(device)
    if topk not in GVR2_TOPK_SIZES:
        raise ValueError(
            f"flashinfer_gvr2 supports index top-k {GVR2_TOPK_SIZES}, got {topk}"
        )
    if device.type != "cuda":
        raise ValueError(f"flashinfer_gvr2 needs a CUDA device, got {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not gvr2_device_supported(index):
        raise ValueError(
            "flashinfer_gvr2 needs an SM 100/103/107 GPU, got SM "
            "%d%d" % torch.cuda.get_device_capability(index)
        )


def _gvr2_unsupported(logits: torch.Tensor, topk: int) -> Optional[str]:
    if topk not in GVR2_TOPK_SIZES:
        return f"top-k {topk} not in {GVR2_TOPK_SIZES}"
    if not logits.is_cuda or not gvr2_device_supported(logits.device.index):
        return f"device {logits.device} is not SM 100/103/107"
    if logits.dtype != torch.float32 or logits.dim() != 2:
        return f"logits must be 2-D float32, got {logits.dim()}-D {logits.dtype}"
    if logits.stride(1) != 1 or (
        logits.shape[0] > 1 and logits.stride(0) < logits.shape[1]
    ):
        return f"logits rows must be contiguous, got strides {logits.stride()}"
    if logits.stride(0) % 4 != 0 or logits.data_ptr() % 16 != 0:
        return "logits rows must be 16-byte aligned"
    return None


def gvr2_supports(logits: torch.Tensor, topk: int) -> bool:
    """Whether the GVR V2 kernel accepts this score buffer and ``topk``."""
    return _gvr2_unsupported(logits, topk) is None


def gvr2_check(logits: torch.Tensor, topk: int) -> None:
    """Raise with the reason if the GVR V2 kernel does not accept this call."""
    reason = _gvr2_unsupported(logits, topk)
    if reason is not None:
        raise RuntimeError(f"flashinfer_gvr2: {reason}")


@functools.cache
def gvr2_fused_page_table_available() -> bool:
    """Whether the installed FlashInfer's gvr_2 accepts ``page_table=``."""
    import inspect

    from flashinfer import top_k_varlen

    return "page_table" in inspect.signature(top_k_varlen).parameters


def _gvr2_fused_page_table(
    logits: torch.Tensor, page_table: torch.Tensor, page_size: int
) -> bool:
    return (
        gvr2_fused_page_table_available()
        and page_table.dtype == torch.int32
        and page_table.is_contiguous()
        and page_table.shape[1] * page_size >= logits.shape[1]
    )


@triton.jit
def _gvr2_page_transform_kernel(
    indices_ptr,
    page_table_ptr,
    page_table_stride,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    in_row = offsets < TOPK
    positions = tl.load(indices_ptr + row * TOPK + offsets, mask=in_row, other=-1)
    valid = positions >= 0
    pages = tl.load(
        page_table_ptr + row * page_table_stride + positions // PAGE_SIZE,
        mask=in_row & valid,
        other=0,
    )
    slots = tl.where(valid, pages * PAGE_SIZE + positions % PAGE_SIZE, -1)
    tl.store(indices_ptr + row * TOPK + offsets, slots, mask=in_row)


def gvr2_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Top-k of ``logits[r, :lengths[r]]`` as row-local positions, ``-1`` padded.

    ``lengths`` must be non-negative; rows with ``lengths[r] <= topk`` select
    every column.
    """
    from flashinfer import top_k_varlen

    assert lengths.shape == (logits.shape[0],)
    indices, _ = top_k_varlen(
        logits,
        lengths.to(torch.int32),
        topk,
        backend="gvr_2",
    )
    return indices


def gvr2_topk_transform_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """:func:`gvr2_topk` as physical KV slots, ``-1`` padded.

    Row ``r`` maps through ``page_table[r]`` (page ``page_size``).
    """
    num_rows = logits.shape[0]
    assert page_table.shape[0] == num_rows and page_table.stride(1) == 1
    if _gvr2_fused_page_table(logits, page_table, page_size):
        from flashinfer import top_k_varlen

        assert lengths.shape == (num_rows,)
        indices, _ = top_k_varlen(
            logits,
            lengths.to(torch.int32),
            topk,
            page_table=page_table,
            page_size=page_size,
            backend="gvr_2",
        )
        return indices
    indices = gvr2_topk(logits, lengths, topk)
    if num_rows > 0:
        _gvr2_page_transform_kernel[(num_rows,)](
            indices,
            page_table,
            page_table.stride(0),
            TOPK=topk,
            PAGE_SIZE=page_size,
            BLOCK=triton.next_power_of_2(topk),
            num_warps=4,
        )
    return indices
