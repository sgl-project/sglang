"""Fused Triton kernels that clear / copy conv-state pool slots across all
conv-state tensors of a hybrid (mamba-style) pool in a single launch.

``MambaPool.clear_slots`` / ``copy_from`` otherwise loop over every conv-state
tensor (models with several short-conv streams have a handful, each a distinct
scattered-index kernel), and the speculative-decode draft worker replays that
loop across every draft head's pool — so one fresh-request / radix-COW event
fans out into many tiny launch-bound kernels on the forward stream. These
kernels fold the whole conv list into one launch.

The conv tensors are heterogeneous only in their trailing feature size and share
the leading ``[num_layers, pool_size]`` dims + dtype, so they are addressed via a
per-tensor pointer / stride / feature-length array. The kernel reads each
tensor's real ``layer_stride`` / ``slot_stride``, so it is layout-general; it
only requires the per-slot feature block to be contiguous. Temporal state
(different dtype/shape) is handled by the caller with a plain indexed op.
"""

from __future__ import annotations

from typing import List, NamedTuple

import torch
import triton
import triton.language as tl

_BLOCK = 1024


class ConvSlotDescriptor(NamedTuple):
    ptr: torch.Tensor  # [T] int64 base byte-addresses
    feat: torch.Tensor  # [T] int64 per-slot feature length (elements)
    layer_stride: torch.Tensor  # [T] int64 element stride between layers
    slot_stride: torch.Tensor  # [T] int64 element stride between slots
    num_layers: int
    max_feat_blocks: int


@triton.jit
def _fused_slot_clear_kernel(
    ptr_arr,
    feat_arr,
    layer_stride_arr,
    slot_stride_arr,
    index_arr,
    MAX_FEAT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    iid = tl.program_id(0)
    tid = tl.program_id(1)
    lid = tl.program_id(2)
    base_addr = tl.load(ptr_arr + tid)
    feat = tl.load(feat_arr + tid)
    slot = tl.load(index_arr + iid)
    base = base_addr.to(tl.pointer_type(tl.bfloat16))
    row = (
        base
        + lid * tl.load(layer_stride_arr + tid)
        + slot * tl.load(slot_stride_arr + tid)
    )
    zeros = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    for fb in tl.static_range(MAX_FEAT_BLOCKS):
        cols = fb * BLOCK + tl.arange(0, BLOCK)
        tl.store(row + cols, zeros, mask=cols < feat)


@triton.jit
def _fused_slot_copy_kernel(
    ptr_arr,
    feat_arr,
    layer_stride_arr,
    slot_stride_arr,
    src_arr,
    dst_arr,
    MAX_FEAT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    iid = tl.program_id(0)
    tid = tl.program_id(1)
    lid = tl.program_id(2)
    base_addr = tl.load(ptr_arr + tid)
    feat = tl.load(feat_arr + tid)
    layer_off = lid * tl.load(layer_stride_arr + tid)
    slot_stride = tl.load(slot_stride_arr + tid)
    base = base_addr.to(tl.pointer_type(tl.bfloat16))
    src_row = base + layer_off + tl.load(src_arr + iid) * slot_stride
    dst_row = base + layer_off + tl.load(dst_arr + iid) * slot_stride
    for fb in tl.static_range(MAX_FEAT_BLOCKS):
        cols = fb * BLOCK + tl.arange(0, BLOCK)
        mask = cols < feat
        tl.store(dst_row + cols, tl.load(src_row + cols, mask=mask), mask=mask)


def build_conv_slot_descriptor(tensors: List[torch.Tensor]) -> ConvSlotDescriptor:
    """Build the pool-stable addressing descriptor for a conv-tensor list.

    Requires bf16 tensors sharing the leading (num_layers, pool_size) dims with a
    contiguous per-slot feature block (the kernel reads each tensor's real
    strides, so the block may sit inside a larger strided envelope). Cache the
    result and reuse it — conv tensors don't move after allocation.
    """
    t0 = tensors[0]
    num_layers = t0.shape[0]
    device = t0.device
    ptr, feat, layer_stride, slot_stride = [], [], [], []
    max_feat = 0
    for t in tensors:
        assert t.dtype == torch.bfloat16, "fused slot ops assume bf16 conv state"
        assert t.shape[0] == num_layers, "conv tensors must share num_layers"
        assert t.device == device
        assert t[0, 0].is_contiguous(), "per-slot feature block must be contiguous"
        ptr.append(t.data_ptr())
        feat.append(t[0, 0].numel())
        layer_stride.append(t.stride(0))
        slot_stride.append(t.stride(1))
        max_feat = max(max_feat, t[0, 0].numel())
    to_i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=device)
    # Base addresses can exceed the signed-int64 range on some devices (e.g. XPU
    # maps buffers high in the address space). Wrap them into signed-int64 range
    # in Python — the 64-bit pattern the kernel reinterprets as a pointer is
    # identical, and this avoids torch.uint64, which isn't supported on all
    # PyTorch versions/platforms.
    ptr_signed = [p if p < 2**63 else p - 2**64 for p in ptr]
    ptr_i64 = torch.tensor(ptr_signed, dtype=torch.int64, device=device)
    return ConvSlotDescriptor(
        ptr=ptr_i64,
        feat=to_i64(feat),
        layer_stride=to_i64(layer_stride),
        slot_stride=to_i64(slot_stride),
        num_layers=num_layers,
        max_feat_blocks=triton.cdiv(max_feat, _BLOCK),
    )


def fused_clear_conv_slots(desc: ConvSlotDescriptor, indices: torch.Tensor):
    """Zero ``indices`` slots (dim 1) across every conv tensor in one launch."""
    if desc.ptr.numel() == 0 or indices.numel() == 0:
        return
    index_arr = indices.to(torch.int64)
    # Slot count on the unbounded grid axis (gridDim.y/z cap at 65535).
    grid = (index_arr.numel(), desc.ptr.numel(), desc.num_layers)
    _fused_slot_clear_kernel[grid](
        desc.ptr,
        desc.feat,
        desc.layer_stride,
        desc.slot_stride,
        index_arr,
        MAX_FEAT_BLOCKS=desc.max_feat_blocks,
        BLOCK=_BLOCK,
    )


def fused_copy_conv_slots(
    desc: ConvSlotDescriptor, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy conv state from ``src`` slots to ``dst`` slots across every conv
    tensor in one launch.

    ``src`` and ``dst`` must be disjoint (the COW invariant: radix-checkpoint
    slots copied into freshly-allocated slots). Unlike the gather-then-scatter
    reference, this kernel reads and writes in one pass, so overlapping ranges
    would race.
    """
    if desc.ptr.numel() == 0 or src_indices.numel() == 0:
        return
    src_arr = src_indices.to(torch.int64)
    dst_arr = dst_indices.to(torch.int64)
    grid = (src_arr.numel(), desc.ptr.numel(), desc.num_layers)
    _fused_slot_copy_kernel[grid](
        desc.ptr,
        desc.feat,
        desc.layer_stride,
        desc.slot_stride,
        src_arr,
        dst_arr,
        MAX_FEAT_BLOCKS=desc.max_feat_blocks,
        BLOCK=_BLOCK,
    )
