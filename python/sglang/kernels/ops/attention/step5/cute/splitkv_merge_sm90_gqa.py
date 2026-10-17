# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from . import cute_utils as utils
from .compile_cache import cached_compile_function

_THREADS_PER_BLOCK_D128 = 128


_THREADS_PER_BLOCK_D192 = 256


_MAX_HEAD_DIM = 192


_LN2 = 0.6931471805599453


_LOG2E = 1.4426950408889634


_DYNAMIC_SPLIT_CAPACITY = 16


_MIN_WINDOWS_PER_DYNAMIC_SPLIT = 32


_MIN_DYNAMIC_SPLIT_WAVE_NUMERATOR = 7


_MIN_DYNAMIC_SPLIT_WAVE_DENOMINATOR = 8


def _gqa_cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


def _threads_per_block_for_head_dim(head_dim: int) -> int:
    head_dim = int(head_dim)
    if head_dim <= _THREADS_PER_BLOCK_D128:
        return _THREADS_PER_BLOCK_D128
    if head_dim <= _MAX_HEAD_DIM:
        return _THREADS_PER_BLOCK_D192
    raise ValueError(f"head_dim must be in (0, {_MAX_HEAD_DIM}], got {head_dim}")


@cute.kernel
def _merge_variable_split_nat_lse_states_kernel_gqa(
    mPartialOut: cute.Tensor,
    mPartialLSE: cute.Tensor,
    mOut: cute.Tensor,
    mLSE: cute.Tensor,
    batch: Int32,
    num_heads: Int32,
    head_dim: Int32,
    n_split4: Int32,
    n_split2: Int32,
):
    batch_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    dim = Int32(tidx)

    if batch_idx < batch and head_idx < num_heads:
        work_offset = Int32(0)
        split_count = Int32(1)
        if batch_idx < n_split4:
            work_offset = batch_idx * Int32(4)
            split_count = Int32(4)
        elif batch_idx < n_split4 + n_split2:
            local_batch = batch_idx - n_split4
            work_offset = n_split4 * Int32(4) + local_batch * Int32(2)
            split_count = Int32(2)
        else:
            local_batch = batch_idx - n_split4 - n_split2
            work_offset = n_split4 * Int32(4) + n_split2 * Int32(2) + local_batch
            split_count = Int32(1)

        s_max = -Float32.inf
        for split_idx_const in cutlass.range_constexpr(4):
            split_idx = Int32(split_idx_const)
            if split_idx < split_count:
                s = Float32(mPartialLSE[work_offset + split_idx, head_idx])
                s_max = cute.arch.fmax(s_max, s)
        valid_sum = s_max > -Float32.inf
        denom = Float32(0.0)
        acc = Float32(0.0)
        for split_idx_const in cutlass.range_constexpr(4):
            split_idx = Int32(split_idx_const)
            if split_idx < split_count:
                s = Float32(mPartialLSE[work_offset + split_idx, head_idx])
                w = Float32(0.0)
                if valid_sum:
                    w = utils.exp2f((s - s_max) * Float32(_LOG2E))
                    denom += w
                if dim < head_dim:
                    v = Float32(mPartialOut[work_offset + split_idx, head_idx, dim])
                    acc += v * w
        if dim < head_dim:
            merged = Float32(0.0)
            if valid_sum:
                merged = acc / denom
            mOut[batch_idx, head_idx, dim] = mOut.element_type(merged)
        if tidx == Int32(0):
            lse = -Float32.inf
            if valid_sum:
                lse = s_max + utils.log2f(denom) * Float32(_LN2)
            mLSE[batch_idx, head_idx] = lse


def _make_merge_dynamic_split_nat_lse_states_kernel_gqa(*, mtp_q_len: int):
    @cute.kernel
    def _merge_dynamic_split_nat_lse_states_kernel_gqa(
        mPartialOut: cute.Tensor,
        mPartialLSE: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        mHeadGate: Optional[cute.Tensor],
        batch: Int32,
        num_heads: Int32,
        head_dim: Int32,
        sm_count: Int32,
    ):
        if cutlass.const_expr(mHeadGate is not None):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        batch_idx, head_idx, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        dim = Int32(tidx)

        @cute.struct
        class SharedStorage:
            weights: cute.struct.MemRange[Float32, _DYNAMIC_SPLIT_CAPACITY]
            schedule: cute.struct.MemRange[Int32, 2]
            stats: cute.struct.MemRange[Float32, 2]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sWeights = storage.weights.get_tensor(
            cute.make_layout((_DYNAMIC_SPLIT_CAPACITY,), stride=(1,))
        )
        sSchedule = storage.schedule.get_tensor(cute.make_layout((2,), stride=(1,)))
        sStats = storage.stats.get_tensor(cute.make_layout((2,), stride=(1,)))

        if batch_idx < batch and head_idx < num_heads:
            if tidx == Int32(0):
                active_count = Int32(0)
                for row in cutlass.range(batch, unroll=1):
                    if mRegionCounts[row] > Int32(0):
                        active_count += Int32(1)

                target_parts = Int32(1)
                if active_count > Int32(0):
                    min_wave_work = sm_count * Int32(_MIN_DYNAMIC_SPLIT_WAVE_NUMERATOR)
                    split_scale = Int32(_MIN_DYNAMIC_SPLIT_WAVE_DENOMINATOR)
                    if active_count * split_scale < min_wave_work:
                        target_parts = Int32(16)
                        if active_count * Int32(2) * split_scale >= min_wave_work:
                            target_parts = Int32(2)
                        elif active_count * Int32(4) * split_scale >= min_wave_work:
                            target_parts = Int32(4)
                        elif active_count * Int32(8) * split_scale >= min_wave_work:
                            target_parts = Int32(8)
                if cutlass.const_expr(mtp_q_len > 1):
                    # Match the SM103 decode scheduler: four MTP rows can use
                    # all split slots without leaving poisoned partials.
                    target_parts = Int32(16)

                work_offset = Int32(0)
                split_count = Int32(0)
                for row in cutlass.range(batch, unroll=1):
                    region_count = mRegionCounts[row]
                    parts = Int32(0)
                    if region_count > Int32(0):
                        parts = Int32(1)
                        if target_parts >= Int32(2) and region_count >= Int32(
                            2 * _MIN_WINDOWS_PER_DYNAMIC_SPLIT
                        ):
                            parts = Int32(2)
                        if target_parts >= Int32(4) and region_count >= Int32(
                            4 * _MIN_WINDOWS_PER_DYNAMIC_SPLIT
                        ):
                            parts = Int32(4)
                        if target_parts >= Int32(8) and region_count >= Int32(
                            8 * _MIN_WINDOWS_PER_DYNAMIC_SPLIT
                        ):
                            parts = Int32(8)
                        if target_parts >= Int32(16) and region_count >= Int32(
                            16 * _MIN_WINDOWS_PER_DYNAMIC_SPLIT
                        ):
                            parts = Int32(16)
                    if row < batch_idx:
                        work_offset += parts
                    elif row == batch_idx:
                        split_count = parts

                s_max = -Float32.inf
                for split_idx_const in cutlass.range_constexpr(_DYNAMIC_SPLIT_CAPACITY):
                    split_idx = Int32(split_idx_const)
                    if split_idx < split_count:
                        partial_lse = Float32(
                            mPartialLSE[work_offset + split_idx, head_idx]
                        )
                        s_max = cute.arch.fmax(s_max, partial_lse)
                valid_sum = s_max > -Float32.inf
                denom = Float32(0.0)
                for split_idx_const in cutlass.range_constexpr(_DYNAMIC_SPLIT_CAPACITY):
                    split_idx = Int32(split_idx_const)
                    weight = Float32(0.0)
                    if split_idx < split_count and valid_sum:
                        partial_lse = Float32(
                            mPartialLSE[work_offset + split_idx, head_idx]
                        )
                        weight = utils.exp2f((partial_lse - s_max) * Float32(_LOG2E))
                        denom += weight
                    sWeights[split_idx] = weight
                sSchedule[0] = work_offset
                sSchedule[1] = split_count
                sStats[0] = s_max
                sStats[1] = denom
            cute.arch.sync_threads()

            work_offset = sSchedule[0]
            split_count = sSchedule[1]
            s_max = sStats[0]
            denom = sStats[1]
            valid_sum = s_max > -Float32.inf
            acc = Float32(0.0)
            for split_idx_const in cutlass.range_constexpr(_DYNAMIC_SPLIT_CAPACITY):
                split_idx = Int32(split_idx_const)
                if split_idx < split_count:
                    if dim < head_dim:
                        value = Float32(
                            mPartialOut[work_offset + split_idx, head_idx, dim]
                        )
                        acc += value * sWeights[split_idx]
            if dim < head_dim:
                merged = Float32(0.0)
                if valid_sum:
                    merged = acc / denom
                if cutlass.const_expr(mHeadGate is not None):
                    # Preserve the BF16 merge and sigmoid boundaries.
                    value = Float32(mOut.element_type(merged))
                    gate = Float32(mHeadGate[batch_idx, head_idx])
                    sigmoid = Float32(1.0) / (
                        Float32(1.0) + utils.exp2f(-gate * Float32(_LOG2E))
                    )
                    sigmoid = Float32(mHeadGate.element_type(sigmoid))
                    merged = value * sigmoid
                mOut[batch_idx, head_idx, dim] = mOut.element_type(merged)
            if tidx == Int32(0):
                lse = -Float32.inf
                if valid_sum:
                    lse = s_max + utils.log2f(denom) * Float32(_LN2)
                mLSE[batch_idx, head_idx] = lse

    return _merge_dynamic_split_nat_lse_states_kernel_gqa


def _make_launch_merge_variable_split_nat_lse_states_kernel_gqa(threads_per_block: int):
    @cute.jit
    def _launch_merge_variable_split_nat_lse_states_kernel_gqa(
        mPartialOut: cute.Tensor,
        mPartialLSE: cute.Tensor,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        batch: Int32,
        num_heads: int,
        head_dim: int,
        n_split4: Int32,
        n_split2: Int32,
        stream,
    ):
        _merge_variable_split_nat_lse_states_kernel_gqa(
            mPartialOut,
            mPartialLSE,
            mOut,
            mLSE,
            batch,
            Int32(num_heads),
            Int32(head_dim),
            n_split4,
            n_split2,
        ).launch(
            grid=[batch, num_heads, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )

    return _launch_merge_variable_split_nat_lse_states_kernel_gqa


@cached_compile_function
def _get_compiled_merge_variable_split_nat_lse_states_kernel_gqa(
    partial_out_signature: tuple[object, ...],
    partial_lse_signature: tuple[object, ...],
    out_signature: tuple[object, ...],
    lse_signature: tuple[object, ...],
    partial_out_align: int,
    partial_lse_align: int,
    out_align: int,
    lse_align: int,
    num_heads: int,
    head_dim: int,
    device_key: tuple[str, int, tuple[int, int]],
):
    device = utils.device_from_cache_key(device_key)
    launch_kernel = _make_launch_merge_variable_split_nat_lse_states_kernel_gqa(
        _threads_per_block_for_head_dim(head_dim)
    )
    partial_out = utils.placeholder_from_signature(
        partial_out_signature, device=device, dynamic_shape_fill=1
    )
    partial_lse = utils.placeholder_from_signature(
        partial_lse_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=1
    )
    lse = utils.placeholder_from_signature(
        lse_signature, device=device, dynamic_shape_fill=1
    )
    mPartialOut = utils.make_fake_tensor_like_with_dynamic_dim(
        partial_out, alignment=partial_out_align, dynamic_shape_dims=(0,)
    )
    mPartialLSE = utils.make_fake_tensor_like_with_dynamic_dim(
        partial_lse, alignment=partial_lse_align, dynamic_shape_dims=(0,)
    )
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=out_align, dynamic_shape_dims=(0,)
    )
    mLSE = utils.make_fake_tensor_like_with_dynamic_dim(
        lse, alignment=lse_align, dynamic_shape_dims=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _gqa_cute_compile(
        launch_kernel,
        mPartialOut,
        mPartialLSE,
        mOut,
        mLSE,
        Int32(1),
        int(num_heads),
        int(head_dim),
        Int32(0),
        Int32(0),
        stream_fake,
    )


def _make_launch_merge_dynamic_split_nat_lse_states_kernel_gqa(
    threads_per_block: int,
    mtp_q_len: int,
    fuse_head_gate: bool = False,
):
    merge_kernel = _make_merge_dynamic_split_nat_lse_states_kernel_gqa(
        mtp_q_len=int(mtp_q_len)
    )

    @cute.jit
    def _launch_merge_dynamic_split_nat_lse_states_kernel_gqa(
        mPartialOut: cute.Tensor,
        mPartialLSE: cute.Tensor,
        mRegionCounts: cute.Tensor,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        mHeadGate: Optional[cute.Tensor],
        batch: Int32,
        num_heads: int,
        head_dim: int,
        sm_count: int,
        stream,
    ):
        merge_kernel(
            mPartialOut,
            mPartialLSE,
            mRegionCounts,
            mOut,
            mLSE,
            mHeadGate,
            batch,
            Int32(num_heads),
            Int32(head_dim),
            Int32(sm_count),
        ).launch(
            grid=[batch, num_heads, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
            use_pdl=fuse_head_gate,
        )

    return _launch_merge_dynamic_split_nat_lse_states_kernel_gqa


@cached_compile_function
def _get_compiled_merge_dynamic_split_nat_lse_states_kernel_gqa(
    partial_out_signature: tuple[object, ...],
    partial_lse_signature: tuple[object, ...],
    region_counts_signature: tuple[object, ...],
    out_signature: tuple[object, ...],
    lse_signature: tuple[object, ...],
    partial_out_align: int,
    partial_lse_align: int,
    region_counts_align: int,
    out_align: int,
    lse_align: int,
    num_heads: int,
    head_dim: int,
    sm_count: int,
    mtp_q_len: int,
    device_key: tuple[str, int, tuple[int, int]],
    fuse_head_gate: bool = False,
):
    device = utils.device_from_cache_key(device_key)
    launch_kernel = _make_launch_merge_dynamic_split_nat_lse_states_kernel_gqa(
        _threads_per_block_for_head_dim(head_dim),
        int(mtp_q_len),
        fuse_head_gate,
    )
    partial_out = utils.placeholder_from_signature(
        partial_out_signature, device=device, dynamic_shape_fill=1
    )
    partial_lse = utils.placeholder_from_signature(
        partial_lse_signature, device=device, dynamic_shape_fill=1
    )
    region_counts = utils.placeholder_from_signature(
        region_counts_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=1
    )
    lse = utils.placeholder_from_signature(
        lse_signature, device=device, dynamic_shape_fill=1
    )
    mPartialOut = utils.make_fake_tensor_like_with_dynamic_dim(
        partial_out, alignment=partial_out_align, dynamic_shape_dims=(0,)
    )
    mPartialLSE = utils.make_fake_tensor_like_with_dynamic_dim(
        partial_lse, alignment=partial_lse_align, dynamic_shape_dims=(0,)
    )
    mRegionCounts = utils.make_fake_tensor_like_with_dynamic_dim(
        region_counts, alignment=region_counts_align, dynamic_shape_dims=(0,)
    )
    mOut = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=out_align, dynamic_shape_dims=(0,)
    )
    mLSE = utils.make_fake_tensor_like_with_dynamic_dim(
        lse, alignment=lse_align, dynamic_shape_dims=(0,)
    )
    mHeadGate = None
    if fuse_head_gate:
        gate = torch.empty((1, num_heads), dtype=out.dtype, device=device)
        mHeadGate = utils.make_fake_tensor_like_with_dynamic_dim(
            gate, alignment=2, dynamic_shape_dims=(0,)
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _gqa_cute_compile(
        launch_kernel,
        mPartialOut,
        mPartialLSE,
        mRegionCounts,
        mOut,
        mLSE,
        mHeadGate,
        Int32(1),
        int(num_heads),
        int(head_dim),
        int(sm_count),
        stream_fake,
    )


def _validate_pointer_alignment(
    tensor: torch.Tensor, *, name: str, min_align: int
) -> None:
    if int(tensor.numel()) == 0:
        return
    ptr = int(tensor.data_ptr())
    if ptr % int(min_align) != 0:
        raise ValueError(
            f"{name}.data_ptr()={ptr} is not {int(min_align)}-byte aligned; "
            "split-KV merge requires stable pointer alignment for precompiled "
            "CuTeDSL kernels"
        )


def merge_variable_split_nat_lse_states_sm90_gqa(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    n_split4: int,
    n_split2: int,
    stream: Optional[cuda.CUstream] = None,
) -> None:
    if stream is not None:
        raise ValueError("split-KV merge uses the TVM-FFI environment stream")
    if partial_out.device.type != "cuda" or partial_lse.device.type != "cuda":
        raise ValueError("partial_out/partial_lse must be CUDA tensors")
    if out.device != partial_out.device or lse.device != partial_out.device:
        raise ValueError("out/lse must be on the same CUDA device as partial_out")
    if partial_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"partial_out must be fp16/bf16/fp32, got {partial_out.dtype}")
    if out.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"out must be fp16/bf16, got {out.dtype}")
    if partial_out.dtype != torch.float32 and out.dtype != partial_out.dtype:
        raise TypeError(
            "out dtype must match a low-precision partial_out, got "
            f"{out.dtype} and {partial_out.dtype}"
        )
    if partial_lse.dtype != torch.float32 or lse.dtype != torch.float32:
        raise TypeError("partial_lse/lse must be float32")
    if not partial_out.is_contiguous() or not partial_lse.is_contiguous():
        raise ValueError("partial_out/partial_lse must be contiguous")
    if not out.is_contiguous() or not lse.is_contiguous():
        raise ValueError("out/lse must be contiguous")
    if partial_out.ndim != 3:
        raise ValueError(
            f"partial_out must have shape (work, H, D), got {tuple(partial_out.shape)}"
        )
    if partial_lse.ndim != 2:
        raise ValueError(
            f"partial_lse must have shape (work, H), got {tuple(partial_lse.shape)}"
        )
    work_count, num_heads, head_dim = [int(v) for v in partial_out.shape]
    batch = int(out.shape[0])
    n_split4 = int(n_split4)
    n_split2 = int(n_split2)
    if n_split4 < 0 or n_split2 < 0 or n_split4 + n_split2 > batch:
        raise ValueError(
            "invalid variable split plan: "
            f"batch={batch}, n_split4={n_split4}, n_split2={n_split2}"
        )
    expected_work_count = n_split4 * 4 + n_split2 * 2 + (batch - n_split4 - n_split2)
    if work_count != expected_work_count:
        raise ValueError(
            f"partial_out first dimension must be {expected_work_count}, got {work_count}"
        )
    if tuple(partial_lse.shape) != (work_count, num_heads):
        raise ValueError(
            f"partial_lse must have shape {(work_count, num_heads)}, got {tuple(partial_lse.shape)}"
        )
    if tuple(out.shape) != (batch, num_heads, head_dim):
        raise ValueError(
            f"out must have shape {(batch, num_heads, head_dim)}, got {tuple(out.shape)}"
        )
    if tuple(lse.shape) != (batch, num_heads):
        raise ValueError(
            f"lse must have shape {(batch, num_heads)}, got {tuple(lse.shape)}"
        )
    if head_dim <= 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim must be in (0, {_MAX_HEAD_DIM}], got {head_dim}")
    _validate_pointer_alignment(partial_out, name="partial_out", min_align=16)
    _validate_pointer_alignment(partial_lse, name="partial_lse", min_align=4)
    _validate_pointer_alignment(out, name="out", min_align=16)
    _validate_pointer_alignment(lse, name="lse", min_align=4)

    compiled = _get_compiled_merge_variable_split_nat_lse_states_kernel_gqa(
        utils.tensor_signature_dynamic(partial_out, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(partial_lse, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(out, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(lse, dynamic_shape_dims=(0,)),
        16,
        4,
        16,
        4,
        int(num_heads),
        int(head_dim),
        utils.device_cache_key(partial_out.device),
    )
    compiled(
        partial_out,
        partial_lse,
        out,
        lse,
        int(batch),
        int(num_heads),
        int(head_dim),
        int(n_split4),
        int(n_split2),
    )


def merge_dynamic_split_nat_lse_states_sm90_gqa(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    region_counts: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    mtp_q_len: int = 1,
    stream: Optional[cuda.CUstream] = None,
    head_gate: Optional[torch.Tensor] = None,
) -> None:
    if stream is not None:
        raise ValueError("dynamic split-KV merge uses the TVM-FFI environment stream")
    tensors = (partial_out, partial_lse, region_counts, out, lse)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("dynamic split-KV merge requires CUDA tensors")
    if any(tensor.device != partial_out.device for tensor in tensors[1:]):
        raise ValueError("dynamic split-KV merge tensors must share one device")
    if partial_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"partial_out must be fp16/bf16/fp32, got {partial_out.dtype}")
    if out.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"out must be fp16/bf16, got {out.dtype}")
    if partial_out.dtype != torch.float32 and out.dtype != partial_out.dtype:
        raise TypeError("out dtype must match a low-precision partial_out")
    if partial_lse.dtype != torch.float32 or lse.dtype != torch.float32:
        raise TypeError("partial_lse/lse must be float32")
    if region_counts.dtype != torch.int32:
        raise TypeError("region_counts must be int32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("dynamic split-KV merge tensors must be contiguous")
    if partial_out.ndim != 3 or partial_lse.ndim != 2:
        raise ValueError("partial_out/partial_lse must have ranks 3 and 2")
    if out.ndim != 3 or lse.ndim != 2 or region_counts.ndim != 1:
        raise ValueError("out/lse/region_counts must have ranks 3, 2, and 1")

    work_count, num_heads, head_dim = [int(value) for value in partial_out.shape]
    batch = int(out.shape[0])
    mtp_q_len = int(mtp_q_len)
    if mtp_q_len < 1 or mtp_q_len > 16 or batch % mtp_q_len != 0:
        raise ValueError(
            "dynamic split merge requires batch divisible by mtp_q_len in [1, 16], "
            f"got batch={batch}, mtp_q_len={mtp_q_len}"
        )
    expected_work_count = batch * _DYNAMIC_SPLIT_CAPACITY
    if work_count != expected_work_count:
        raise ValueError(
            f"partial work capacity must be {expected_work_count}, got {work_count}"
        )
    if tuple(partial_lse.shape) != (work_count, num_heads):
        raise ValueError("partial_lse shape does not match partial_out")
    if tuple(region_counts.shape) != (batch,):
        raise ValueError(f"region_counts must have shape {(batch,)}")
    if tuple(out.shape) != (batch, num_heads, head_dim):
        raise ValueError("out shape does not match partial_out")
    if tuple(lse.shape) != (batch, num_heads):
        raise ValueError("lse shape does not match partial_out")
    if head_dim <= 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim must be in (0, {_MAX_HEAD_DIM}], got {head_dim}")
    _validate_pointer_alignment(partial_out, name="partial_out", min_align=16)
    _validate_pointer_alignment(partial_lse, name="partial_lse", min_align=4)
    _validate_pointer_alignment(region_counts, name="region_counts", min_align=4)
    _validate_pointer_alignment(out, name="out", min_align=16)
    _validate_pointer_alignment(lse, name="lse", min_align=4)

    if head_gate is not None:
        if (
            (batch, num_heads, head_dim, mtp_q_len) != (4, 16, 192, 4)
            or partial_out.dtype != torch.float32
            or out.dtype != torch.bfloat16
            or head_gate.dtype != out.dtype
            or head_gate.device != out.device
            or head_gate.shape != (batch, num_heads)
            or not head_gate.is_contiguous()
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(out.device) != (10, 3)
        ):
            raise ValueError("fused head gate requires SM103 BF16 M4/H16/D192")
        _validate_pointer_alignment(head_gate, name="head_gate", min_align=2)

    sm_count = torch.cuda.get_device_properties(
        partial_out.device
    ).multi_processor_count
    compiled = _get_compiled_merge_dynamic_split_nat_lse_states_kernel_gqa(
        utils.tensor_signature_dynamic(partial_out, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(partial_lse, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(region_counts, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(out, dynamic_shape_dims=(0,)),
        utils.tensor_signature_dynamic(lse, dynamic_shape_dims=(0,)),
        16,
        4,
        4,
        16,
        4,
        num_heads,
        head_dim,
        int(sm_count),
        mtp_q_len,
        utils.device_cache_key(partial_out.device),
        head_gate is not None,
    )
    compiled(
        partial_out,
        partial_lse,
        region_counts,
        out,
        lse,
        head_gate,
        batch,
        num_heads,
        head_dim,
        int(sm_count),
    )


__all__ = [
    "merge_dynamic_split_nat_lse_states_sm90_gqa",
    "merge_variable_split_nat_lse_states_sm90_gqa",
]
