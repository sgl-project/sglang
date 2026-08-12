# SPDX-License-Identifier: Apache-2.0

"""System-scope publication and reuse markers for fixed-width DP metadata."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_and_publish(dst_rows, dst_markers, src, generation):
    peer = tl.program_id(0)
    dst_row = tl.load(dst_rows + peer)
    dst_marker = tl.load(dst_markers + peer)
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            .reg .u32 tid;
            .reg .u64 value;
            mov.u32 tid, %tid.x;
            setp.ne.u32 p, tid, 0;
            @p bra done;
            ld.relaxed.gpu.global.u64 value, [$1 + 0];
            st.relaxed.sys.global.u64 [$2 + 0], value;
            ld.relaxed.gpu.global.u64 value, [$1 + 8];
            st.relaxed.sys.global.u64 [$2 + 8], value;
            ld.relaxed.gpu.global.u64 value, [$1 + 16];
            st.relaxed.sys.global.u64 [$2 + 16], value;
            ld.relaxed.gpu.global.u64 value, [$1 + 24];
            st.relaxed.sys.global.u64 [$2 + 24], value;
            ld.relaxed.gpu.global.u64 value, [$1 + 32];
            st.relaxed.sys.global.u64 [$2 + 32], value;
            ld.relaxed.gpu.global.u64 value, [$1 + 40];
            st.relaxed.sys.global.u64 [$2 + 40], value;
            ld.relaxed.gpu.global.u64 value, [$1 + 48];
            st.relaxed.sys.global.u64 [$2 + 48], value;
            fence.proxy.alias;
            atom.global.release.sys.exch.b32 tid, [$3], $4;
            done:
            mov.u32 $0, 0;
        }
        """,
        "=r,l,l,l,r",
        args=[src, dst_row, dst_marker, generation],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def copy_row_and_publish(
    dst_rows: torch.Tensor,
    dst_markers: torch.Tensor,
    src: torch.Tensor,
    generation: int,
):
    _copy_row_and_publish[(dst_rows.numel(),)](
        dst_rows, dst_markers, src, generation, num_warps=1
    )


@triton.jit
def _publish_value(dst_ptrs, value):
    peer = tl.program_id(0)
    dst = tl.load(dst_ptrs + peer)
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            .reg .u32 tid;
            mov.u32 tid, %tid.x;
            setp.ne.u32 p, tid, 0;
            @p bra done;
            atom.global.release.sys.exch.b32 tid, [$1], $2;
            done:
            mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        args=[dst, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def publish_value(dst_ptrs: torch.Tensor, value: int):
    _publish_value[(dst_ptrs.numel(),)](dst_ptrs, value, num_warps=1)


@triton.jit
def _load_acquire(ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            setp.eq.s32 p, $2, 1;
            mov.u32 $0, 0;
            @p atom.global.acquire.sys.cas.b32 $0, [$1], 0xffffffff, 0xffffffff;
        }
        """,
        "=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_relaxed_sys(ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            setp.eq.s32 p, $2, 1;
            mov.u64 $0, 0;
            @p fence.proxy.alias;
            @p ld.relaxed.sys.global.u64 $0, [$1];
        }
        """,
        "=l,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _snapshot_rows_acquire(
    src_rows,
    markers,
    dst_rows,
    ready,
    generation,
    world_size: tl.constexpr,
    numel: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    active = offsets < numel
    complete = active
    for peer in range(world_size):
        complete &= _load_acquire(markers + peer, active) == generation
    values = _load_relaxed_sys(src_rows + offsets, complete)
    tl.store(dst_rows + offsets, values, mask=complete)
    tl.store(ready + offsets, complete.to(tl.uint32), mask=offsets == 0)


def snapshot_rows_acquire(
    src_rows: torch.Tensor,
    markers: torch.Tensor,
    dst_rows: torch.Tensor,
    ready: torch.Tensor,
    generation: int,
):
    _snapshot_rows_acquire[(1,)](
        src_rows,
        markers,
        dst_rows,
        ready,
        generation,
        markers.numel(),
        src_rows.numel(),
        triton.next_power_of_2(src_rows.numel()),
        num_warps=4,
    )


@triton.jit
def _all_values_acquire(
    src,
    ready,
    expected,
    numel: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    active = offsets < numel
    values = _load_acquire(src + offsets, active)
    complete = tl.sum((values == expected).to(tl.int32), axis=0) == numel
    tl.store(ready, complete.to(tl.uint32))


def all_values_acquire(src: torch.Tensor, ready: torch.Tensor, expected: int):
    _all_values_acquire[(1,)](
        src,
        ready,
        expected,
        src.numel(),
        triton.next_power_of_2(src.numel()),
        num_warps=1,
    )
