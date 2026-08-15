# SPDX-License-Identifier: Apache-2.0

"""System-scope publication markers for fixed-width DP metadata."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_and_publish(dst_rows, dst_markers, src, generation):
    peer = tl.program_id(0)
    dst_row = tl.load(dst_rows + peer).to(tl.int64)
    dst_marker = tl.load(dst_markers + peer).to(tl.int64)
    src_addr = src.to(tl.int64)
    gen64 = tl.full([1], generation, tl.int64)
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred p;
            .reg .u32 tid, gen32;
            .reg .u64 value;
            mov.u32 tid, %tid.x;
            setp.ne.u32 p, tid, 0;
            @p bra done;
            cvt.u32.u64 gen32, $4;
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
            atom.global.release.sys.exch.b32 tid, [$3], gen32;
            done:
            mov.u32 $0, 0;
        }
        """,
        "=r,l,l,l,l",
        args=[src_addr, dst_row, dst_marker, gen64],
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
def _snapshot_row_acquire(src_rows, markers, dst_rows, ready, generation):
    peer = tl.program_id(0)
    src = (src_rows + peer * 7).to(tl.int64)
    marker = (markers + peer).to(tl.int64)
    dst = (dst_rows + peer * 7).to(tl.int64)
    peer_ready = (ready + peer).to(tl.int64)
    gen64 = tl.full([1], generation, tl.int64)
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred p, incomplete;
            .reg .u32 tid, observed, is_ready, gen32;
            .reg .u64 value;
            mov.u32 tid, %tid.x;
            setp.ne.u32 p, tid, 0;
            @p bra done;
            mov.u32 is_ready, 0;
            cvt.u32.u64 gen32, $5;
            atom.global.acquire.sys.cas.b32 observed, [$2], 0xffffffff, 0xffffffff;
            setp.ne.u32 incomplete, observed, gen32;
            @incomplete bra publish;
            fence.proxy.alias;
            ld.relaxed.sys.global.u64 value, [$1 + 0];
            st.relaxed.gpu.global.u64 [$3 + 0], value;
            ld.relaxed.sys.global.u64 value, [$1 + 8];
            st.relaxed.gpu.global.u64 [$3 + 8], value;
            ld.relaxed.sys.global.u64 value, [$1 + 16];
            st.relaxed.gpu.global.u64 [$3 + 16], value;
            ld.relaxed.sys.global.u64 value, [$1 + 24];
            st.relaxed.gpu.global.u64 [$3 + 24], value;
            ld.relaxed.sys.global.u64 value, [$1 + 32];
            st.relaxed.gpu.global.u64 [$3 + 32], value;
            ld.relaxed.sys.global.u64 value, [$1 + 40];
            st.relaxed.gpu.global.u64 [$3 + 40], value;
            ld.relaxed.sys.global.u64 value, [$1 + 48];
            st.relaxed.gpu.global.u64 [$3 + 48], value;
            mov.u32 is_ready, 1;
            publish:
            st.relaxed.gpu.global.u32 [$4], is_ready;
            done:
            mov.u32 $0, 0;
        }
        """,
        "=r,l,l,l,l,l",
        args=[src, marker, dst, peer_ready, gen64],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def snapshot_rows_acquire(
    src_rows: torch.Tensor,
    markers: torch.Tensor,
    dst_rows: torch.Tensor,
    ready: torch.Tensor,
    generation: int,
):
    _snapshot_row_acquire[(markers.numel(),)](
        src_rows,
        markers,
        dst_rows,
        ready,
        generation,
        num_warps=1,
    )
