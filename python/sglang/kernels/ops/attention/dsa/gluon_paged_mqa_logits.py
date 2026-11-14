# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 paged MQA FP8 logits for the DSA indexer (decode / MTP verify rows).

Derived from aiter's gluon ``_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle``
(aiter/ops/triton/gluon/pa_mqa_logits.py, aiter e7d2453) and its host wrapper,
reduced to the path sglang takes here (gfx950, preshuffled 64-token pages,
32 heads x 128, one query per row) and re-tuned:

- the KV split is decoupled from the occupancy hint and set per row count, and
  program ids are swizzled so that the pieces of one row share an XCD (L2);
- MTP verify rows of one request (6 consecutive rows) are dispatched longest
  group first;
- ``_gluon_paged_mqa_logits_fused2`` scores two sibling rows per program when
  their page tables match (checked on the device, so any input is exact);
- gfx950's full-rate scaled fp8 MFMA, a uniform-SGPR row index, and ``relu``
  as one ``v_maximum3``.

Only ``paged_mqa_logits.aiter_paged_mqa_logits`` calls into this module, and
only when :func:`supported` holds; every other shape keeps aiter's kernel.
"""

from __future__ import annotations

from functools import cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@cache
def _gfx() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def get_cdna_version() -> int:
    """4 on gfx950 (CDNA4), 3 on gfx942 (CDNA3), -1 otherwise."""
    gfx = _gfx()
    return 4 if gfx.startswith("gfx95") else (3 if gfx.startswith("gfx94") else -1)


def get_gfx() -> str:
    return _gfx()


@cache
def get_num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


# Number of XCDs the phase-1 grid is round-robined over (gfx950 SPX = 8).
# Used to relabel program ids so that the KV-split pieces that share page-table
# entries land on the same XCD and hit its private L2 instead of the fabric.
_NUM_XCD = tl.constexpr(8)

try:
    from triton.experimental.gluon.language.amd.cdna3 import (
        s_set_prio as _amd_s_set_prio,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_group_barrier as _amd_iglp_sched_group_barrier,
    )
except ImportError:
    # ignore iglp hint
    @gluon.jit
    def _amd_iglp_sched_barrier(inst_mask):
        pass

    @gluon.jit
    def _amd_iglp_sched_group_barrier(inst_mask, cnt, _):
        pass

    @gluon.jit
    def _amd_s_set_prio(prio):
        pass


# for some newer triton>=3.5 version, a 3D instr_shape is required.
try:
    _cdna_version = get_cdna_version()
    _version = _cdna_version if _cdna_version > 0 else 3
    _: gl.constexpr = gl.amd.AMDMFMALayout(
        version=_version,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, 1],
        tiles_per_warp=[1, 1],
    )
    _Use_2d_instr_shape_mfma_layout = tl.constexpr(True)
except Exception:  # noqa: BLE001
    _Use_2d_instr_shape_mfma_layout = tl.constexpr(False)


@triton.jit
def _sum_combine(a, b):
    return a + b


@gluon.jit
def _uniform_i32(x):
    # v_readfirstlane into an SGPR ("=s" output): the value is then provably uniform, so
    # buffer descriptors / loop bounds derived from it stay scalar (no waterfall loops).
    # The s_nops cover the VALU/DPP-write -> readfirstlane-read and the VALU-SGPR-write ->
    # use hazards, which the compiler cannot see through inline asm (without the leading
    # nop the read returned a stale lane value, measured).
    return gl.inline_asm_elementwise(
        "s_nop 4\n v_readfirstlane_b32 $0, $1\n s_nop 4",
        "=s,v",
        [x],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _dot_fp8(a, b, acc, CDNA_VERSION: gl.constexpr):
    # gfx950: the full-rate v_mfma_*_f8f6f4 instructions are only reachable through the
    # scaled-MFMA op (plain tt.dot has no fp8 mapping for K=64/128 shapes); unit e8m0
    # scales (0x7F == 2^0) make it a plain fp8 dot. The legacy v_mfma_f32_16x16x32_fp8
    # runs at half of gfx950's fp8 rate (16 cycles each, measured).
    # Epilogue relu: gl.maximum(.., propagate_nan=ALL) lowers to one v_maximum3_f32 per
    # element; the default maxnum form costs a canonicalize v_max plus the v_max.
    # Bitwise identical for finite inputs; a NaN MFMA output (only from fp8 NaN codes,
    # which the ue8m0 cache writer does not produce from finite K) now propagates
    # instead of being clipped to 0.
    if CDNA_VERSION == 4:
        out = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
    else:
        out = gl.amd.cdna3.mfma(a, b, acc)
    return out


# ===========================================================================
# Device side: the preshuffle kernel (aiter e7d2453 + the changes listed in the module docstring)
# ===========================================================================
@gluon.jit
def _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    num_block,
    SplitKV: gl.constexpr,
    ChunkQ: gl.constexpr,
    ChunkK: gl.constexpr,
    HiddenDim: gl.constexpr,
    KVBlockSize: gl.constexpr = 16,
    CDNA_VERSION: gl.constexpr = 3,
    ARCH: gl.constexpr = "gfx942",
    LPT_GROUP: gl.constexpr = 0,
    NG2: gl.constexpr = 16,
):
    IS_GFX1250: gl.constexpr = ARCH == "gfx1250"
    # ===---------------------------------------------------
    # Gluon Layout
    # ===---------------------------------------------------
    if IS_GFX1250:
        NumWarps: gl.constexpr = 1
    else:
        NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 32 if IS_GFX1250 else 64

    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],  # q type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    if IS_GFX1250:
        ChunkKPerStage: gl.constexpr = 16
    else:
        ChunkKPerStage: gl.constexpr = ChunkK // 2
    MFMAPerWarp: gl.constexpr = ChunkKPerStage // 16 // NumWarps

    if IS_GFX1250:
        FP8_K_DIM: gl.constexpr = 128 if HiddenDim > 64 else 64
        if NumWarps == 1:
            warp_bases: gl.constexpr = []
        elif NumWarps == 2:
            warp_bases: gl.constexpr = [[0, 1]]
        elif NumWarps == 4:
            warp_bases: gl.constexpr = [[0, 1], [0, 2]]
        else:
            warp_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
        mfma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
            version=3,
            transposed=False,
            instr_shape=[16, 16, FP8_K_DIM],
            warp_bases=warp_bases,
        )
    elif _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )
    else:
        # gfx950, 32 heads: one 32x32x64 f8f6f4 tile per warp (32 heads x 32 tokens,
        # 2 MFMAs per stage). The head sum is then in-thread plus one permlane32 swap
        # (16x16 tiles need permlane32 + permlane16 rounds), and each lane stores one
        # logit. K addresses / weight / scale / store layouts all derive from
        # mfma_layout, so the preshuffled cache is read unchanged (4 x dwordx4 per lane
        # per stage). The head-sum order changes, so logits differ at rounding level;
        # the selected top-k sets are unchanged. Other shapes: 16x16x128.
        if CDNA_VERSION == 4 and ChunkQ == 32 and ChunkKPerStage == 32 * NumWarps:
            mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
                version=CDNA_VERSION,
                instr_shape=[32, 32, 64],
                transposed=False,
                warps_per_cta=[1, NumWarps],
                tiles_per_warp=[1, 1],
            )
        else:
            mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
                version=CDNA_VERSION,
                instr_shape=[16, 16, 128] if CDNA_VERSION == 4 else [16, 16, 32],
                transposed=False,
                warps_per_cta=[1, NumWarps],
                tiles_per_warp=[1, MFMAPerWarp],
            )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    if IS_GFX1250:
        NUM_BUFFERS: gl.constexpr = 2
        KV_SHARED: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )
        kv_shared = gl.allocate_shared_memory(
            KV_buffer.type.element_ty,
            [NUM_BUFFERS, 1, KVBlockSize * HiddenDim],
            KV_SHARED,
        )
        kv_scale_shared = gl.allocate_shared_memory(
            scale_buffer.type.element_ty,
            [NUM_BUFFERS, 1, KVBlockSize * HiddenDim // 128],
            KV_SHARED,
        )

        pid = tl.program_id(0)
        pid_batch, remain_pid = pid % batch_size, pid // batch_size
        pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n
        context_length = gl.load(context_len_ptr + pid_batch)

        context_chunk_num = tl.cdiv(context_length, ChunkK)
        split_context_chunk_num = context_chunk_num // SplitKV
        residual_context_chunks = context_chunk_num % SplitKV
        split_context_start = (
            pid_split_kv * split_context_chunk_num * ChunkK
            + min(pid_split_kv, residual_context_chunks) * ChunkK
        )
        split_context_length = min(
            context_length - split_context_start,
            split_context_chunk_num * ChunkK
            + (ChunkK if pid_split_kv < residual_context_chunks else 0),
        )
        if split_context_length <= 0:
            return

        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )
        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        mfma_q = gl.convert_layout(q, mfma_layout_a)
        zero_acc = gl.zeros((ChunkQ, KVBlockSize), dtype=tl.float32, layout=mfma_layout)
        col = gl.arange(0, KVBlockSize, layout=gl.SliceLayout(0, mfma_layout))

        n_blocks = tl.cdiv(split_context_length, KVBlockSize)
        cblk_base = split_context_start // KVBlockSize
        kv_idx_base = kv_indices + pid_batch * max_block_len + cblk_base

        # Prologue: kick off the async_load for block 0 into buffer 0.
        save_blk_idx: gl.int32 = n_blocks - 1
        blk_cur = gl.load(kv_idx_base)
        blk_next = gl.load(kv_idx_base + gl.minimum(1, save_blk_idx))
        desc_cur = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=KV_buffer,
            shape=(num_block, KVBlockSize * HiddenDim),
            strides=(stride_k_seq, 1),
            block_shape=(1, KVBlockSize * HiddenDim),
            layout=KV_SHARED,
        )
        BLOCKSCALE_SIZE: gl.constexpr = 128
        assert HiddenDim // BLOCKSCALE_SIZE == 1
        desc_scale_cur = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=scale_buffer,
            shape=(num_block, KVBlockSize * HiddenDim // BLOCKSCALE_SIZE),
            strides=(stride_scale_seq, 1),
            block_shape=(1, KVBlockSize * HiddenDim // BLOCKSCALE_SIZE),
            layout=KV_SHARED,
        )
        gl.amd.gfx1250.tdm.async_load(desc_cur, [blk_cur, 0], kv_shared.index(0))
        gl.amd.gfx1250.tdm.async_load(
            desc_scale_cur, [blk_cur, 0], kv_scale_shared.index(0)
        )

        for j in range(n_blocks):
            buf = j % NUM_BUFFERS
            context_idx = split_context_start + j * KVBlockSize
            # blk = blk_cur
            # k_scale_f = gl.amd.cdna3.buffer_load(
            #     ptr=scale_buffer,
            #     offsets=blk * stride_scale_seq + col,
            # )

            # Prefetch block j+1 into the other buffer before waiting on block j.
            if j + 1 < n_blocks:
                blk_cur = blk_next
                blk_next = gl.load(kv_idx_base + gl.minimum(j + 2, save_blk_idx))
                gl.amd.gfx1250.tdm.async_load(
                    desc_cur, [blk_cur, 0], kv_shared.index((j + 1) % NUM_BUFFERS)
                )
                gl.amd.gfx1250.tdm.async_load(
                    desc_scale_cur,
                    [blk_cur, 0],
                    kv_scale_shared.index((j + 1) % NUM_BUFFERS),
                )
                # Leave exactly the just-issued load (block j+1) in flight.
                gl.amd.gfx1250.tdm.async_wait(3)
            else:
                gl.amd.gfx1250.tdm.async_wait(1)

            K_WIDTH: gl.constexpr = 16
            mfma_k = (
                kv_shared.index(buf)
                .reshape(
                    (
                        KVBlockSize // 16,
                        HiddenDim // (2 * K_WIDTH),
                        2,
                        16,
                        K_WIDTH,
                    )
                )
                .permute((0, 3, 1, 2, 4))
                .reshape((KVBlockSize, HiddenDim))
                .permute([1, 0])
                .load(layout=mfma_layout_b)
            )
            o = gl.amd.gfx1250.wmma(mfma_q, mfma_k, zero_acc)

            if j + 1 < n_blocks:
                gl.amd.gfx1250.tdm.async_wait(2)
            else:
                gl.amd.gfx1250.tdm.async_wait(0)

            k_scale_f = (
                kv_scale_shared.index(buf)
                .reshape((1, KVBlockSize))
                .load(layout=mfma_layout)
            )
            o = o * k_scale_f
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]

            valid = (context_idx + col) <= (context_length - next_n + pid_next_n)
            o = tl.where(valid[None, :], o, float("-inf"))
            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)  # gfx1250 path

            store_off = context_idx + col
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=store_off,
                mask=(store_off >= 0) & (store_off < max_model_len),
            )
        return

    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize
    ChunkKStagePerContextBlock: gl.constexpr = KVBlockSize // ChunkKPerStage

    LoadBlockIndiceForEachStage: gl.constexpr = ChunkKPerStage % KVBlockSize == 0
    # Rows per shuffled group, matching the `layout=(ShuffleRows, 16)` the host
    # passed to shuffle_weight. A page shorter than the 16-token MFMA tile can
    # only be shuffled in groups of its own length, so the tile is then read
    # from 16 // KVBlockSize pages -- which the per-lane page index already does.
    ShuffleRows: gl.constexpr = min(KVBlockSize, 16)

    # DS_WRITE: gl.constexpr = 0x200
    DS_READ: gl.constexpr = 0x100
    BUFFER_LOAD: gl.constexpr = 0x020
    MFMA: gl.constexpr = 0x008
    # VALU: gl.constexpr = 0x002

    # ===---------------------------------------------------
    # Mapping WorkTile
    # ===---------------------------------------------------
    pid = tl.program_id(0)

    # XCD-local swizzle: the hardware round-robins workgroup i onto XCD i % NUM_XCD,
    # so the rows that share a page table (the 6 TARGET_VERIFY siblings of a request)
    # get scattered over 6 private L2s. Relabelling the program id so that each XCD
    # owns a contiguous slab of (split_kv-major, row-minor) work puts every row of a
    # given KV split on one XCD. This is a bijection over the whole grid, so the
    # logits are bitwise identical.
    NUM_XCD: gl.constexpr = _NUM_XCD
    if LPT_GROUP > 0:
        # Longest-group-first dispatch inside each XCD slab (host-enabled only when
        # next_n == 1, rows % LPT_GROUP == 0, rows // LPT_GROUP <= NG2 and
        # SplitKV % NUM_XCD == 0). XCD x still owns splits [x*SX, (x+1)*SX) of every
        # row (same per-XCD work, same L2 locality); only the order inside the XCD
        # changes: (group rank q, split t, sibling s), rank 0 = longest group, ties by
        # index. Every (row, split) is still computed once with the same code, so the
        # logits are bitwise identical. Each warp ranks the groups redundantly (the
        # 16x16 tile is replicated across warps: every reduction stays in-warp), and
        # the result is made scalar with readfirstlane.
        SX = SplitKV // NUM_XCD
        GS = LPT_GROUP * SX
        x_id = pid % NUM_XCD
        j = pid // NUM_XCD
        q_rank = j // GS
        t_split = (j % GS) // LPT_GROUP
        s_sib = j % LPT_GROUP
        ngroups = batch_size // LPT_GROUP
        L_RANK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, NumWarps], [1, 0])
        gi = gl.arange(0, NG2, layout=gl.SliceLayout(1, L_RANK))
        gj = gl.arange(0, NG2, layout=gl.SliceLayout(0, L_RANK))
        Li = gl.load(context_len_ptr + gi * LPT_GROUP, mask=gi < ngroups, other=-1)
        Lj = gl.load(context_len_ptr + gj * LPT_GROUP, mask=gj < ngroups, other=-1)
        ahead = (Lj[None, :] > Li[:, None]) | (
            (Lj[None, :] == Li[:, None]) & (gj[None, :] < gi[:, None])
        )
        rank = gl.sum(ahead.to(gl.int32), axis=1)
        g_sel = gl.sum(gl.where((rank == q_rank) & (gi < ngroups), gi, 0), axis=0)
        g_sel = _uniform_i32(g_sel)
        pid_batch = g_sel * LPT_GROUP + s_sib
        pid_next_n = 0
        pid_split_kv = x_id * SX + t_split
    else:
        if NUM_XCD > 1:
            grid_size = batch_size * next_n * SplitKV
            wg_per_xcd = grid_size // NUM_XCD
            if wg_per_xcd * NUM_XCD == grid_size:
                pid = (pid % NUM_XCD) * wg_per_xcd + pid // NUM_XCD

        # ===---------------------------------------------------
        pid_batch, remain_pid = pid % batch_size, pid // batch_size
        pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n
    # ===---------------------------------------------------
    context_length = gl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = context_chunk_num // SplitKV
    residual_context_chunks = context_chunk_num % SplitKV
    split_context_start = (
        pid_split_kv * split_context_chunk_num * ChunkK
        + min(pid_split_kv, residual_context_chunks) * ChunkK
    )
    split_context_length = min(
        context_length - split_context_start,
        split_context_chunk_num * ChunkK
        + (ChunkK if pid_split_kv < residual_context_chunks else 0),
    )

    if split_context_length <= 0:
        return

    if LoadBlockIndiceForEachStage:
        split_context_block = tl.cdiv(split_context_length, KVBlockSize)
        split_context_length = split_context_block * KVBlockSize

        residual_context_blocks = (
            ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
        ) % ContextBlockPerChunkK
        residual_context = residual_context_blocks * KVBlockSize

        # ===---------------------------------------------------
        # Pipeline Start
        _amd_iglp_sched_barrier(0x0)
        # ===---------------------------------------------------
        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_idx = split_context_start - residual_context

        mask_kv_next_0 = (
            context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_0,
        )

        mask_kv_next_1 = (
            (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_1,
        )

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * (ShuffleRows * 16)
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % ShuffleRows
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // ShuffleRows
            * ShuffleRows
            * HiddenDim
        )[None, :]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        context_kv_idx_next_0 = tl.where(mask_kv_next_0, context_kv_idx_next_0, 0)
        k_next_0 = gl.load(
            KV_buffer
            + offset_k_fixed
            + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
        )
        k_scale_f_next_0 = gl.load(
            scale_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        if context_idx + ChunkK < split_context_start + split_context_length:
            context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        context_kv_idx_next_1 = tl.where(mask_kv_next_1, context_kv_idx_next_1, 0)
        k_next_1 = gl.load(
            KV_buffer
            + offset_k_fixed
            + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
        )
        k_scale_f_next_1 = gl.load(
            scale_buffer
            + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = (
            gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
            * scale_weight[:, None]
        )
        _amd_s_set_prio(1)

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        store_cols = context_idx + gl.arange(
            0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
        )
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=store_cols,
            mask=(store_cols >= split_context_start) & (store_cols < max_model_len),
        )

        for context_idx in range(
            split_context_start - residual_context,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK + ChunkKPerStage) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
            k_next_0 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_0 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = (
                gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
                * scale_weight[:, None]
            )
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            store_cols = (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            )
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=store_cols,
                mask=(store_cols >= split_context_start) & (store_cols < max_model_len),
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            # #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            # #!=----------------------------
            if (
                context_idx + ChunkK + ChunkK
                < split_context_start + split_context_length
            ):
                context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=pid_batch * max_block_len
                    + (context_idx + ChunkK + ChunkK) // KVBlockSize
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                    // KVBlockSize,
                )
            k_next_1 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_1 = gl.load(
                scale_buffer
                + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )
            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = (
                gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
                * scale_weight[:, None]
            )
            _amd_s_set_prio(0)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=(
                    context_idx
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=(
                    context_idx
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                )
                < max_model_len,
            )

        context_idx = split_context_start + split_context_length - ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = (
            gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
            * scale_weight[:, None]
        )
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        logits = tl.where(mask, logits, float("-inf"))
        store_cols = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
        )
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=store_cols,
            mask=(store_cols >= split_context_start) & (store_cols < max_model_len),
        )
    else:
        context_idx = split_context_start
        current_chunk_rank = context_idx // ChunkKPerStage % ChunkKStagePerContextBlock
        block_idx = context_idx // KVBlockSize
        batch_blocks = tl.cdiv(context_length, KVBlockSize)

        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_kv_idx_next_0 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx
        )
        block_idx += 1

        context_kv_idx_next_1 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx,
            mask=block_idx < batch_blocks,
        )
        block_idx += 1

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * (ShuffleRows * 16)
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % ShuffleRows
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // ShuffleRows
            * ShuffleRows
            * HiddenDim
        )[None, :]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        k_next_0 = gl.load(
            KV_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
            + offset_k_fixed
            + context_idx % KVBlockSize * HiddenDim
        )
        k_scale_f_next_0 = gl.load(
            scale_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        k_next_1 = gl.load(
            KV_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
            + offset_k_fixed
            + (context_idx + ChunkKPerStage) % KVBlockSize * HiddenDim
        )
        k_scale_f_next_1 = gl.load(
            scale_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )
        current_chunk_rank += 2

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = (
            gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
            * scale_weight[:, None]
        )
        _amd_s_set_prio(1)

        mask = (
            context_idx
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=(
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=(
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            )
            < max_model_len,
        )

        for context_idx_ in range(
            split_context_start,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            if current_chunk_rank == ChunkKStagePerContextBlock:
                current_chunk_rank = 0
                context_kv_idx_next_0 = context_kv_idx_next_1
                context_kv_idx_next_1 = gl.load(
                    kv_indices + pid_batch * max_block_len + block_idx,
                    mask=block_idx < batch_blocks,
                )
                block_idx += 1

            k_next_0 = gl.load(
                KV_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
                + offset_k_fixed
                + (context_idx_ + ChunkK) % KVBlockSize * HiddenDim
            )
            k_scale_f_next_0 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = (
                gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
                * scale_weight[:, None]
            )
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=(
                    context_idx_
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=(
                    context_idx_
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                )
                < max_model_len,
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_next_1 = gl.load(
                KV_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
                + offset_k_fixed
                + (context_idx_ + ChunkK + ChunkKPerStage) % KVBlockSize * HiddenDim
            )
            k_scale_f_next_1 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize
            )
            current_chunk_rank += 2

            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = (
                gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
                * scale_weight[:, None]
            )
            _amd_s_set_prio(0)

            mask = (
                context_idx_
                + ChunkK
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
                <= context_length - next_n + pid_next_n
            )

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            logits = tl.where(mask, logits, float("-inf"))
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=(
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=(
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                )
                < max_model_len,
            )
            context_idx = context_idx_ + ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = (
            gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL)
            * scale_weight[:, None]
        )
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=(
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=(
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            )
            < max_model_len,
        )


# ===========================================================================
# Sibling-fused phase 1 (gfx950, preshuffle, page 64, 32 heads x 128, next_n == 1).
# The TARGET_VERIFY rows of one request (GROUP consecutive rows, lengths L+1..L+GROUP)
# share one page table, so every K page is read by GROUP rows. Here one workgroup
# computes a PAIR of sibling rows: each K / scale register tile is loaded once and fed
# to two MFMA pairs (one per row's Q). Same per-token arithmetic as the single-row
# kernel (same MFMA tile, same epilogue / head-sum order), so the logits over [0, L)
# are bitwise identical to it; only the KV partition is taken from the longer row.
# The pairing is verified at run time: the two rows' page-table entries over this
# workgroup's pages (below the shorter row's length) must match, else the workgroup
# runs two single-row passes instead (each row with its own pages and partition).
# Dispatch order inside each XCD slab: longest group first (see LPT in the kernel above).
# ===========================================================================
@gluon.jit
def _gluon_paged_mqa_logits_fused2(
    batch_size,
    Q_buffer,
    stride_q_batch,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    SplitKV,
    ChunkQ: gl.constexpr,
    ChunkK: gl.constexpr,
    KVBlockSize: gl.constexpr,
    HiddenDim: gl.constexpr,
    GROUP: gl.constexpr,
    NG2: gl.constexpr,
    NUM_XCD: gl.constexpr,
):
    NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 64
    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )
    ChunkKPerStage: gl.constexpr = ChunkK // 2
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NumWarps],
        tiles_per_warp=[1, 1],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    layout_cols: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    layout_kcols: gl.constexpr = gl.SliceLayout(0, mfma_layout_b)
    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize
    ShuffleRows: gl.constexpr = min(KVBlockSize, 16)
    PAIRS: gl.constexpr = GROUP // 2

    # ---- work tile: (group rank, split, pair) inside each XCD slab -------------------
    pid = gl.program_id(0)
    SX = SplitKV // NUM_XCD
    GS = PAIRS * SX
    x_id = pid % NUM_XCD
    j = pid // NUM_XCD
    q_rank = j // GS
    t_split = (j % GS) // PAIRS
    p_pair = j % PAIRS
    ngroups = batch_size // GROUP
    L_RANK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, NumWarps], [1, 0])
    gi = gl.arange(0, NG2, layout=gl.SliceLayout(1, L_RANK))
    gj = gl.arange(0, NG2, layout=gl.SliceLayout(0, L_RANK))
    Li = gl.load(context_len_ptr + gi * GROUP, mask=gi < ngroups, other=-1)
    Lj = gl.load(context_len_ptr + gj * GROUP, mask=gj < ngroups, other=-1)
    ahead = (Lj[None, :] > Li[:, None]) | (
        (Lj[None, :] == Li[:, None]) & (gj[None, :] < gi[:, None])
    )
    rank = gl.sum(ahead.to(gl.int32), axis=1)
    g_sel = gl.sum(gl.where((rank == q_rank) & (gi < ngroups), gi, 0), axis=0)
    g_sel = _uniform_i32(g_sel)
    r0 = g_sel * GROUP + 2 * p_pair
    r1 = r0 + 1
    L0 = gl.load(context_len_ptr + r0)
    L1 = gl.load(context_len_ptr + r1)
    pid_split_kv = x_id * SX + t_split
    lead_is_1 = (L1 >= L0).to(gl.int32)
    Lmax = gl.maximum(L0, L1)
    Lmin = gl.minimum(L0, L1)
    lead = r0 + lead_is_1

    # ---- pairing check: do the two rows share every page of their common prefix? ------
    # Checked over the WHOLE common prefix, not just this workgroup's window, so that
    # every split of a pair takes the same branch: the fused pass partitions by the
    # longer row and the single-row passes by each row's own length, and mixing the
    # two across splits leaves columns of the shorter row unwritten.
    cnum = tl.cdiv(Lmax, ChunkK)
    scnum = cnum // SplitKV
    rcnum = cnum % SplitKV
    s_start = pid_split_kv * scnum * ChunkK + min(pid_split_kv, rcnum) * ChunkK
    s_len = min(
        Lmax - s_start, scnum * ChunkK + (ChunkK if pid_split_kv < rcnum else 0)
    )
    if s_len <= 0:
        return
    CHK: gl.constexpr = 64 * NumWarps
    L_CHK: gl.constexpr = gl.BlockedLayout([1], [64], [NumWarps], [0])
    pg_lo = 0
    pg_hi = tl.cdiv(Lmin, KVBlockSize)
    bad = gl.zeros([CHK], gl.int32, layout=L_CHK)
    for pb in range(pg_lo, pg_hi, CHK):
        offs = pb + gl.arange(0, CHK, layout=L_CHK)
        mchk = offs < pg_hi
        pa = gl.amd.cdna3.buffer_load(
            ptr=kv_indices, offsets=r0 * max_block_len + offs, mask=mchk, other=0
        )
        pbv = gl.amd.cdna3.buffer_load(
            ptr=kv_indices, offsets=r1 * max_block_len + offs, mask=mchk, other=0
        )
        bad = bad | (pa != pbv).to(gl.int32)
    fbi = _uniform_i32(gl.max(bad, axis=0))  # 0: fused pass, 1: two single-row passes

    n_pass = 1 + fbi
    for ip in range(n_pass):
        rowA = r0 + ip * fbi
        rowB = r1 - fbi + ip * fbi
        prow = lead * (1 - fbi) + rowA * fbi
        LA = L0 + ip * fbi * (L1 - L0)
        LB = L1 * (1 - fbi) + LA * fbi
        Lp = Lmax * (1 - fbi) + LA * fbi

        context_chunk_num = tl.cdiv(Lp, ChunkK)
        split_context_chunk_num = context_chunk_num // SplitKV
        residual_context_chunks = context_chunk_num % SplitKV
        split_context_start = (
            pid_split_kv * split_context_chunk_num * ChunkK
            + min(pid_split_kv, residual_context_chunks) * ChunkK
        )
        split_context_length = min(
            Lp - split_context_start,
            split_context_chunk_num * ChunkK
            + (ChunkK if pid_split_kv < residual_context_chunks else 0),
        )
        if split_context_length > 0:
            split_context_block = tl.cdiv(split_context_length, KVBlockSize)
            split_context_length = split_context_block * KVBlockSize
            residual_context_blocks = (
                ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
            ) % ContextBlockPerChunkK
            residual_context = residual_context_blocks * KVBlockSize

            _amd_iglp_sched_barrier(0x0)
            q_off = (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None] + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[
                None, :
            ]
            qA = gl.amd.cdna3.buffer_load(
                ptr=Q_buffer, offsets=rowA * stride_q_batch + q_off
            )
            qB = gl.amd.cdna3.buffer_load(
                ptr=Q_buffer, offsets=rowB * stride_q_batch + q_off
            )

            context_idx = split_context_start - residual_context
            kcol = gl.arange(0, ChunkKPerStage, layout=layout_kcols)
            mask_kv_next_0 = (
                context_idx // KVBlockSize + kcol // KVBlockSize
            ) >= split_context_start // KVBlockSize
            context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=prow * max_block_len
                + context_idx // KVBlockSize
                + kcol // KVBlockSize,
                mask=mask_kv_next_0,
            )
            mask_kv_next_1 = (
                (context_idx + ChunkKPerStage) // KVBlockSize + kcol // KVBlockSize
            ) >= split_context_start // KVBlockSize
            context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=prow * max_block_len
                + (context_idx + ChunkKPerStage) // KVBlockSize
                + kcol // KVBlockSize,
                mask=mask_kv_next_1,
            )
            w_off = gl.arange(0, ChunkQ, layout=layout_scale)
            wA = gl.amd.cdna3.buffer_load(
                ptr=weights, offsets=rowA * stride_w_batch + w_off
            )
            wB = gl.amd.cdna3.buffer_load(
                ptr=weights, offsets=rowB * stride_w_batch + w_off
            )

            offset_k_fixed = (
                gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
                + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
                // 16
                * (ShuffleRows * 16)
            )[:, None] + (
                kcol % ShuffleRows * 16
                + kcol % KVBlockSize // ShuffleRows * ShuffleRows * HiddenDim
            )[None, :]
            kscol = kcol % KVBlockSize

            _amd_iglp_sched_barrier(0x0)
            mfma_qA = gl.convert_layout(qA, mfma_layout_a)
            mfma_qB = gl.convert_layout(qB, mfma_layout_a)

            context_kv_idx_next_0 = tl.where(mask_kv_next_0, context_kv_idx_next_0, 0)
            k_next_0 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_0 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + kscol
            )
            if context_idx + ChunkK < split_context_start + split_context_length:
                context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=prow * max_block_len
                    + (context_idx + ChunkK) // KVBlockSize
                    + kcol // KVBlockSize,
                )
            _amd_iglp_sched_barrier(0x0)

            zero = gl.zeros(
                (ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout
            )
            outA = OutLogits_buffer + rowA.to(tl.int64) * stride_out_batch
            outB = OutLogits_buffer + rowB.to(tl.int64) * stride_out_batch
            ccol = gl.arange(0, ChunkKPerStage, layout=layout_cols)

            k = k_next_0
            k_scale_f = k_scale_f_next_0
            _amd_iglp_sched_barrier(0x0)
            context_kv_idx_next_1 = tl.where(mask_kv_next_1, context_kv_idx_next_1, 0)
            k_next_1 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_1 = gl.load(
                scale_buffer
                + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
                + kscol
            )
            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
            oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
            _amd_iglp_sched_barrier(0x0)
            k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
            oA = gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
            oB = gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
            _amd_s_set_prio(1)
            logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
            logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
            store_cols = context_idx + ccol
            smask = (store_cols >= split_context_start) & (store_cols < max_model_len)
            gl.amd.cdna3.buffer_store(logitsA, ptr=outA, offsets=store_cols, mask=smask)
            gl.amd.cdna3.buffer_store(logitsB, ptr=outB, offsets=store_cols, mask=smask)

            for context_idx in range(
                split_context_start - residual_context,
                split_context_start + split_context_length - ChunkK,
                ChunkK,
            ):
                k = k_next_1
                k_scale_f = k_scale_f_next_1
                _amd_iglp_sched_barrier(0x0)
                context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=prow * max_block_len
                    + (context_idx + ChunkK + ChunkKPerStage) // KVBlockSize
                    + kcol // KVBlockSize,
                )
                k_next_0 = gl.load(
                    KV_buffer
                    + offset_k_fixed
                    + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
                )
                k_scale_f_next_0 = gl.load(
                    scale_buffer
                    + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                    + kscol
                )
                _amd_s_set_prio(3)
                mfma_k = gl.convert_layout(k, mfma_layout_b)
                oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
                oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
                _amd_iglp_sched_barrier(0x0)
                k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
                oA = (
                    gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
                )
                oB = (
                    gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
                )
                _amd_s_set_prio(1)
                logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
                logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
                store_cols = context_idx + ChunkKPerStage + ccol
                smask = (store_cols >= split_context_start) & (
                    store_cols < max_model_len
                )
                gl.amd.cdna3.buffer_store(
                    logitsA, ptr=outA, offsets=store_cols, mask=smask
                )
                gl.amd.cdna3.buffer_store(
                    logitsB, ptr=outB, offsets=store_cols, mask=smask
                )

                k = k_next_0
                k_scale_f = k_scale_f_next_0
                _amd_iglp_sched_barrier(0x0)
                if (
                    context_idx + ChunkK + ChunkK
                    < split_context_start + split_context_length
                ):
                    context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                        ptr=kv_indices,
                        offsets=prow * max_block_len
                        + (context_idx + ChunkK + ChunkK) // KVBlockSize
                        + kcol // KVBlockSize,
                    )
                k_next_1 = gl.load(
                    KV_buffer
                    + offset_k_fixed
                    + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
                )
                k_scale_f_next_1 = gl.load(
                    scale_buffer
                    + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
                    + kscol
                )
                _amd_s_set_prio(2)
                mfma_k = gl.convert_layout(k, mfma_layout_b)
                oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
                oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
                _amd_iglp_sched_barrier(0x0)
                k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
                oA = (
                    gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
                )
                oB = (
                    gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
                )
                _amd_s_set_prio(0)
                logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
                logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
                store_cols = context_idx + ChunkK + ccol
                smask = store_cols < max_model_len
                gl.amd.cdna3.buffer_store(
                    logitsA, ptr=outA, offsets=store_cols, mask=smask
                )
                gl.amd.cdna3.buffer_store(
                    logitsB, ptr=outB, offsets=store_cols, mask=smask
                )

            context_idx = split_context_start + split_context_length - ChunkK
            k = k_next_1
            k_scale_f = k_scale_f_next_1
            _amd_s_set_prio(1)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            oA = _dot_fp8(mfma_qA, mfma_k, zero, 4)
            oB = _dot_fp8(mfma_qB, mfma_k, zero, 4)
            k_scale_f = gl.convert_layout(k_scale_f, layout_cols)
            oA = gl.maximum(oA, 0.0, propagate_nan=tl.PropagateNan.ALL) * wA[:, None]
            oB = gl.maximum(oB, 0.0, propagate_nan=tl.PropagateNan.ALL) * wB[:, None]
            _amd_s_set_prio(0)
            store_cols = context_idx + ChunkKPerStage + ccol
            logitsA = gl.reduce(oA, axis=0, combine_fn=_sum_combine) * k_scale_f
            logitsA = tl.where(store_cols <= LA - 1, logitsA, float("-inf"))
            logitsB = gl.reduce(oB, axis=0, combine_fn=_sum_combine) * k_scale_f
            logitsB = tl.where(store_cols <= LB - 1, logitsB, float("-inf"))
            smask = (store_cols >= split_context_start) & (store_cols < max_model_len)
            gl.amd.cdna3.buffer_store(logitsA, ptr=outA, offsets=store_cols, mask=smask)
            gl.amd.cdna3.buffer_store(logitsB, ptr=outB, offsets=store_cols, mask=smask)


# ===========================================================================
# Host side (aiter/ops/triton/attention/pa_mqa_logits.py, reduced to this path)
# ===========================================================================
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.experimental.gluon._runtime import GluonASTSource as ASTSource  # noqa: E402


@cache
def _compile_deepgemm_fp8_paged_mqa_logits(
    ChunkQ,
    ChunkK,
    Preshuffle,
    KVBlockSize,
    HiddenDim,
    is_padded_mode: bool,
    WavePerEU: int = 2,
    WavesPerEuHint: int = 2,
    LptGroup: int = 0,
):
    gfx_version = get_gfx()
    assert gfx_version == "gfx950", gfx_version
    assert Preshuffle
    cdna_version = get_cdna_version()
    warp_size = 64
    target = GPUTarget("hip", gfx_version, warp_size)
    gfx_fp8_pointer = "*fp8e4nv"

    fn_signature = {
        "batch_size": "i32",
        "next_n": "i32",
        "heads_num": "i32",
        "Q_buffer": gfx_fp8_pointer,
        "stride_q_batch": "i32",
        "stride_q_next_n": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": gfx_fp8_pointer,
        "stride_k_seq": "i32",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i64",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "num_block": "i32",
        "SplitKV": "i32",
    }
    fn_signature["ChunkQ"] = "constexpr"
    fn_signature["ChunkK"] = "constexpr"
    fn_signature["KVBlockSize"] = "constexpr"
    fn_signature["HiddenDim"] = "constexpr"
    fn_signature["CDNA_VERSION"] = "constexpr"
    fn_signature["ARCH"] = "constexpr"
    fn_signature["LPT_GROUP"] = "constexpr"
    fn_signature["NG2"] = "constexpr"

    options = {
        "num_warps": 4,
        "waves_per_eu": WavesPerEuHint,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": gfx_version,
        "backend_name": "hip",
        "warp_size": warp_size,
        "name": "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle",
    }

    kv_cache_attr = []
    if is_padded_mode:
        kv_cache_attr.append(["tt.divisibility", 16])

    src = ASTSource(
        fn=_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
            "CDNA_VERSION": cdna_version,
            "ARCH": gfx_version,
            "LPT_GROUP": LptGroup,
            "NG2": _LPT_NG2,
        },
        attrs={
            (2,): [["tt.divisibility", 16]],  # heads_num
            (3,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q_buffer
            (4,): [["tt.divisibility", 16]],  # stride_q_batch
            (5,): [["tt.divisibility", 16]],  # stride_q_next_n
            (6,): [["tt.divisibility", 16]],  # stride_q_heads
            (7,): kv_cache_attr,  # KV_buffer
            (8,): kv_cache_attr,  # stride_k_seq
            (9,): kv_cache_attr,  # scale_buffer
            (10,): kv_cache_attr,  # stride_scale_seq
            (11,): [["tt.pointer_range", 32]],  # context_len_ptr
            (12,): [["tt.pointer_range", 32]],  # kv_indices
            (13,): [
                ["tt.divisibility", 16],
                ["tt.pointer_range", 32],
            ],  # weights
            (14,): [["tt.divisibility", 16]],  # stride_w_batch
            # OutLogits_buffer: no tt.pointer_range 32 (row offsets of wide logits exceed 2**31 B)
        },
    )
    return triton.compile(src, target=target, options=options)


@cache
def _compile_paged_mqa_logits_fused2(
    ChunkQ,
    ChunkK,
    KVBlockSize,
    HiddenDim,
    is_padded_mode,
    WavesPerEuHint,
    Group,
    NumXcd,
):
    gfx_version = get_gfx()
    assert gfx_version == "gfx950", gfx_version
    warp_size = 64
    target = GPUTarget("hip", gfx_version, warp_size)
    fp8_ptr = "*fp8e4nv"
    fn_signature = {
        "batch_size": "i32",
        "Q_buffer": fp8_ptr,
        "stride_q_batch": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": fp8_ptr,
        "stride_k_seq": "i32",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i64",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "SplitKV": "i32",
        "ChunkQ": "constexpr",
        "ChunkK": "constexpr",
        "KVBlockSize": "constexpr",
        "HiddenDim": "constexpr",
        "GROUP": "constexpr",
        "NG2": "constexpr",
        "NUM_XCD": "constexpr",
    }
    options = {
        "num_warps": 4,
        "waves_per_eu": WavesPerEuHint,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": gfx_version,
        "backend_name": "hip",
        "warp_size": warp_size,
        "name": "_gluon_paged_mqa_logits_fused2",
    }
    kv_cache_attr = [["tt.divisibility", 16]] if is_padded_mode else []
    src = ASTSource(
        fn=_gluon_paged_mqa_logits_fused2,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
            "GROUP": Group,
            "NG2": _LPT_NG2,
            "NUM_XCD": NumXcd,
        },
        attrs={
            (1,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q_buffer
            (2,): [["tt.divisibility", 16]],  # stride_q_batch
            (3,): [["tt.divisibility", 16]],  # stride_q_heads
            (4,): kv_cache_attr,  # KV_buffer
            (5,): kv_cache_attr,  # stride_k_seq
            (6,): kv_cache_attr,  # scale_buffer
            (7,): kv_cache_attr,  # stride_scale_seq
            (8,): [["tt.pointer_range", 32]],  # context_len_ptr
            (9,): [["tt.pointer_range", 32]],  # kv_indices
            (10,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # weights
            (11,): [["tt.divisibility", 16]],  # stride_w_batch
        },
    )
    return triton.compile(src, target=target, options=options)


# KV-split granularity multiplier (see SplitKV below). Swept on GLM-5.2 verify rows (48-84) once
# the XCD-local program-id swizzle was in place: the optimum moves down from 32 to
# 4-8 (median_ms 1:0.368, 2:0.322, 4:0.3033, 6:0.3030, 8:0.3044, 12:0.3103,
# 16:0.3208, 24:0.3341, 32:0.3527, 48:0.3894) because the swizzle already fixes the
# makespan imbalance that the fine split was papering over, and coarser pieces
# amortize the per-piece prologue. SplitKV = 80 at mul=8 keeps grid % 8 == 0 for
# any row count, which the swizzle needs.
_SPLIT_MUL = 8

# Occupancy hint passed to the Triton AMD backend, decoupled from `WavePerEU` (which
# also feeds the SplitKV partition arithmetic -- changing that symbol would silently
# move SplitKV off 80 and break the `rows*SplitKV % 8 == 0` XCD-swizzle guard).
# The compiled kernel uses 92 VGPRs but the backend emits amdgpu-waves-per-eu="2,2"
# and reserves 169, capping occupancy at 2 waves/SIMD with zero LDS in play.
# Swept on GLM-5.2 verify rows (per-case ms, 48/60/72/84 rows): 2 -> .2482/.2467/.2941/.2956,
# 3 -> .2304/.2289/.2673/.2743, 4 -> .2268/.2230/.2622/.2657, 5 -> .2305/.2227/
# .2628/.2652. 4 is the knee; 6 would budget 512/6 = 85 < 92 VGPR and spill.
_WAVES_PER_EU = 4

# Longest-group-first phase-1 dispatch (see the LPT_GROUP branch of the kernel's
# work-tile mapping). Groups are the TARGET_VERIFY siblings (6 consecutive rows); the
# kernel ranks up to _LPT_NG2 groups. 0 disables (old split-major order).
_LPT = 1
_LPT_GROUP_ROWS = 6
_LPT_NG2 = 16

# Sibling-pair fusion of phase 1 (_gluon_paged_mqa_logits_fused2). 0 disables.
_FUSE = 1
_FUSE_SPLITKV = 64
_FUSE_WAVES_PER_EU = 3


def gluon_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # [batch, next_n, heads, hidden] fp8 e4m3fn
    kv_cache,  # [num_pages, KVBlockSize, 1, hidden + 4] (preshuffled fp8 + fp32 scales per page)
    weights: torch.Tensor,  # [batch * next_n, heads] fp32
    out_logits: torch.Tensor,  # [batch * next_n, >= max_model_len] fp32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 1,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_Size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    TileQCount = batch_size * next_n
    # Decouple KV split granularity from the waves_per_eu launch hint: the stock
    # formula yields SplitKV = 10 for every batch size here, so every program starts
    # at t=0 and the makespan equals the longest row (max/mean imbalance 1.8-2.4x).
    # A finer split gives the scheduler smaller pieces to backfill with.
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU * _SPLIT_MUL

    assert ChunkK % KVBlockSize == 0 or KVBlockSize % ChunkK == 0
    assert block_Size == KVBlockSize
    assert Preshuffle and KVBlockSize % 16 == 0

    kv_cache = kv_cache.view(-1, KVBlockSize * index_dim)
    num_block = kv_cache.shape[0]
    kv_cache_fp8, kv_cache_scale = (
        kv_cache[..., : KVBlockSize * hidden_dim],
        kv_cache[..., KVBlockSize * hidden_dim :],
    )
    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fn)
    kv_cache_scale = kv_cache_scale.view(torch.float32)

    is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0
    nx = _NUM_XCD.value
    if (
        _FUSE
        and next_n == 1
        and heads == 32
        and hidden_dim == 128
        and KVBlockSize == 64
        and ChunkK == 256
        and get_cdna_version() == 4
        and nx > 1
        and _FUSE_SPLITKV % nx == 0
        and batch_size % _LPT_GROUP_ROWS == 0
        and batch_size // _LPT_GROUP_ROWS <= _LPT_NG2
    ):
        kernel = _compile_paged_mqa_logits_fused2(
            heads,
            ChunkK,
            KVBlockSize,
            hidden_dim,
            is_padded_mode,
            _FUSE_WAVES_PER_EU,
            _LPT_GROUP_ROWS,
            nx,
        )
        grid = (batch_size // 2 * _FUSE_SPLITKV, 1, 1)
        kernel[grid](
            batch_size,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(2),
            kv_cache_fp8,
            kv_cache_fp8.stride(0),
            kv_cache_scale,
            kv_cache_scale.stride(0),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            _FUSE_SPLITKV,
            heads,
            ChunkK,
            KVBlockSize,
            hidden_dim,
            _LPT_GROUP_ROWS,
            _LPT_NG2,
            nx,
        )
        return

    grid = (batch_size * next_n * SplitKV, 1, 1)
    lpt_group = 0
    if (
        _LPT
        and next_n == 1
        and _NUM_XCD.value > 1
        and SplitKV % _NUM_XCD.value == 0
        and batch_size % _LPT_GROUP_ROWS == 0
        and batch_size // _LPT_GROUP_ROWS <= _LPT_NG2
    ):
        lpt_group = _LPT_GROUP_ROWS
    kernel = _compile_deepgemm_fp8_paged_mqa_logits(
        ChunkQ=heads,
        ChunkK=ChunkK,
        Preshuffle=Preshuffle,
        KVBlockSize=KVBlockSize,
        HiddenDim=hidden_dim,
        is_padded_mode=is_padded_mode,
        WavePerEU=WavePerEU,
        WavesPerEuHint=_WAVES_PER_EU,
        LptGroup=lpt_group,
    )
    cdna_version = get_cdna_version()
    kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_block_len,
        num_block,
        SplitKV,
        # constexpr
        heads,
        ChunkK,
        KVBlockSize,
        hidden_dim,
        cdna_version,
        get_gfx(),
        lpt_group,
        _LPT_NG2,
    )


# Validated row range (decode and MTP-verify batches of one DP rank).
MAX_ROWS = 128


def supported(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    *,
    preshuffle: bool,
    kv_block_size: int,
) -> bool:
    """Whether :func:`gluon_fp8_paged_mqa_logits` serves this call. ``q_fp8`` is
    ``[rows, heads, dim]`` and ``kv_cache_fp8`` ``[num_pages, page, 1, dim + 4]``."""
    return (
        preshuffle
        and kv_block_size == 64
        and q_fp8.dim() == 3
        and q_fp8.shape[1] == 32
        and q_fp8.shape[2] == 128
        and 1 <= q_fp8.shape[0] <= MAX_ROWS
        and kv_cache_fp8.dim() == 4
        and kv_cache_fp8.shape[1] == 64
        and get_gfx() == "gfx950"
    )
