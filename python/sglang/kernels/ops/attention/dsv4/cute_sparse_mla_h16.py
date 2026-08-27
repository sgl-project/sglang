"""SM90 CuTe sparse-MLA kernel for SGLang DSV4 TP=8 prefill.

This kernel is specialized for SGLang's DSV4 sparse-prefill contract:

* Q is padded to 64 heads by the model, while only the first 16 are logical.
* Q/K/V width is 512 (448 NoPE + 64 RoPE; V uses the same 512-wide row).
* indices address a per-call, rebased BF16 workspace and are already causal.
* topk_length is per query and attn_sink is a per-head virtual zero-value key.

The kernel below preserves the useful H=16 ``m64n16`` WGMMA decomposition,
uses a single-pass numerically stable online softmax, rescales FP32 PV state
when the running maximum changes, and overlaps the next sparse gather with the
current tile's PV WGMMA retirement.  This tuning variant also reduces the
transposed KQ fragment in registers: lane-XOR 4/8/16 covers the 16 rows owned
by each warp and only four per-warp partials per head cross shared memory.
To match FlashMLA's BF16 probability path, the online maximum covers real KV
rows only; the virtual attention sink is merged in the final denominator.
Its four 128-channel QK stages match the DSV4-512 layout, and the reduced
shared-memory footprint is launched with a two-CTA-per-SM occupancy request.

The public entry point returns only the 16 logical heads, with shape
``[TQ, 16, 512]``.  Callers must keep other architectures and DSV4 shapes on
their existing attention backend.
"""

from __future__ import annotations

import math
import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.cute.nvgpu.warpgroup import MmaF16BF16Op, OperandSource
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.dsv4.cutedsl_h16_contract import (
    DSV4_CUTEDSL_H16_MAX_TOPK as MAX_TOPK,
)
from sglang.kernels.ops.attention.dsv4.cutedsl_h16_contract import (
    DSV4_CUTEDSL_H16_TOPK_ALIGNMENT as TOPK_ALIGNMENT,
)
from sglang.srt.utils.custom_op import register_custom_op

# SGLang DSV4 TP=8 contract.
H = 16
DQK = 512
DV = 512
BN = 64
NTHR = 128
NG_QK = 4  # four 128-channel WGMMA groups cover DQK=512

# Interleaved shared-memory layout inherited from the tuned benchmark kernel.
DB_STRIDE = 72
# Eight 64-channel chunks.  The original DQK=576 kernel used 72 d-blocks
# (nine chunks); DSV4-512 only needs 64 and saves 9 KiB of shared memory.
NB_STRIDE = 64 * DB_STRIDE
KV_ELEMS = 8 * NB_STRIDE
CHUNK_SB = 8 * DB_STRIDE * 2

Q_CHUNK = H * 64
Q_ELEMS = (DQK // 64) * Q_CHUNK
P_ELEMS = BN * H
OT_STRIDE = DV + 8

LOG2E = 1.4426950408889634
NO_SINK = -1.0e30
# Must stay below the conventional no-sink score after LOG2E conversion so
# invalid rows can never raise the running maximum during the shuffle reduce.
INVALID_SCORE = -3.0e38


def _gp(addr):
    return cute.make_ptr(
        cutlass.BFloat16,
        addr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _sp(addr):
    return cute.make_ptr(
        cutlass.BFloat16,
        addr,
        cute.AddressSpace.smem,
        assumed_align=16,
    )


def _resolve_rows(
    kv_ptr,
    idx_ptr,
    skv_ptr,
    smask_ptr,
    base,
    g8,
    k8,
    topk_length,
    s_kv,
):
    """Resolve four gathered rows per 8-thread group and emit validity masks.

    SGLang has already applied causal/request masking before rebasing indices
    into its flat workspace.  Validity is therefore only:
      slot < topk_length and 0 <= index < s_kv.
    """

    ga = []
    sa = []
    predicates = []
    for c in range(4):
        n = c * 16 + g8
        slot = base + n
        row_raw = idx_ptr[slot]

        # Keep the helper branch-free and redirect every invalid address to
        # row zero. The copy predicate below zero-fills shared memory, while
        # the emitted score mask removes the row from attention.
        prefix_ok = cutlass.max(
            cutlass.min(topk_length - slot, cutlass.Int32(1)), cutlass.Int32(0)
        )
        lower_ok = cutlass.max(
            cutlass.min(row_raw + 1, cutlass.Int32(1)), cutlass.Int32(0)
        )
        upper_ok = cutlass.max(
            cutlass.min(s_kv - row_raw, cutlass.Int32(1)), cutlass.Int32(0)
        )
        ok = prefix_ok * lower_ok * upper_ok
        row = row_raw * ok
        smask_ptr[n] = ok.to(cutlass.Float32)

        # cp.async predication zero-fills the shared-memory destination for an
        # invalid row.  Merely redirecting the address to row zero is not
        # sufficient: a later BF16 ``0 * NaN`` in PV would still propagate a
        # poisoned value from kv[0].
        predicate = cute.make_rmem_tensor((1,), cute.Boolean)
        predicate.fill(ok != cutlass.Int32(0))
        predicates.append(predicate)

        ga.append((kv_ptr + row * DQK + k8 * 8).toint())
        sa.append(
            (skv_ptr + (n // 8) * NB_STRIDE + (n % 8) * 8 + k8 * DB_STRIDE).toint()
        )
    return ga, sa, predicates


def _prefetch_chunk(ga, sa, predicates, chunk):
    """Gather one 64-channel chunk for all 64 selected KV rows."""

    g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    # Expose the 128-bit BF16 copy atom explicitly as ATOM_V=8 and
    # ATOM_REST=1.  With a flat (8, 1) profile CuTe DSL 4.6 treats the vector
    # mode as predicate-visible and rejects our one-predicate-per-row tensor.
    # The nested profile makes the source/destination
    # ((ATOM_V, ATOM_REST), REST) and the predicate (ATOM_REST, REST), exactly
    # matching the documented predicated-copy contract while still emitting
    # one 16-byte cp.async transaction.
    l8 = cute.make_layout(((8, 1), 1))
    for c in range(4):
        cute.copy(
            g2s,
            cute.make_tensor(_gp(ga[c] + chunk * 128), l8),
            cute.make_tensor(_sp(sa[c] + chunk * CHUNK_SB), l8),
            pred=predicates[c],
        )


def _prefetch_tile(ga, sa, predicates):
    for chunk in range(DQK // 64):
        _prefetch_chunk(ga, sa, predicates, chunk)
        # Keep each cp.async group to two 64-channel chunks (eight copies per
        # thread).  Issuing all eight chunks in one uncommitted group exceeds
        # the conservative Hopper pipeline shape used by the proven kernel.
        if chunk % 2 == 1:
            cute.arch.cp_async_commit_group()


def _make_qk_fragments(mma_qk, thr_qk, skv_ptr, sq_ptr):
    """Build four K/Q fragments, each covering 128 of the 512 channels."""

    kv_frags = []
    q_frags = []
    channels = DQK // NG_QK
    dblocks = channels // 8
    for group in range(NG_QK):
        s_k = cute.make_tensor(
            skv_ptr + group * dblocks * DB_STRIDE,
            cute.make_layout(
                ((8, 8), (8, dblocks)),
                stride=((8, NB_STRIDE), (1, DB_STRIDE)),
            ),
        )
        s_q = cute.make_tensor(
            sq_ptr + group * 2 * Q_CHUNK,
            cute.make_layout(
                ((8, 2), (64, 2)),
                stride=((64, 512), (1, Q_CHUNK)),
            ),
        )
        kv_frags.append(mma_qk.make_fragment_A(thr_qk.partition_A(s_k)))
        q_frags.append(mma_qk.make_fragment_B(thr_qk.partition_B(s_q)))
    return kv_frags, q_frags


def _make_pv_fragments(mma_pv, thr_pv, skv_ptr):
    """Build eight 64-channel V fragments and FP32 output accumulators."""

    v_frags = []
    out_acc = []
    id_o = cute.make_identity_tensor((64, H))
    for chunk in range(DV // 64):
        s_v = cute.make_tensor(
            skv_ptr + 8 * chunk * DB_STRIDE,
            cute.make_layout(
                ((8, 8), (8, 8)),
                stride=((1, DB_STRIDE), (8, NB_STRIDE)),
            ),
        )
        v_frags.append(mma_pv.make_fragment_A(thr_pv.partition_A(s_v)))
        acc = cute.make_rmem_tensor(thr_pv.partition_C(id_o).shape, cutlass.Float32)
        acc.fill(0.0)
        out_acc.append(acc)
    return v_frags, out_acc


def _issue_qk_staged(mma_qk, score_acc, k_frags, q_frags):
    """Consume four cp.async groups and enqueue four 128-channel QK MMAs."""

    cute.nvgpu.warpgroup.fence()
    mma_qk.set(warpgroup.Field.ACCUMULATE, False)
    for group in range(NG_QK):
        cute.arch.cp_async_wait_group(NG_QK - 1 - group)
        cute.arch.barrier()
        cute.gemm(mma_qk, score_acc, k_frags[group], q_frags[group], score_acc)
        cute.nvgpu.warpgroup.commit_group()
        mma_qk.set(warpgroup.Field.ACCUMULATE, True)


def _pv_step(mma_pv, out_acc, v_frags, prob_frag, chunk, fence=False):
    if fence:
        cute.nvgpu.warpgroup.fence()
    cute.gemm(
        mma_pv,
        out_acc[chunk],
        v_frags[chunk],
        prob_frag,
        out_acc[chunk],
    )
    cute.nvgpu.warpgroup.commit_group()


def _queue_pv_final(mma_pv, out_acc, v_frags, prob_frag):
    """Last tile: enqueue PV and let denominator reduction hide its latency."""

    _pv_step(mma_pv, out_acc, v_frags, prob_frag, 0, fence=True)
    for chunk in range(1, DV // 64):
        _pv_step(mma_pv, out_acc, v_frags, prob_frag, chunk)


def _run_pv_rolling(
    mma_pv,
    out_acc,
    v_frags,
    prob_frag,
    next_ga,
    next_sa,
    next_predicates,
):
    """Accumulate PV while each retired V chunk is refilled for the next tile."""

    _pv_step(mma_pv, out_acc, v_frags, prob_frag, 0, fence=True)
    _pv_step(mma_pv, out_acc, v_frags, prob_frag, 1)
    for chunk in range(6):
        _pv_step(mma_pv, out_acc, v_frags, prob_frag, chunk + 2)
        cute.nvgpu.warpgroup.wait_group(2)
        _prefetch_chunk(next_ga, next_sa, next_predicates, chunk)
        if chunk % 2 == 1:
            cute.arch.cp_async_commit_group()
    cute.nvgpu.warpgroup.wait_group(1)
    _prefetch_chunk(next_ga, next_sa, next_predicates, 6)
    cute.nvgpu.warpgroup.wait_group(0)
    _prefetch_chunk(next_ga, next_sa, next_predicates, 7)
    cute.arch.cp_async_commit_group()


def _reduce_kq_fragment_heads(fragment, op):
    """Reduce the two rows/thread then the 16 rows/warp for four heads.

    The ``m64n16`` accumulator TV layout expected by this reduction is::

      Shape ((4, 8, 4), (2, 2, 2))
      Stride((128,1,16), (64,8,512))

    The value coordinates therefore flatten as ``v0 + 2*v1 + 4*v2``.
    ``v1`` selects the two M rows held by a thread while ``(v0,v2)``
    selects its four N/head values.  For a fixed head, lanes with the same
    ``lane % 4`` own all 16 rows in a warp, hence XOR 4, 8 and 16.
    """

    r0 = op(fragment[0], fragment[2])
    r1 = op(fragment[1], fragment[3])
    r2 = op(fragment[4], fragment[6])
    r3 = op(fragment[5], fragment[7])
    for offset in (4, 8, 16):
        r0 = op(r0, cute.arch.shuffle_sync_bfly(r0, offset=offset))
        r1 = op(r1, cute.arch.shuffle_sync_bfly(r1, offset=offset))
        r2 = op(r2, cute.arch.shuffle_sync_bfly(r2, offset=offset))
        r3 = op(r3, cute.arch.shuffle_sync_bfly(r3, offset=offset))
    return r0, r1, r2, r3


def _store_output(thr_pv, sout_ptr, inv_den_ptr, out_acc):
    den_view = cute.make_tensor(inv_den_ptr, cute.make_layout((64, H), stride=(0, 1)))
    den_frag = thr_pv.partition_C(den_view)
    for chunk in range(DV // 64):
        out_view = cute.make_tensor(
            sout_ptr + chunk * 64,
            cute.make_layout((64, H), stride=(1, OT_STRIDE)),
        )
        out_frag = thr_pv.partition_C(out_view)
        for i in range(cute.size(out_acc[chunk])):
            out_frag[i] = (out_acc[chunk][i] * den_frag[i]).to(cutlass.BFloat16)


@cute.kernel
def _kernel(
    m_q: cute.Tensor,
    m_kv: cute.Tensor,
    m_indices: cute.Tensor,
    m_topk_length: cute.Tensor,
    m_attn_sink: cute.Tensor,
    m_out: cute.Tensor,
    mma_qk: cute.TiledMma,
    mma_pv: cute.TiledMma,
    sm_scale: cutlass.Float32,
    tq: cutlass.Int32,
    s_kv: cutlass.Int32,
    topk: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # Reverse query order to retain the benchmark kernel's launch behaviour;
    # unlike the old kernel, this value is not used as a causal KV position.
    qpos = cutlass.Int32(tq - 1) - cutlass.Int32(bidx)
    g8 = tidx // 8
    k8 = tidx % 8

    smem = utils.SmemAllocator()
    skv_ptr = smem.allocate_array(cutlass.BFloat16, KV_ELEMS, byte_alignment=1024)
    sq_raw = smem.allocate_array(cutlass.BFloat16, Q_ELEMS, byte_alignment=1024)
    q_atom = warpgroup.make_smem_layout_atom(
        warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
    )
    sq_ptr = cute.recast_ptr(sq_raw, q_atom.inner)
    sp_ptr = smem.allocate_array(cutlass.BFloat16, P_ELEMS, byte_alignment=1024)
    # Four warp-local partials per head replace the old full 64x16 FP32
    # score spill.  The same allocation is reused for max and denominator.
    spartial_ptr = smem.allocate_array(cutlass.Float32, 4 * H, byte_alignment=128)
    smask_ptr = smem.allocate_array(cutlass.Float32, 2 * BN, byte_alignment=128)
    smax_ptr = smem.allocate_array(cutlass.Float32, H, byte_alignment=128)
    salpha_ptr = smem.allocate_array(cutlass.Float32, H, byte_alignment=128)
    ssink_ptr = smem.allocate_array(cutlass.Float32, H, byte_alignment=128)
    inv_den_ptr = smem.allocate_array(cutlass.Float32, H, byte_alignment=128)

    # The output epilogue aliases the consumed KV staging storage.
    sout_ptr = skv_ptr

    # Accept SGLang's native compact H64 row without materializing q[:,:16].
    # Only the first H*DQK elements of each row are loaded.
    q_ptr = m_q.iterator + qpos * m_q.stride[0]
    kv_ptr = m_kv.iterator
    idx_ptr = m_indices.iterator + qpos * topk
    out_ptr = m_out.iterator + qpos * (H * DV)
    # Keep malformed metadata from repeating the final tile.  This scalar
    # clamp lives in the CTA, so it adds no host-side tensor allocation or
    # auxiliary CUDA launch on the production path.
    topk_length = cutlass.max(
        cutlass.min(m_topk_length.iterator[qpos], cutlass.Int32(topk)),
        cutlass.Int32(0),
    )
    # Run one masked tile for length=0 so the launch has a uniform prologue.
    n_tiles = cutlass.max((topk_length + (BN - 1)) // BN, cutlass.Int32(1))
    scale_log2 = sm_scale * LOG2E

    # Load the full 16x512 logical Q tile once.  Keep this cp.async group in
    # flight while the four tile-0 KV groups are issued below.
    g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    l8 = cute.make_layout((8, 1))
    for j in cutlass.range_constexpr(DQK // 64):
        linear = (j * NTHR + tidx) * 8
        head = linear // DQK
        dim = linear - head * DQK
        qoff = (dim // 64) * Q_CHUNK + (head // 8) * 512 + (head % 8) * 64 + (dim % 64)
        qdst = cute.make_ptr(
            cutlass.BFloat16,
            (sq_raw + qoff).toint(),
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        qdst = cute.recast_ptr(qdst, q_atom.inner)
        cute.copy(
            g2s,
            cute.make_tensor(_gp((q_ptr + linear).toint()), l8),
            cute.make_tensor(qdst, l8),
        )
    cute.arch.cp_async_commit_group()

    if tidx < H:
        sink = m_attn_sink.iterator[tidx] * LOG2E
        ssink_ptr[tidx] = sink
        # Match FlashMLA's numerical path: the online maximum is formed from
        # real K rows only.  The virtual zero-value sink is merged into the
        # denominator in the epilogue, after P has been rounded to BF16.
        smax_ptr[tidx] = cutlass.Float32(NO_SINK)
        salpha_ptr[tidx] = cutlass.Float32(1.0)
    cute.arch.barrier()

    # Tile-0 prologue.  Together with Q this leaves five committed groups;
    # _issue_qk_staged(wait=3,2,1,0) releases Q+each 128-channel KV pair just in
    # time for the corresponding WGMMA.
    ga0, sa0, predicates0 = _resolve_rows(
        kv_ptr,
        idx_ptr,
        skv_ptr,
        smask_ptr,
        cutlass.Int32(0),
        g8,
        k8,
        topk_length,
        s_kv,
    )
    _prefetch_tile(ga0, sa0, predicates0)

    thr_qk = mma_qk.get_slice(tidx)
    thr_pv = mma_pv.get_slice(tidx)
    k_frags, q_frags = _make_qk_fragments(mma_qk, thr_qk, skv_ptr, sq_ptr)
    v_frags, out_acc = _make_pv_fragments(mma_pv, thr_pv, skv_ptr)

    id_score = cute.make_identity_tensor((BN, H))
    score_acc = cute.make_rmem_tensor(
        thr_qk.partition_C(id_score).shape, cutlass.Float32
    )
    prob_acc = cute.make_rmem_tensor(score_acc.shape, cutlass.BFloat16)
    sum_acc = cute.make_rmem_tensor(score_acc.shape, cutlass.Float32)
    sum_acc.fill(0.0)

    # QK writes P through the C-fragment view.  PV must consume the same bytes
    # through a distinct transposed B-fragment view; the two MMA layouts are
    # not interchangeable even though they alias the same shared allocation.
    prob_store_view = cute.make_tensor(
        sp_ptr,
        cute.make_layout(((8, 8), (8, 2)), stride=((8, 128), (1, 64))),
    )
    prob_store_frag = thr_qk.partition_C(prob_store_view)
    prob_mma_view = cute.make_tensor(
        sp_ptr,
        cute.make_layout(((8, 2), (8, 8)), stride=((1, 64), (8, 128))),
    )
    prob_mma_frag = mma_pv.make_fragment_B(thr_pv.partition_B(prob_mma_view))

    max_view = cute.make_tensor(smax_ptr, cute.make_layout((BN, H), stride=(0, 1)))
    max_frag = thr_qk.partition_C(max_view)
    alpha_qk_view = cute.make_tensor(
        salpha_ptr, cute.make_layout((BN, H), stride=(0, 1))
    )
    alpha_qk_frag = thr_qk.partition_C(alpha_qk_view)
    alpha_pv_view = cute.make_tensor(
        salpha_ptr, cute.make_layout((64, H), stride=(0, 1))
    )
    alpha_pv_frag = thr_pv.partition_C(alpha_pv_view)

    mma_pv.set(warpgroup.Field.ACCUMULATE, True)

    # Single-pass stable online softmax.  Each tile updates the per-head max,
    # rescales the FP32 denominator and PV accumulators by exp(old_max-new_max),
    # then accumulates probabilities relative to the new max.
    for tile in cutlass.range(n_tiles, unroll=1):
        buf = tile % 2
        next_buf = cutlass.Int32(1) - buf
        current_mask_ptr = smask_ptr + buf * BN

        _issue_qk_staged(mma_qk, score_acc, k_frags, q_frags)

        # Hide next-index loads/address arithmetic under the four in-flight QK
        # WGMMA groups.  The alternate mask buffer stays live until next tile.
        next_base = cutlass.min((tile + 1) * BN, cutlass.Int32(topk - BN))
        next_ga, next_sa, next_predicates = _resolve_rows(
            kv_ptr,
            idx_ptr,
            skv_ptr,
            smask_ptr + next_buf * BN,
            next_base,
            g8,
            k8,
            topk_length,
            s_kv,
        )

        cute.nvgpu.warpgroup.wait_group(0)

        current_mask_view = cute.make_tensor(
            current_mask_ptr, cute.make_layout((BN, H), stride=(1, 0))
        )
        current_mask_frag = thr_qk.partition_C(current_mask_view)

        # QK sets ACCUMULATE=False at the next tile, so scale/mask in place.
        for i in cutlass.range_constexpr(cute.size(score_acc)):
            score_acc[i] = (
                score_acc[i] * scale_log2
                if current_mask_frag[i] > 0.0
                else cutlass.Float32(INVALID_SCORE)
            )
        warp_max = _reduce_kq_fragment_heads(score_acc, cutlass.max)
        # Keep this dynamic lane branch directly in @cute.kernel so its SSA
        # values remain in the kernel's traced control flow.
        lane_partial = tidx % 32
        if lane_partial < 4:
            warp_partial = tidx // 32
            head_lo = 2 * lane_partial
            spartial_ptr[warp_partial * H + head_lo] = warp_max[0]
            spartial_ptr[warp_partial * H + head_lo + 1] = warp_max[1]
            spartial_ptr[warp_partial * H + head_lo + 8] = warp_max[2]
            spartial_ptr[warp_partial * H + head_lo + 9] = warp_max[3]
        cute.arch.barrier()

        if tidx < H:
            old_max = smax_ptr[tidx]
            new_max = old_max
            for warp in cutlass.range_constexpr(4):
                new_max = cutlass.max(new_max, spartial_ptr[warp * H + tidx])
            smax_ptr[tidx] = new_max
            salpha_ptr[tidx] = (
                cutlass.Float32(1.0)
                if old_max == new_max
                else cute.math.exp2(old_max - new_max, fastmath=True)
            )
        cute.arch.barrier()

        if tile > 0:
            for i in cutlass.range_constexpr(cute.size(score_acc)):
                sum_acc[i] = sum_acc[i] * alpha_qk_frag[i]

        # Multiplying after exp2 is essential for length=0: the uniform masked
        # tile must contribute exactly zero to both P and the denominator.
        for i in cutlass.range_constexpr(cute.size(score_acc)):
            p = (
                cute.math.exp2(score_acc[i] - max_frag[i], fastmath=True)
                * current_mask_frag[i]
            )
            prob_acc[i] = p.to(cutlass.BFloat16)
            sum_acc[i] = sum_acc[i] + p

        # PV accumulators have a different C-fragment distribution from QK;
        # use the matching broadcast view when applying the same head alpha.
        if tile > 0:
            for i in cutlass.range_constexpr(cute.size(out_acc[0])):
                alpha = alpha_pv_frag[i]
                for chunk in cutlass.range_constexpr(DV // 64):
                    out_acc[chunk][i] = out_acc[chunk][i] * alpha

        cute.autovec_copy(prob_acc, prob_store_frag)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        if tile + 1 < n_tiles:
            _run_pv_rolling(
                mma_pv,
                out_acc,
                v_frags,
                prob_mma_frag,
                next_ga,
                next_sa,
                next_predicates,
            )
        else:
            _queue_pv_final(mma_pv, out_acc, v_frags, prob_mma_frag)

    # Reduce the real-key denominator directly from its distributed register
    # fragment and add the zero-value sink term.  This avoids a second 64x16
    # FP32 spill plus the old 16-thread serial scan of 64 rows.
    warp_sum = _reduce_kq_fragment_heads(sum_acc, lambda x, y: x + y)
    lane_partial = tidx % 32
    if lane_partial < 4:
        warp_partial = tidx // 32
        head_lo = 2 * lane_partial
        spartial_ptr[warp_partial * H + head_lo] = warp_sum[0]
        spartial_ptr[warp_partial * H + head_lo + 1] = warp_sum[1]
        spartial_ptr[warp_partial * H + head_lo + 8] = warp_sum[2]
        spartial_ptr[warp_partial * H + head_lo + 9] = warp_sum[3]
    cute.arch.barrier()
    if tidx < H:
        denom = cutlass.Float32(0.0)
        for warp in cutlass.range_constexpr(4):
            denom = denom + spartial_ptr[warp * H + tidx]
        sink_mass = (
            cutlass.Float32(1.0)
            if ssink_ptr[tidx] == smax_ptr[tidx]
            else cute.math.exp2(ssink_ptr[tidx] - smax_ptr[tidx], fastmath=True)
        )
        denom = denom + sink_mass
        inv_den_ptr[tidx] = (
            cutlass.Float32(1.0) / denom if denom > 0.0 else cutlass.Float32(0.0)
        )
    cute.arch.barrier()

    # The final tile's PV groups were deliberately left in flight while the
    # independent denominator reduction ran above.
    cute.nvgpu.warpgroup.wait_group(0)
    _store_output(thr_pv, sout_ptr, inv_den_ptr, out_acc)
    cute.arch.barrier()

    for chunk in cutlass.range_constexpr(DV // 64):
        linear = chunk * NTHR + tidx
        head = linear // 64
        dim = linear % 64
        cute.autovec_copy(
            cute.make_tensor(_sp((sout_ptr + head * OT_STRIDE + dim * 8).toint()), l8),
            cute.make_tensor(_gp((out_ptr + head * DV + dim * 8).toint()), l8),
        )


@cute.jit
def _launch(
    m_q: cute.Tensor,
    m_kv: cute.Tensor,
    m_indices: cute.Tensor,
    m_topk_length: cute.Tensor,
    m_attn_sink: cute.Tensor,
    m_out: cute.Tensor,
    tq: cutlass.Int32,
    s_kv: cutlass.Int32,
    sm_scale: cutlass.Float32,
    stream,
):
    topk = cute.size(m_indices, mode=[1])

    op_qk = MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        (64, H, 16),
        OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    mma_qk = cute.make_tiled_mma(cute.make_mma_atom(op_qk), (1, 1, 1))
    op_pv = MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        (64, H, 16),
        OperandSource.SMEM,
        OperandMajorMode.MN,
        OperandMajorMode.MN,
    )
    mma_pv = cute.make_tiled_mma(cute.make_mma_atom(op_pv), (1, 1, 1))

    _kernel(
        m_q,
        m_kv,
        m_indices,
        m_topk_length,
        m_attn_sink,
        m_out,
        mma_qk,
        mma_pv,
        sm_scale,
        tq,
        s_kv,
        topk,
    ).launch(
        grid=(tq, 1, 1),
        block=(NTHR, 1, 1),
        stream=stream,
        min_blocks_per_mp=2,
    )


_COMPILE_CACHE: dict[tuple, object] = {}
_COMPILE_LOCK = threading.Lock()


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Validate the allocation-free DSV4 H16 launch contract."""

    if q.ndim != 3 or q.shape[1] not in (H, 64) or q.shape[2] != DQK:
        raise ValueError(f"q must have shape [TQ, 16|64, 512], got {tuple(q.shape)}")
    if kv.ndim != 2 or kv.shape[1] != DQK:
        raise ValueError(f"kv must have shape [SKV, 512], got {tuple(kv.shape)}")
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"indices must have shape [TQ, TOPK], got {tuple(indices.shape)}"
        )
    if topk_length.shape != (q.shape[0],):
        raise ValueError(
            f"topk_length must have shape [{q.shape[0]}], "
            f"got {tuple(topk_length.shape)}"
        )
    if attn_sink.ndim != 1 or attn_sink.numel() < H:
        raise ValueError(
            f"attn_sink must have shape [>=16], got {tuple(attn_sink.shape)}"
        )

    topk = indices.shape[1]
    if topk == 0 or topk % TOPK_ALIGNMENT != 0 or topk > MAX_TOPK:
        raise ValueError(
            f"TOPK width must be in [{TOPK_ALIGNMENT}, {MAX_TOPK}] and divisible by "
            f"{TOPK_ALIGNMENT}; got {topk}"
        )
    if kv.shape[0] == 0:
        raise ValueError("kv workspace must contain at least one row")

    tensors = (q, kv, indices, topk_length, attn_sink)
    if not all(t.is_cuda for t in tensors):
        raise ValueError(
            "q, kv, indices, topk_length, and attn_sink must be CUDA tensors"
        )
    if not all(t.device == q.device for t in tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError(f"q and kv must be bfloat16, got {q.dtype} and {kv.dtype}")
    if indices.dtype != torch.int32:
        raise TypeError(f"indices must be int32, got {indices.dtype}")
    if topk_length.dtype != torch.int32:
        raise TypeError(f"topk_length must be int32, got {topk_length.dtype}")
    if attn_sink.dtype != torch.float32:
        raise TypeError(f"attn_sink must be float32, got {attn_sink.dtype}")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous")
    if q.data_ptr() % 16 != 0 or kv.data_ptr() % 16 != 0:
        raise ValueError("q and kv storage must be aligned to 16 bytes")

    capability = torch.cuda.get_device_capability(q.device)
    if capability != (9, 0):
        raise ValueError(
            "cute_sparse_mla_h16_fwd requires a Hopper SM90 GPU, "
            f"got compute capability {capability[0]}.{capability[1]}"
        )

    scale = float(sm_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"sm_scale must be finite and positive, got {sm_scale}")

    # The model may carry a sink padded to the query's H64 storage shape.  A
    # leading contiguous view keeps only the 16 logical heads without a copy.
    return q, kv, indices, topk_length, attn_sink[:H], scale


def _run_impl(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
) -> None:
    with torch.cuda.device(q.device):
        # The leading extents are dynamic.  Inner modes, compact strides,
        # dtypes, and alignment remain specialized in the cached executor.
        m_q = from_dlpack(q, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        m_kv = from_dlpack(kv, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        m_indices = from_dlpack(indices, assumed_align=4).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        m_topk_length = from_dlpack(
            topk_length, assumed_align=4
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
        m_attn_sink = from_dlpack(attn_sink, assumed_align=4)
        m_out = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        tq = cutlass.Int32(int(q.shape[0]))
        s_kv = cutlass.Int32(int(kv.shape[0]))
        scale = cutlass.Float32(sm_scale)

        # TQ, KV mode 0, and s_kv are runtime values, so one executor is
        # reusable across changing chunk and workspace lengths.
        capability = torch.cuda.get_device_capability(q.device)
        key = (
            q.device.index,
            capability,
            int(q.shape[1]),
            int(indices.shape[1]),
        )
        fn = _COMPILE_CACHE.get(key)
        if fn is None:
            with _COMPILE_LOCK:
                fn = _COMPILE_CACHE.get(key)
                if fn is None:
                    fn = cute.compile(
                        _launch,
                        m_q,
                        m_kv,
                        m_indices,
                        m_topk_length,
                        m_attn_sink,
                        m_out,
                        tq,
                        s_kv,
                        scale,
                        stream,
                    )
                    _COMPILE_CACHE[key] = fn
        fn(
            m_q,
            m_kv,
            m_indices,
            m_topk_length,
            m_attn_sink,
            m_out,
            tq,
            s_kv,
            scale,
            stream,
        )


@register_custom_op(
    op_name="cute_sparse_mla_h16",
    mutates_args=["output"],
)
def _cute_sparse_mla_h16_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
) -> None:
    _run_impl(q, kv, indices, topk_length, attn_sink, output, sm_scale)


def cute_sparse_mla_h16_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Run native-H16 DSV4 sparse prefill and return ``[TQ, 16, 512]``."""

    q, kv, indices, topk_length, attn_sink, scale = _validate_inputs(
        q, kv, indices, topk_length, attn_sink, sm_scale
    )
    output = torch.empty((q.shape[0], H, DV), dtype=torch.bfloat16, device=q.device)
    if q.shape[0] == 0:
        return output

    _cute_sparse_mla_h16_op(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        output,
        scale,
    )
    return output


__all__ = ["cute_sparse_mla_h16_fwd"]
