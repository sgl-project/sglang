# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined.
# ruff: noqa  -- vendored kernel library, minimal local deltas
"""K4-only fused KDA kernel: persistent variant with varlen support.

Persistent scheduler (GDNTileScheduler) with 3D tensor layout, TensorMapManager
for per-tile TMA descriptor updates, and domain_offset+flat_divide for per-chunk
addressing. Supports variable-length sequences via cu_seqlens.

K4 chunk loop with 6 MMAs per chunk:
  MMA1: W = AB @ KS          (K-MN, K=64)
  MMA2: U = AB @ V           (K-MN, K=64)
  MMA3: NV = U + W_bf16 @ S  (SS-mode: A=SMEM sO, B=SMEM sST, K=128, accumulate)
  MMA4: OI = QS @ S          (K-MN, K=128)
  MMA5: O = OI + AQC @ NV    (K-MN, K=64, accumulate)
  MMA6: State += NV^T @ KG   (MN-MN, K=64, accumulate on decayed state)

Execution order: MMA1->MMA2->MMA4->MMA3->MMA5->MMA6

Warp assignment (3 warpgroups, 12 warps, 384 threads):
  WG0 (W0-3):  W0=MMA issue, W2=TMA load/store, W1/W3=idle
  WG1 (W4-7):  State 2-pass: TMEM->bf16->sST, gk(SMEM)->decay->TMEM
  WG2 (W8-11): GDN readout: W/NV/O TMEM->bf16->SMEM

TMA pipelines (all PipelineTmaUmma, 1-stage):
  - KS, V, QS: prefetch c+1 while MMA uses c
  - AB, AQC, KG: 1-stage with explicit consumer release (no prefetch)
  - TMA warp decoupled from O readout (no store_nbar wait)

State management:
  - tCtState [128,128] fp32 @ TMEM offset 256: persistent
  - 2-pass per chunk: bf16->sST early, gk decay->TMEM late
  - gk_last preloaded to SMEM (coalesced), read in Pass 2
  - single state_ready_mbar after both passes (TMEM no-overlap constraint)
"""

import os
import sys
import warnings

import cutlass.cutlass_dsl as _dsl_mod

if not hasattr(_dsl_mod, "CuteExperimentalDSL"):

    class _DummyExperimentalDSL:
        jit = None
        kernel = None
        compile = None

    _dsl_mod.CuteExperimentalDSL = _DummyExperimentalDSL

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute import KeepCUBIN, PtxasOptions
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import Field, OperandMajorMode, OperandSource
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int32, T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# GDN tile scheduler: import from the public FlashInfer release
# (`pip install flashinfer-python>=0.6.13`). Fall back to a local
# dynamic-kernel-generator checkout only if FlashInfer is unavailable.
try:
    from flashinfer.gdn_kernels.blackwell.gated_delta_net_tile_scheduler import (
        GDNTileScheduler,
        GDNTileSchedulerParams,
    )
except ImportError:
    # Try multiple locations for GDNTileScheduler (workspace layout vs kda_optimized layout)
    _GDN_PATHS = [
        # workspace/flash-linear-attention/prefill -> ../../../dynamic-kernel-generator
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "dynamic-kernel-generator",
            "cutlass_ir",
            "compiler",
            "python",
            "examples",
            "blackwell",
            "gated_delta_net",
        ),
        # scripts/kda_optimized -> ../../dynamic-kernel-generator
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "dynamic-kernel-generator",
            "cutlass_ir",
            "compiler",
            "python",
            "examples",
            "blackwell",
            "gated_delta_net",
        ),
    ]
    for _p in _GDN_PATHS:
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
            break
    from gated_delta_net_tile_scheduler import GDNTileSchedulerParams, GDNTileScheduler

from cutlass.utils import TensorMapManager, TensorMapUpdateMode


def transform_partitioned_tensor_layout(tensor):
    layout = tensor.layout
    stored_layout = layout
    if isinstance(stored_layout, cute.ComposedLayout):
        layout = layout.outer
    shape = layout.shape
    stride = layout.stride
    new_shape = ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:])
    new_stride = ((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:])
    new_layout = cute.make_layout(shape=new_shape, stride=new_stride)
    if isinstance(stored_layout, cute.ComposedLayout):
        new_layout = cute.make_composed_layout(
            stored_layout.inner, stored_layout.offset, new_layout
        )
    return cute.make_tensor(tensor.iterator, new_layout)


mma_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32
out_dtype = cutlass.BFloat16

M = 64
N = 128
K = 64
K3 = 128
M6 = 128
N6 = 128
K6 = 64

threads_per_cta = 384
warp_threads = 32
warpgroup_threads = 128

BYTES_PER_TENSORMAP = 128
NUM_TENSORMAPS = 7  # a, b, v, q, aqc, kg, o

MMA_WARP = 0
O_STORE_WARP = 1
TMA_WARP = 2
STATE_WG = 1
READOUT_WG = 2

MAX_REGS = 168

try:
    from cutlass.cutlass_dsl.cutlass import CuTeDSL as _CuTeDSL

    _orig_get_pipeline = _CuTeDSL._get_pipeline
    _patch_applied = False

    def _patched_get_pipeline(self, _pipeline):
        global _patch_applied
        result = _orig_get_pipeline(self, _pipeline)
        if result and "ptx-options=" not in result:
            if "cubin-format=bin" in result:
                result = result.replace(
                    "cubin-format=bin", "cubin-format=bin ptx-options='--uumn'"
                )
                _patch_applied = True
            else:
                warnings.warn(
                    "kda_nvidia_prefill: cubin-format=bin not found in the CuTeDSL "
                    "pipeline, ptx-options='--uumn' was not injected"
                )
        elif result and "ptx-options=" in result:
            _patch_applied = True
        return result

    _CuTeDSL._get_pipeline = _patched_get_pipeline
except Exception as e:
    warnings.warn(f"kda_nvidia_prefill: CuTeDSL ptx-options patch failed: {e}")
    _patch_applied = False


@cute.kernel
def k4_persistent_kernel(
    tiled_mma_kmn: cute.TiledMma,
    tiled_mma_mn_mn: cute.TiledMma,
    tma_a,
    a_sl: cute.ComposedLayout,
    tma_b,
    b_sl: cute.ComposedLayout,
    tma_v,
    v_sl: cute.ComposedLayout,
    s_sl: cute.ComposedLayout,
    tma_q,
    q_sl: cute.ComposedLayout,
    tma_aqc,
    aqc_sl: cute.ComposedLayout,
    tma_kg,
    kg_sl: cute.ComposedLayout,
    tma_o,
    store_sl: cute.ComposedLayout,
    readout_k_sl: cute.ComposedLayout,
    nv_b_sl: cute.ComposedLayout,
    nv_a_sl: cute.ComposedLayout,
    kg_a_sl: cute.ComposedLayout,
    nv_b_mn_sl: cute.ComposedLayout,
    mGkLastExp: cute.Tensor,
    mS_fp32: cute.Tensor,
    cu_seqlens: cute.Tensor,
    chunk_offsets: cute.Tensor,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mV_g: cute.Tensor,
    mQ: cute.Tensor,
    mAQC: cute.Tensor,
    mKG: cute.Tensor,
    mO: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    scheduler_params: GDNTileSchedulerParams,
):
    bidx, bidy, bidz = cute.arch.block_idx()
    grid_dim = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
    warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
    warpgroup_tidx = tidx % warpgroup_threads
    lane_id = tidx % warp_threads
    thr_kmn = tiled_mma_kmn.get_slice(0)
    thr_mn = tiled_mma_mn_mn.get_slice(0)
    dice = (None, None, None)

    cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
    tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
    tm_ws = cute.make_tensor(
        tensormap_workspace.iterator,
        cute.make_layout(
            (
                grid_dim[0] * grid_dim[1] * grid_dim[2],
                NUM_TENSORMAPS,
                BYTES_PER_TENSORMAP,
            ),
            stride=(NUM_TENSORMAPS * BYTES_PER_TENSORMAP, BYTES_PER_TENSORMAP, 1),
        ),
    )
    tm_a_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 0, None)].iterator
    )
    tm_b_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 1, None)].iterator
    )
    tm_v_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 2, None)].iterator
    )
    tm_q_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 3, None)].iterator
    )
    tm_aqc_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 4, None)].iterator
    )
    tm_kg_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 5, None)].iterator
    )
    tm_o_ptr = tensormap_manager.get_tensormap_ptr(
        tm_ws[(cta_linear_idx, 6, None)].iterator
    )

    smem = cutlass.utils.SmemAllocator()
    AL = 128
    sA = smem.allocate_tensor(mma_dtype, a_sl.outer, AL, a_sl.inner)
    sB = smem.allocate_tensor(mma_dtype, b_sl.outer, AL, b_sl.inner)
    sV = smem.allocate_tensor(mma_dtype, v_sl.outer, AL, v_sl.inner)
    sST = smem.allocate_tensor(mma_dtype, s_sl.outer, AL, s_sl.inner)
    sQ = smem.allocate_tensor(mma_dtype, q_sl.outer, AL, q_sl.inner)
    sAQC = smem.allocate_tensor(mma_dtype, aqc_sl.outer, AL, aqc_sl.inner)
    sKG = smem.allocate_tensor(mma_dtype, kg_sl.outer, AL, kg_sl.inner)
    sNV = smem.allocate_tensor(mma_dtype, readout_k_sl.outer, AL, readout_k_sl.inner)
    sO = smem.allocate_tensor(mma_dtype, readout_k_sl.outer, AL, readout_k_sl.inner)
    sO_out = smem.allocate_tensor(mma_dtype, readout_k_sl.outer, AL, readout_k_sl.inner)
    sGk_buf = smem.allocate_array(acc_dtype, N)

    sNV_b = cute.make_tensor(
        cute.recast_ptr(sNV.iterator, nv_b_sl.inner, mma_dtype), nv_b_sl.outer
    )
    sNV_a = cute.make_tensor(
        cute.recast_ptr(sNV.iterator, nv_a_sl.inner, mma_dtype), nv_a_sl.outer
    )
    sKG_a = cute.make_tensor(
        cute.recast_ptr(sKG.iterator, kg_a_sl.inner, mma_dtype), kg_a_sl.outer
    )
    sNV_b_mn = cute.make_tensor(
        cute.recast_ptr(sNV.iterator, nv_b_mn_sl.inner, mma_dtype), nv_b_mn_sl.outer
    )
    sO_st = cute.make_tensor(
        cute.recast_ptr(sO.iterator, store_sl.inner, out_dtype), store_sl.outer
    )
    sO_out_st = cute.make_tensor(
        cute.recast_ptr(sO_out.iterator, store_sl.inner, out_dtype), store_sl.outer
    )

    tmem_smem = smem.allocate_array(cutlass.Int32, 1)
    if warp_idx == 0:
        cute.arch.alloc_tmem(512, tmem_smem)

    sNV_ready_nbar = pipeline.NamedBarrier(3, warpgroup_threads + warp_threads)
    sW_ready_nbar = pipeline.NamedBarrier(2, warpgroup_threads + warp_threads)
    gk_load_nbar = pipeline.NamedBarrier(5, warpgroup_threads)

    elect_one = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    wg_coop = pipeline.CooperativeGroup(pipeline.Agent.Thread, warpgroup_threads)
    warp_coop = pipeline.CooperativeGroup(pipeline.Agent.Thread, warp_threads)

    mma6_done_mbar = smem.allocate_array(cutlass.Int64, 1)
    gmem_done_mbar = smem.allocate_array(cutlass.Int64, 1)
    state_ready_mbar = smem.allocate_array(cutlass.Int64, 1)
    final_state_done_mbar = smem.allocate_array(cutlass.Int64, 1)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mma6_done_mbar, warp_threads)
            cute.arch.mbarrier_init(gmem_done_mbar, warpgroup_threads)
            cute.arch.mbarrier_init(state_ready_mbar, warpgroup_threads)
            cute.arch.mbarrier_init(final_state_done_mbar, warpgroup_threads)

    def _make_tma_pipe(total_byte_count, num_stages=1):
        ptr = smem.allocate_array(cutlass.Int64, 2 * num_stages)
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=ptr,
            num_stages=num_stages,
            producer_group=elect_one,
            consumer_group=elect_one,
            tx_count=total_byte_count // num_stages,
            defer_sync=True,
        ).make_participants()

    b_prod, b_cons = _make_tma_pipe(cute.size_in_bytes(mma_dtype, b_sl), num_stages=1)
    v_prod, v_cons = _make_tma_pipe(cute.size_in_bytes(mma_dtype, v_sl), num_stages=1)
    q_prod, q_cons = _make_tma_pipe(cute.size_in_bytes(mma_dtype, q_sl), num_stages=1)
    a_prod, a_cons = _make_tma_pipe(cute.size_in_bytes(mma_dtype, a_sl), num_stages=1)
    aqc_prod, aqc_cons = _make_tma_pipe(
        cute.size_in_bytes(mma_dtype, aqc_sl), num_stages=1
    )
    kg_prod, kg_cons = _make_tma_pipe(
        cute.size_in_bytes(mma_dtype, kg_sl), num_stages=1
    )

    def _make_umma_pipe():
        ptr = smem.allocate_array(cutlass.Int64, 2)
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=ptr,
            num_stages=1,
            producer_group=elect_one,
            consumer_group=wg_coop,
            defer_sync=True,
        ).make_participants()

    w_prod, w_cons = _make_umma_pipe()
    nv_prod, nv_cons = _make_umma_pipe()
    o_prod, o_cons = _make_umma_pipe()

    o_store_mbar = smem.allocate_array(cutlass.Int64, 2)
    o_store_prod, o_store_cons = pipeline.PipelineAsync.create(
        barrier_storage=o_store_mbar,
        num_stages=1,
        producer_group=wg_coop,
        consumer_group=warp_coop,
        defer_sync=True,
    ).make_participants()

    cute.arch.sync_threads()

    tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_smem)

    tCtW_shape = tiled_mma_kmn.partition_shape_C((M, N))
    tCtW_fake = tiled_mma_kmn.make_fragment_C(tCtW_shape)
    tCtW = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 0, dtype=acc_dtype), tCtW_fake.layout
    )

    tCtNV_shape = tiled_mma_kmn.partition_shape_C((M, N))
    tCtNV_fake = tiled_mma_kmn.make_fragment_C(tCtNV_shape)
    tCtNV = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 128, dtype=acc_dtype), tCtNV_fake.layout
    )
    tCtO = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 384, dtype=acc_dtype), tCtNV_fake.layout
    )

    tCtS_shape = tiled_mma_mn_mn.partition_shape_C((M6, N6))
    tCtS_fake = tiled_mma_mn_mn.make_fragment_C(tCtS_shape)
    tCtState = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 256, dtype=acc_dtype), tCtS_fake.layout
    )

    mc = (0, 0, 0, 0)
    ml = cute.make_layout((1, 1, 1, 1))

    # PDL: setup is done; now wait for upstream akk_inv to commit before reading
    # gmem inputs (TMA loads, gk_last_exp reads). PDL allows the entire setup phase
    # above (~tmem alloc, smem layout, pipeline init) to overlap with akk_inv's tail.
    cute.arch.griddepcontrol_wait()

    # ==== WG1: State readout + decay ====
    if warpgroup_idx == STATE_WG:
        cId_128 = cute.make_identity_tensor((M6, N6))
        tCtState_mn = transform_partitioned_tensor_layout(tCtState)

        atom_state_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), acc_dtype
        )
        tiled_state_t2r = tcgen05.make_tmem_copy(
            atom_state_t2r, tCtState[(None, None), 0, 0]
        )
        thr_state_t2r = tiled_state_t2r.get_slice(warpgroup_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn)
        tTR_tCcState = thr_state_t2r.partition_D(cId_128)
        tRrState = cute.make_rmem_tensor_like(tTR_tCcState, acc_dtype)

        atom_state_r2t = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), acc_dtype
        )
        tiled_state_r2t = tcgen05.make_tmem_copy(
            atom_state_r2t, tCtState[(None, None), 0, 0]
        )
        thr_state_r2t = tiled_state_r2t.get_slice(warpgroup_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn)

        atom_state_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), acc_dtype, num_bits_per_copy=128
        )
        tiled_state_g2r = cute.make_tiled_copy_S(atom_state_g2r, tiled_state_r2t)
        thr_state_g2r = tiled_state_g2r.get_slice(warpgroup_tidx)

        atom_state_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), acc_dtype, num_bits_per_copy=128
        )
        tiled_state_r2g = cute.make_tiled_copy_D(atom_state_r2g, tiled_state_t2r)
        thr_state_r2g = tiled_state_r2g.get_slice(warpgroup_tidx)

        atom_state_r2s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mma_dtype, num_bits_per_copy=128
        )
        tiled_state_r2s = cute.make_tiled_copy_D(atom_state_r2s, tiled_state_t2r)
        thr_state_r2s = tiled_state_r2s.get_slice(warpgroup_tidx)
        sST_vk_view = transform_partitioned_tensor_layout(sST)
        sST_kv_view = cute.make_tensor(
            sST.iterator, cute.select(sST_vk_view.layout, mode=[1, 0, 2])
        )
        tCsState_inp = thr_state_r2s.partition_D(sST_kv_view)
        tRrState_bf16 = cute.make_rmem_tensor_like(tTR_tCcState, mma_dtype)
        tCrState_bf16 = tiled_state_r2s.retile(tRrState_bf16)

        sGk = cute.make_tensor(sGk_buf, cute.make_layout(((N,), (1,))))

        scheduler = GDNTileScheduler.create(
            scheduler_params, (bidx, bidy, bidz), grid_dim
        )
        work = scheduler.initial_work_tile_info()
        global_chunk = Int32(0)

        while work.is_valid_tile:
            batch_idx, head_idx, _ = work.tile_idx
            batch_start = cu_seqlens[batch_idx]
            batch_end = cu_seqlens[batch_idx + 1]
            num_chunks = cute.ceil_div(batch_end - batch_start, M)
            # chunk_base from cumulative chunk-offsets array (handles non-64-aligned varlen).
            # batch_start // M only works when all seq lengths are multiples of M.
            chunk_base = chunk_offsets[batch_idx]

            gS_init = cute.flat_divide(
                mS_fp32[batch_idx, head_idx, None, None], (M6, N6)
            )[None, None, 0, 0]
            tGR_tCgState_in = thr_state_g2r.partition_S(gS_init)
            tGR_tCrState_in = thr_state_g2r.retile(tRrState)
            cute.copy(tiled_state_g2r, tGR_tCgState_in, tGR_tCrState_in)
            num_state_subs = tRrState.shape[2]
            for sub in cutlass.range(num_state_subs):
                cute.copy(
                    tiled_state_r2t, tRrState[None, 0, sub], tRT_tCtState[None, 0, sub]
                )
            cute.arch.fence_view_async_tmem_store()

            sGk[warpgroup_tidx] = cutlass.Float32(
                mGkLastExp[chunk_base, head_idx, warpgroup_tidx]
            )
            gk_load_nbar.arrive_and_wait()

            for chunk_c in cutlass.range(num_chunks):
                num_state_subs = tRrState.shape[2]
                _sub_tile_size = cute.size(tRrState.shape[0])

                if chunk_c > 0:
                    cute.arch.mbarrier_wait(
                        mma6_done_mbar, phase=(global_chunk - 1) % 2
                    )
                    gk_load_nbar.arrive_and_wait()

                cute.copy(tiled_state_t2r, tTR_tCtState, tRrState)
                tRrState_bf16.store(tRrState.load().to(mma_dtype))
                cute.copy(
                    tiled_state_r2s, tCrState_bf16, tCsState_inp[None, None, None, 0]
                )
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(gmem_done_mbar)

                for sub in cutlass.range(num_state_subs):
                    cute.copy(
                        tiled_state_t2r,
                        tTR_tCtState[None, 0, sub],
                        tRrState[None, 0, sub],
                    )
                cute.arch.fence_view_async_tmem_load()
                for sub in cutlass.range(num_state_subs):
                    for i in cutlass.range(_sub_tile_size):
                        coord = tTR_tCcState[i, 0, sub]
                        # FIX: probe shows kernel state TMEM[m=K, n=V] (m corresponds to K-dim);
                        # OLD K4 / fla wants K-axis decay (state[k, v] *= gk[k]) → use coord[0]=m=K-idx.
                        k_idx = coord[0]
                        gk_val = cutlass.Float32(sGk[k_idx])
                        tRrState[i, 0, sub] = tRrState[i, 0, sub] * gk_val
                    cute.copy(
                        tiled_state_r2t,
                        tRrState[None, 0, sub],
                        tRT_tCtState[None, 0, sub],
                    )
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(state_ready_mbar)

                if chunk_c + 1 < num_chunks:
                    sGk[warpgroup_tidx] = cutlass.Float32(
                        mGkLastExp[chunk_base + chunk_c + 1, head_idx, warpgroup_tidx]
                    )
                global_chunk = global_chunk + 1

            cute.arch.mbarrier_wait(mma6_done_mbar, phase=(global_chunk - 1) % 2)
            gS_out = cute.flat_divide(
                mS_fp32[batch_idx, head_idx, None, None], (M6, N6)
            )[None, None, 0, 0]
            tGR_tCgState_out = thr_state_r2g.partition_D(gS_out)
            tGR_tCrState_out = thr_state_r2g.retile(tRrState)
            num_state_subs_final = tRrState.shape[2]
            for sub in cutlass.range(num_state_subs_final):
                cute.copy(
                    tiled_state_t2r, tTR_tCtState[None, 0, sub], tRrState[None, 0, sub]
                )
            cute.arch.fence_view_async_tmem_load()
            for sub in cutlass.range(num_state_subs_final):
                cute.copy(
                    tiled_state_r2g,
                    tGR_tCrState_out[None, 0, sub],
                    tGR_tCgState_out[None, 0, sub],
                )

            cute.arch.mbarrier_arrive(final_state_done_mbar)

            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

    # ==== MMA warp (warp 0): 6 MMAs ====
    elif warp_idx == MMA_WARP:
        fA_kmn = thr_kmn.make_fragment_A(sA)
        fB_ks = thr_kmn.make_fragment_B(sB)
        fB_v = thr_kmn.make_fragment_B(sV)
        fB_s = thr_kmn.make_fragment_B(sST)
        fA_q = thr_kmn.make_fragment_A(sQ)
        fA_aqc = thr_kmn.make_fragment_A(sAQC)
        fB_nv = thr_kmn.make_fragment_B(sNV_b)
        fA_kg = thr_mn.make_fragment_A(sKG_a)
        fB_nv_mn = thr_mn.make_fragment_B(sNV_b_mn)
        fA_w = thr_kmn.make_fragment_A(sO)

        scheduler = GDNTileScheduler.create(
            scheduler_params, (bidx, bidy, bidz), grid_dim
        )
        work = scheduler.initial_work_tile_info()
        global_chunk = Int32(0)
        tile_count = Int32(0)

        while work.is_valid_tile:
            batch_idx_m, head_idx_m, _ = work.tile_idx
            num_chunks_m = cute.ceil_div(
                cu_seqlens[batch_idx_m + 1] - cu_seqlens[batch_idx_m], M
            )

            for chunk_c in cutlass.range(num_chunks_m):
                c_phase = global_chunk % 2

                ah = a_cons.wait_and_advance()
                bh = b_cons.wait_and_advance()
                w_h = w_prod.acquire_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(cute.size(sB.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtW,
                        fA_kmn[dice + (0,)][None, None, k],
                        fB_ks[dice + (bh.index,)][None, None, k],
                        tCtW,
                    )
                    if k == 0:
                        tiled_mma_kmn.set(Field.ACCUMULATE, True)
                bh.release()
                w_h.commit()

                vh = v_cons.wait_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(cute.size(sV.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtNV,
                        fA_kmn[dice + (0,)][None, None, k],
                        fB_v[dice + (vh.index,)][None, None, k],
                        tCtNV,
                    )
                    if k == 0:
                        tiled_mma_kmn.set(Field.ACCUMULATE, True)
                vh.release()
                ah.release()

                cute.arch.mbarrier_wait(gmem_done_mbar, phase=c_phase)
                sW_ready_nbar.arrive_and_wait()

                nv_h = nv_prod.acquire_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(cute.size(sST.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtNV,
                        fA_w[dice + (0,)][None, None, k],
                        fB_s[dice + (0,)][None, None, k],
                        tCtNV,
                    )
                nv_h.commit()

                qh = q_cons.wait_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(cute.size(sST.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtO,
                        fA_q[dice + (qh.index,)][None, None, k],
                        fB_s[dice + (0,)][None, None, k],
                        tCtO,
                    )
                    if k == 0:
                        tiled_mma_kmn.set(Field.ACCUMULATE, True)
                qh.release()

                sNV_ready_nbar.arrive_and_wait()

                aqch = aqc_cons.wait_and_advance()
                o_h = o_prod.acquire_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(cute.size(sNV_b.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtO,
                        fA_aqc[dice + (0,)][None, None, k],
                        fB_nv[dice + (0,)][None, None, k],
                        tCtO,
                    )
                o_h.commit()
                aqch.release()

                cute.arch.mbarrier_wait(state_ready_mbar, phase=c_phase)
                kgh = kg_cons.wait_and_advance()
                tiled_mma_mn_mn.set(Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(cute.size(sKG_a.shape[2])):
                    cute.gemm(
                        tiled_mma_mn_mn,
                        tCtState,
                        fA_kg[dice + (0,)][None, None, k],
                        fB_nv_mn[dice + (0,)][None, None, k],
                        tCtState,
                    )
                kgh.release()

                w_prod.tail()
                nv_prod.tail()
                o_prod.tail()
                tcgen05.commit(mma6_done_mbar)
                global_chunk = global_chunk + 1

            cute.arch.mbarrier_wait(final_state_done_mbar, phase=tile_count % 2)
            tile_count = tile_count + 1

            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(tmem_ptr, 512)

    # ==== TMA warp (warp 2) ====
    elif warp_idx == TMA_WARP:
        cta_layout = cute.make_layout(1)

        scheduler = GDNTileScheduler.create(
            scheduler_params, (bidx, bidy, bidz), grid_dim
        )
        work = scheduler.initial_work_tile_info()

        if work.is_valid_tile:
            tensormap_manager.init_tensormap_from_atom(tma_a[0], tm_a_ptr, TMA_WARP)
            tensormap_manager.init_tensormap_from_atom(tma_b[0], tm_b_ptr, TMA_WARP)
            tensormap_manager.init_tensormap_from_atom(tma_v[0], tm_v_ptr, TMA_WARP)
            tensormap_manager.init_tensormap_from_atom(tma_q[0], tm_q_ptr, TMA_WARP)
            tensormap_manager.init_tensormap_from_atom(tma_aqc[0], tm_aqc_ptr, TMA_WARP)
            tensormap_manager.init_tensormap_from_atom(tma_kg[0], tm_kg_ptr, TMA_WARP)
            tensormap_manager.fence_tensormap_initialization()

        while work.is_valid_tile:
            batch_idx_t, head_idx_t, _ = work.tile_idx
            batch_start_t = cu_seqlens[batch_idx_t]
            batch_end_t = cu_seqlens[batch_idx_t + 1]
            num_chunks_t = cute.ceil_div(batch_end_t - batch_start_t, M)

            bounded_a = cute.make_tensor(
                mA.iterator,
                cute.make_layout(
                    (batch_end_t, mA.shape[1], mA.shape[2]),
                    stride=(mA.stride[0], mA.stride[1], mA.stride[2]),
                ),
            )
            bounded_b = cute.make_tensor(
                mB.iterator,
                cute.make_layout(
                    (mB.shape[0], batch_end_t, mB.shape[2]),
                    stride=(mB.stride[0], mB.stride[1], mB.stride[2]),
                ),
            )
            bounded_v = cute.make_tensor(
                mV_g.iterator,
                cute.make_layout(
                    (mV_g.shape[0], batch_end_t, mV_g.shape[2]),
                    stride=(mV_g.stride[0], mV_g.stride[1], mV_g.stride[2]),
                ),
            )
            bounded_q = cute.make_tensor(
                mQ.iterator,
                cute.make_layout(
                    (batch_end_t, mQ.shape[1], mQ.shape[2]),
                    stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                ),
            )
            bounded_aqc = cute.make_tensor(
                mAQC.iterator,
                cute.make_layout(
                    (batch_end_t, mAQC.shape[1], mAQC.shape[2]),
                    stride=(mAQC.stride[0], mAQC.stride[1], mAQC.stride[2]),
                ),
            )
            bounded_kg = cute.make_tensor(
                mKG.iterator,
                cute.make_layout(
                    (mKG.shape[0], batch_end_t, mKG.shape[2]),
                    stride=(mKG.stride[0], mKG.stride[1], mKG.stride[2]),
                ),
            )
            tensormap_manager.update_tensormap(
                (bounded_a, bounded_b, bounded_v, bounded_q, bounded_aqc, bounded_kg),
                (tma_a[0], tma_b[0], tma_v[0], tma_q[0], tma_aqc[0], tma_kg[0]),
                (tm_a_ptr, tm_b_ptr, tm_v_ptr, tm_q_ptr, tm_aqc_ptr, tm_kg_ptr),
                TMA_WARP,
                (None, None, None, None, None, None),
            )

            for chunk_c in cutlass.range(num_chunks_t):
                chunk_offset = batch_start_t + chunk_c * M

                # AB: A-operand (T, K, H) → domain_offset on (tokens, 0)
                mA_c = cute.domain_offset(
                    (chunk_offset, Int32(0)), tma_a[1][None, None, head_idx_t]
                )
                gA = cute.flat_divide(mA_c, (M, K))
                tCgA = thr_kmn.partition_A(gA)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_a[0],
                    0,
                    cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                ah = a_prod.acquire_and_advance()
                if chunk_c == 0:
                    tensormap_manager.fence_tensormap_update(tm_a_ptr)
                cute.copy(
                    tma_a[0],
                    tAgA[(None, 0, 0)],
                    tAsA[(None, ah.index)],
                    tma_bar_ptr=ah.barrier,
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_a_ptr, cute.AddressSpace.generic
                    ),
                )

                # KS: B-operand (N, T, H) → domain_offset on (0, tokens)
                mB_c = cute.domain_offset(
                    (Int32(0), chunk_offset), tma_b[1][None, None, head_idx_t]
                )
                gB_c = cute.flat_divide(mB_c, (N, K))
                tCgB = thr_kmn.partition_B(gB_c)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_b[0],
                    0,
                    cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                bh = b_prod.acquire_and_advance()
                if chunk_c == 0:
                    tensormap_manager.fence_tensormap_update(tm_b_ptr)
                cute.copy(
                    tma_b[0],
                    tBgB[(None, 0, 0)],
                    tBsB[(None, bh.index)],
                    tma_bar_ptr=bh.barrier,
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_b_ptr, cute.AddressSpace.generic
                    ),
                )

                # V: B-operand (N, T, H) → domain_offset on (0, tokens)
                mV_c = cute.domain_offset(
                    (Int32(0), chunk_offset), tma_v[1][None, None, head_idx_t]
                )
                gV_c = cute.flat_divide(mV_c, (N, K))
                tCgV = thr_kmn.partition_B(gV_c)
                tBsV, tBgV = cpasync.tma_partition(
                    tma_v[0],
                    0,
                    cta_layout,
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tCgV, 0, 3),
                )
                vh = v_prod.acquire_and_advance()
                if chunk_c == 0:
                    tensormap_manager.fence_tensormap_update(tm_v_ptr)
                cute.copy(
                    tma_v[0],
                    tBgV[(None, 0, 0)],
                    tBsV[(None, vh.index)],
                    tma_bar_ptr=vh.barrier,
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_v_ptr, cute.AddressSpace.generic
                    ),
                )

                # QS: A-operand (T, N, H) → domain_offset on (tokens, 0)
                mQ_c = cute.domain_offset(
                    (chunk_offset, Int32(0)), tma_q[1][None, None, head_idx_t]
                )
                gQ_c = cute.flat_divide(mQ_c, (M, K3))
                tCgQ = thr_kmn.partition_A(gQ_c)
                tAsQ, tAgQ = cpasync.tma_partition(
                    tma_q[0],
                    0,
                    cta_layout,
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tCgQ, 0, 3),
                )
                qh = q_prod.acquire_and_advance()
                if chunk_c == 0:
                    tensormap_manager.fence_tensormap_update(tm_q_ptr)
                cute.copy(
                    tma_q[0],
                    tAgQ[(None, 0, 0)],
                    tAsQ[(None, qh.index)],
                    tma_bar_ptr=qh.barrier,
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_q_ptr, cute.AddressSpace.generic
                    ),
                )

                # AQC: A-operand (T, K, H) → domain_offset on (tokens, 0)
                mAQC_c = cute.domain_offset(
                    (chunk_offset, Int32(0)), tma_aqc[1][None, None, head_idx_t]
                )
                gAQC_c = cute.flat_divide(mAQC_c, (M, K))
                tCgAQC = thr_kmn.partition_A(gAQC_c)
                tAsAQC, tAgAQC = cpasync.tma_partition(
                    tma_aqc[0],
                    0,
                    cta_layout,
                    cute.group_modes(sAQC, 0, 3),
                    cute.group_modes(tCgAQC, 0, 3),
                )
                aqch = aqc_prod.acquire_and_advance()
                if chunk_c == 0:
                    tensormap_manager.fence_tensormap_update(tm_aqc_ptr)
                cute.copy(
                    tma_aqc[0],
                    tAgAQC[(None, 0, 0)],
                    tAsAQC[(None, aqch.index)],
                    tma_bar_ptr=aqch.barrier,
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_aqc_ptr, cute.AddressSpace.generic
                    ),
                )

                # KG: B-operand (N6, T, H) → domain_offset on (0, tokens)
                mKG_c = cute.domain_offset(
                    (Int32(0), chunk_offset), tma_kg[1][None, None, head_idx_t]
                )
                gKG_c = cute.flat_divide(mKG_c, (N6, K6))
                tCgKG = thr_mn.partition_B(gKG_c)
                tBsKG, tBgKG = cpasync.tma_partition(
                    tma_kg[0],
                    0,
                    cta_layout,
                    cute.group_modes(sKG, 0, 3),
                    cute.group_modes(tCgKG, 0, 3),
                )
                kgh = kg_prod.acquire_and_advance()
                if chunk_c == 0:
                    tensormap_manager.fence_tensormap_update(tm_kg_ptr)
                cute.copy(
                    tma_kg[0],
                    tBgKG[(None, 0, 0)],
                    tBsKG[(None, kgh.index)],
                    tma_bar_ptr=kgh.barrier,
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_kg_ptr, cute.AddressSpace.generic
                    ),
                )

            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

    # ==== Warp 1: O TMA store ====
    elif warp_idx == O_STORE_WARP:
        cta_layout_o = cute.make_layout(1)

        scheduler = GDNTileScheduler.create(
            scheduler_params, (bidx, bidy, bidz), grid_dim
        )
        work = scheduler.initial_work_tile_info()

        if work.is_valid_tile:
            tensormap_manager.init_tensormap_from_atom(tma_o[0], tm_o_ptr, O_STORE_WARP)
            tensormap_manager.fence_tensormap_initialization()

        while work.is_valid_tile:
            batch_idx_o, head_idx_o, _ = work.tile_idx
            batch_start_o = cu_seqlens[batch_idx_o]
            batch_end_o = cu_seqlens[batch_idx_o + 1]
            num_chunks_o = cute.ceil_div(batch_end_o - batch_start_o, M)

            bounded_o = cute.make_tensor(
                mO.iterator,
                cute.make_layout(
                    (batch_end_o, mO.shape[1], mO.shape[2]),
                    stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                ),
            )
            tensormap_manager.update_tensormap(
                (bounded_o,), (tma_o[0],), (tm_o_ptr,), O_STORE_WARP, (None,)
            )
            tensormap_manager.fence_tensormap_update(tm_o_ptr)

            for chunk_c in cutlass.range(num_chunks_o):
                os_h = o_store_cons.wait_and_advance()
                chunk_offset_o = batch_start_o + chunk_c * M
                mO_c = cute.domain_offset(
                    (chunk_offset_o, Int32(0)), tma_o[1][None, None, head_idx_o]
                )
                gOo = cute.flat_divide(mO_c, (M, N))
                sOt, gOt = cpasync.tma_partition(
                    tma_o[0],
                    0,
                    cta_layout_o,
                    cute.group_modes(sO_out_st, 0, 2),
                    cute.group_modes(gOo, 0, 2),
                )
                cute.copy(
                    tma_o[0],
                    sOt[None],
                    gOt[(None, 0, 0)],
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tm_o_ptr, cute.AddressSpace.generic
                    ),
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                os_h.release()

            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

    # ==== WG2: W/NV/O readout (SS-mode, no Phase 2) ====
    elif warpgroup_idx == READOUT_WG:
        tCtW_mn = transform_partitioned_tensor_layout(tCtW)
        tCtNV_mn = transform_partitioned_tensor_layout(tCtNV)
        tCtO_mn = transform_partitioned_tensor_layout(tCtO)

        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(1)), acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtW[(None, None), 0, 0])
        thr_t2r = tiled_t2r.get_slice(warpgroup_tidx)

        tTR_W = thr_t2r.partition_S(tCtW_mn)
        tTR_NV = thr_t2r.partition_S(tCtNV_mn)
        tTR_O = thr_t2r.partition_S(tCtO_mn)

        atom_r2s_k = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, mma_dtype, acc_dtype, tiled_t2r
        )
        tiled_r2s_k = cute.make_tiled_copy_D(atom_r2s_k, tiled_t2r)
        thr_r2s_k = tiled_r2s_k.get_slice(warpgroup_tidx)
        tCsO = thr_r2s_k.partition_D(transform_partitioned_tensor_layout(sO))
        tCsO_out = thr_r2s_k.partition_D(transform_partitioned_tensor_layout(sO_out))
        tCsNV = thr_r2s_k.partition_D(transform_partitioned_tensor_layout(sNV))

        cId = cute.make_identity_tensor((M, N))
        tTR_cId = thr_t2r.partition_D(cId)

        scheduler = GDNTileScheduler.create(
            scheduler_params, (bidx, bidy, bidz), grid_dim
        )
        work = scheduler.initial_work_tile_info()
        global_chunk = Int32(0)

        while work.is_valid_tile:
            batch_idx_r, head_idx_r, _ = work.tile_idx
            num_chunks_r = cute.ceil_div(
                cu_seqlens[batch_idx_r + 1] - cu_seqlens[batch_idx_r], M
            )

            for chunk_c in cutlass.range(num_chunks_r):
                c_phase = global_chunk % 2
                tRrR = cute.make_rmem_tensor_like(tTR_cId, acc_dtype)
                tRrR_out = cute.make_rmem_tensor_like(tRrR, mma_dtype)
                tCrR_k = tiled_r2s_k.retile(tRrR_out)
                num_subs = tRrR.shape[2]

                wh = w_cons.wait_and_advance()
                for sub in cutlass.range(num_subs):
                    cute.copy(tiled_t2r, tTR_W[None, 0, sub], tRrR[None, 0, sub])
                    tRrR_out[None, 0, sub].store(
                        (-(tRrR[None, 0, sub].load())).to(mma_dtype)
                    )
                    cute.copy(tiled_r2s_k, tCrR_k[None, 0, sub], tCsO[None, 0, sub, 0])
                cute.arch.fence_view_async_tmem_load()
                wh.release()
                cute.arch.fence_view_async_shared()
                sW_ready_nbar.arrive_and_wait()

                nvh = nv_cons.wait_and_advance()
                for sub in cutlass.range(num_subs):
                    cute.copy(tiled_t2r, tTR_NV[None, 0, sub], tRrR[None, 0, sub])
                    tRrR_out[None, 0, sub].store(
                        tRrR[None, 0, sub].load().to(mma_dtype)
                    )
                    cute.copy(tiled_r2s_k, tCrR_k[None, 0, sub], tCsNV[None, 0, sub, 0])
                cute.arch.fence_view_async_tmem_load()
                nvh.release()
                cute.arch.fence_view_async_shared()
                sNV_ready_nbar.arrive_and_wait()

                os_h = o_store_prod.acquire_and_advance()

                oh = o_cons.wait_and_advance()
                for sub in cutlass.range(num_subs):
                    cute.copy(tiled_t2r, tTR_O[None, 0, sub], tRrR[None, 0, sub])
                    tRrR_out[None, 0, sub].store(
                        tRrR[None, 0, sub].load().to(mma_dtype)
                    )
                    cute.copy(
                        tiled_r2s_k, tCrR_k[None, 0, sub], tCsO_out[None, 0, sub, 0]
                    )
                cute.arch.fence_view_async_tmem_load()
                oh.release()
                cute.arch.fence_view_async_shared()
                os_h.commit()
                global_chunk = global_chunk + 1

            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()
        o_store_prod.tail()


def make_host_fn(num_sm=148):
    _num_sm = num_sm

    @cute.jit
    def host_fn(
        a_raw: cute.Tensor,
        b_raw: cute.Tensor,
        v_raw: cute.Tensor,
        q_raw: cute.Tensor,
        aqc_raw: cute.Tensor,
        kg_raw: cute.Tensor,
        o_raw: cute.Tensor,
        gk_last_exp: cute.Tensor,
        s_fp32: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        tm_workspace: cute.Tensor,
        stream: cuda_drv.CUstream,
    ):
        # Raw tensors from from_dlpack have PyTorch layout (T, H, dim)
        # stride (H*dim, dim, 1). Reshape to 3D for TMA:
        #   A-operands: (T, dim, H), stride (dim*H, 1, dim)
        #   B-operands: (dim, T, H), stride (1, dim*H, dim)
        T_tok = a_raw.shape[0]
        H = a_raw.shape[1]

        a = cute.make_tensor(
            a_raw.iterator,
            cute.make_layout(
                (T_tok, a_raw.shape[2], H),
                stride=(a_raw.stride[0], a_raw.stride[2], a_raw.stride[1]),
            ),
        )
        q = cute.make_tensor(
            q_raw.iterator,
            cute.make_layout(
                (T_tok, q_raw.shape[2], H),
                stride=(q_raw.stride[0], q_raw.stride[2], q_raw.stride[1]),
            ),
        )
        aqc = cute.make_tensor(
            aqc_raw.iterator,
            cute.make_layout(
                (T_tok, aqc_raw.shape[2], H),
                stride=(aqc_raw.stride[0], aqc_raw.stride[2], aqc_raw.stride[1]),
            ),
        )
        o_out = cute.make_tensor(
            o_raw.iterator,
            cute.make_layout(
                (T_tok, o_raw.shape[2], H),
                stride=(o_raw.stride[0], o_raw.stride[2], o_raw.stride[1]),
            ),
        )

        b = cute.make_tensor(
            b_raw.iterator,
            cute.make_layout(
                (b_raw.shape[2], T_tok, H),
                stride=(b_raw.stride[2], b_raw.stride[0], b_raw.stride[1]),
            ),
        )
        v = cute.make_tensor(
            v_raw.iterator,
            cute.make_layout(
                (v_raw.shape[2], T_tok, H),
                stride=(v_raw.stride[2], v_raw.stride[0], v_raw.stride[1]),
            ),
        )
        kg = cute.make_tensor(
            kg_raw.iterator,
            cute.make_layout(
                (kg_raw.shape[2], T_tok, H),
                stride=(kg_raw.stride[2], kg_raw.stride[0], kg_raw.stride[1]),
            ),
        )

        tile1 = (M, N, K)
        tile3 = (M, N, K3)
        tile6 = (M6, N6, K6)

        mma_kmn = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            (M, N),
            OperandSource.SMEM,
        )
        mma_mn_mn = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            (M6, N6),
            OperandSource.SMEM,
        )
        sl_a = sm100_utils.make_smem_layout_a(mma_kmn, tile1, mma_dtype, 1)
        sl_b = sm100_utils.make_smem_layout_b(mma_kmn, tile1, mma_dtype, 1)
        sl_v = sm100_utils.make_smem_layout_b(mma_kmn, tile1, mma_dtype, 1)
        sl_s = sm100_utils.make_smem_layout_b(mma_kmn, tile3, mma_dtype, 1)
        sl_q = sm100_utils.make_smem_layout_a(mma_kmn, tile3, mma_dtype, 1)
        sl_aqc = sm100_utils.make_smem_layout_a(mma_kmn, tile1, mma_dtype, 1)
        sl_kg = sm100_utils.make_smem_layout_b(mma_mn_mn, tile6, mma_dtype, 1)

        sl_readout_k = sm100_utils.make_smem_layout_a(mma_kmn, tile3, mma_dtype, 1)
        sl_nv_b = sm100_utils.make_smem_layout_b(mma_kmn, tile1, mma_dtype, 1)
        sl_nv_a = sm100_utils.make_smem_layout_a(mma_mn_mn, tile6, mma_dtype, 1)
        sl_kg_a = sm100_utils.make_smem_layout_a(mma_mn_mn, tile6, mma_dtype, 1)
        sl_nv_b_mn = sm100_utils.make_smem_layout_b(mma_mn_mn, tile6, mma_dtype, 1)

        tma_ld = cpasync.CopyBulkTensorTileG2SOp()

        ta_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld, a, cute.select(sl_a, mode=[0, 1, 2]), tile1, mma_kmn
        )
        ta_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld, b, cute.select(sl_b, mode=[0, 1, 2]), tile1, mma_kmn
        )
        ta_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld, v, cute.select(sl_v, mode=[0, 1, 2]), tile1, mma_kmn
        )
        ta_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld, q, cute.select(sl_q, mode=[0, 1, 2]), tile3, mma_kmn
        )
        ta_aqc = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld, aqc, cute.select(sl_aqc, mode=[0, 1, 2]), tile1, mma_kmn
        )
        ta_kg = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld, kg, cute.select(sl_kg, mode=[0, 1, 2]), tile6, mma_mn_mn
        )
        sk = sm100_utils.get_smem_layout_atom_ab(OperandMajorMode.K, out_dtype, (M, N))
        sl_store = cute.tile_to_shape(
            sm100_utils.make_smem_layout_atom(sk, out_dtype), (M, N), order=(0, 1)
        )
        tma_st = cpasync.CopyBulkTensorTileS2GOp()
        ta_o = cpasync.make_tiled_tma_atom(tma_st, o_out, sl_store, (M, N))

        n_seqs = cu_seqlens.shape[0] - 1
        scheduler_params = GDNTileSchedulerParams(
            num_seqs=n_seqs,
            num_q_heads=H,
            num_v_heads=H,
            is_GQA=False,
            is_persistent=True,
        )
        grid_shape = GDNTileScheduler.get_grid_shape(scheduler_params, _num_sm)

        k4_persistent_kernel(
            mma_kmn,
            mma_mn_mn,
            ta_a,
            sl_a,
            ta_b,
            sl_b,
            ta_v,
            sl_v,
            sl_s,
            ta_q,
            sl_q,
            ta_aqc,
            sl_aqc,
            ta_kg,
            sl_kg,
            ta_o,
            sl_store,
            sl_readout_k,
            sl_nv_b,
            sl_nv_a,
            sl_kg_a,
            sl_nv_b_mn,
            gk_last_exp,
            s_fp32,
            cu_seqlens,
            chunk_offsets,
            a,
            b,
            v,
            q,
            aqc,
            kg,
            o_out,
            tm_workspace,
            scheduler_params,
        ).launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            use_pdl=True,
            stream=stream,
        )

    return host_fn


def _compile(host_fn, *args, max_regs=None, keep_cubin=False):
    mr = max_regs if max_regs is not None else MAX_REGS
    if mr > 0:
        opts = f"--maxrregcount={mr} --uumn"
        if keep_cubin:
            return cute.compile[KeepCUBIN, PtxasOptions(opts)](host_fn, *args)
        return cute.compile[PtxasOptions(opts)](host_fn, *args)
    if keep_cubin:
        return cute.compile[KeepCUBIN](host_fn, *args)
    return cute.compile(host_fn, *args)
