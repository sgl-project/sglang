"""FlashInfer #8854 (56d946ed1, "order TMEM dealloc after the compute warps' tcgen05.ld in the Blackwell GDN
kernels"; after v0.7.3 and v0.7.4rc1, in no release yet) for FlashInfer v0.7.2's Blackwell GDN prefill kernels.
In each kernel one warp frees TMEM while the other warps of its group may still have tcgen05.ld in flight: the
fix has every reading warp drain its TMEM loads (fence_view_async_tmem_load) and meet at a named barrier before
relinquish_alloc_permit / free. Each class below subclasses v0.7.2's and overrides only ``__init__`` (the barrier)
and ``kernel`` (the fence and the barrier), both copied from the v0.7.2 tag (d84335d8c5) with #8854's hunks
applied as is; every other method, the launchers and the dispatch stay FlashInfer's. Outputs are bitwise the
stock kernels' (the change adds a fence and a barrier only). Generated from the tag and #8854's diff by
fi8854-mb/mk_vendor.py; regenerate rather than edit.

This module: the three CP kernels FlashInfer's ``cp_delta_rule_dsl_sm100`` runs (MN precompute, the UTCMMA
fixup, the prefill); ``gdn_prefill_tmem_dealloc.install()`` puts them in place.
"""

import hashlib as _hashlib

from flashinfer.gdn_kernels.blackwell import gated_delta_net_cp as _fi_cp
from flashinfer.gdn_kernels.blackwell import gated_delta_net_cp_prefill as _fi_cp_prefill

FI_CP_SHA256 = "1dccc5f13356a506babe4a4640186933daf22ac2be70d2bca15583025c277c68"
FI_CP_PREFILL_SHA256 = "91bc85a33f9742a0d9f5b4eb256f6cb41e81f17f61172d43d23e85d61c6e0408"


def _file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return _hashlib.sha256(fh.read()).hexdigest()


# The copied methods run against the exact FlashInfer files they were copied from.
if _file_sha256(_fi_cp.__file__) != FI_CP_SHA256:
    raise ImportError(f"{_fi_cp.__file__} is not FlashInfer v0.7.2's")
if _file_sha256(_fi_cp_prefill.__file__) != FI_CP_PREFILL_SHA256:
    raise ImportError(f"{_fi_cp_prefill.__file__} is not FlashInfer v0.7.2's")

# The copied methods resolve their source modules' globals; the two modules share only names bound to the same
# objects (imports), so one namespace serves both.
_cp_names = {k: v for k, v in vars(_fi_cp).items() if not (k.startswith("__") and k.endswith("__"))}
_cpp_names = {k: v for k, v in vars(_fi_cp_prefill).items() if not (k.startswith("__") and k.endswith("__"))}
if any(_cp_names[k] is not _cpp_names[k] for k in _cp_names.keys() & _cpp_names.keys()):
    raise ImportError("FlashInfer's gated_delta_net_cp and gated_delta_net_cp_prefill bind a shared name differently")
globals().update(_cp_names)
globals().update(_cpp_names)


class CPDeltaRuleMNPrecomputeUtcmma1Sm100TmemDealloc(_fi_cp.CPDeltaRuleMNPrecomputeUtcmma1Sm100):
    """``CPDeltaRuleMNPrecomputeUtcmma1Sm100`` (v0.7.2) with FlashInfer #8854's TMEM dealloc ordering."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        cu_seqlens_dtype: Type[cutlass.Numeric] = cutlass.Int32,
    ):
        self.dtype = dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.acc_dtype = cutlass.Float32
        self.BLK = 64
        self.D = 128

        self.k_stage = 3
        self.v_stage = 3
        self.t_stage = 3
        self.alpha_stage = 4
        self.x_stage = 2

        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.transfer_mma_warp_id = 8
        self.tma_warp_id = 9
        self.alpha_warp_id = 10
        self.state_mma_warp_id = 11
        self.threads_per_cta = 384

        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mnk = (1, 1, 1)
        self.tmem_m_offset = 0
        self.tmem_n_offset = 128
        self.tmem_scratch_offset = 256
        self.tmem_m_inp_offset = 320
        self.tmem_n_inp_offset = 384
        self.tmem_x_y_acc_offset = 448

        self.num_regs_compute = 216
        self.num_regs_mma = 72
        self.num_regs_other = 24
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32
            * (
                len(self.compute_group_0_warp_ids)
                + len(self.compute_group_1_warp_ids)
                + 2
            ),
        )
        # All four compute-group-1 warps must have drained their tcgen05.ld
        # before warp 4 deallocates TMEM.
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len(self.compute_group_1_warp_ids)
        )
        self.manual_cache_key(
            "dtype",
            "cu_seqlens_dtype",
            "acc_dtype",
            "BLK",
            "D",
            "k_stage",
            "v_stage",
            "t_stage",
            "alpha_stage",
            "x_stage",
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_x,
        tiled_mma_z,
        tiled_mma_y,
        tiled_mma_m_update,
        tiled_mma_n_update,
        tiled_mma_m_update_ss,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_v,
        tma_tensor_v,
        tma_atom_t,
        tma_tensor_t,
        g_alpha,
        g_transfer_t,
        g_state_t,
        g_cu_seqlens,
        k_layout,
        k_trans_layout,
        v_layout,
        t_layout,
        x_layout,
        alpha_layout,
        cluster_layout_vmnk,
        chunk_len,
        num_k_heads,
        num_v_heads,
        num_sab_heads,
        total_cp_chunks,
        num_seqs,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, seq_idx, _ = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks

        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = cute.ceil_div(valid_chunk_len, self.BLK)
        t_blocks_per_cp_chunk = cute.ceil_div(chunk_len, self.BLK)
        t_block_start = varlen_chunk_idx(
            seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK
        )

        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sK_trans = storage.sK.get_tensor(
            k_trans_layout.outer, swizzle=k_trans_layout.inner
        )
        sV = storage.sV.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sT = storage.sT.get_tensor(t_layout.outer, swizzle=t_layout.inner)
        sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
        sK_tma = self.transform_partitioned_tensor_layout(sK)
        sV_tma = self.transform_partitioned_tensor_layout(sV)
        sT_tma = self.transform_partitioned_tensor_layout(sT)
        sAlpha = storage.sAlpha.get_tensor(alpha_layout)

        def cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        cg_tma = cg(1)
        cg_mma = cg(1)
        cg_mma_both = cg(2)
        cg_alpha = cg(32)
        cg_cg0 = cg(128)
        cg_cg1 = cg(128)
        cg_cg1_v = cg(4)
        cg_both = cg(256)
        load_k_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar.data_ptr(),
            defer_sync=True,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
        )
        load_k_producer = load_k_pipeline.make_producer()
        load_k_transfer_consumer = load_k_pipeline.make_consumer()
        load_k_state_consumer = load_k_pipeline.make_consumer()
        load_t_producer, load_t_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.t_stage,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_t_bytes,
            barrier_storage=storage.load_t_mbar.data_ptr(),
            defer_sync=True,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.v_stage,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar.data_ptr(),
            defer_sync=True,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
            enable_multicast_signaling=False,
        ).make_participants()
        alpha_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.alpha_stage,
            producer_group=cg_alpha,
            consumer_group=cg_both,
            barrier_storage=storage.alpha_mbar.data_ptr(),
            defer_sync=True,
        )
        alpha_producer = alpha_pipeline.make_producer()
        alpha_m_consumer = alpha_pipeline.make_consumer()
        alpha_n_consumer = alpha_pipeline.make_consumer()
        m_init_producer, m_init_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.m_init_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_init_producer, n_init_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.n_init_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        x_acc_producer, x_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.x_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        x_ready_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.x_stage,
            producer_group=cg_cg0,
            consumer_group=cg_mma_both,
            barrier_storage=storage.x_ready_mbar.data_ptr(),
            defer_sync=True,
        )
        x_ready_producer = x_ready_pipeline.make_producer()
        x_ready_m_consumer = x_ready_pipeline.make_consumer()
        x_ready_n_consumer = x_ready_pipeline.make_consumer()
        m_in_producer, m_in_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.m_in_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_in_producer, n_in_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.n_in_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        z_acc_producer, z_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.z_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        z_ready_producer, z_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.z_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        m_acc_producer, m_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.m_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        y_acc_producer, y_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.y_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        y_ready_producer, y_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.y_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_acc_producer, n_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.n_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        done_producer, done_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_cg1,
            barrier_storage=storage.done_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))

        out_layout = cute.make_layout(
            (self.D, self.D, num_sab_heads, total_cp_chunks),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_sab_heads),
        )
        mM = cute.make_tensor(g_transfer_t.iterator, out_layout)
        mN = cute.make_tensor(g_state_t.iterator, out_layout)
        gM = mM[None, None, sab_head_idx, cp_chunk_idx]
        gN = mN[None, None, sab_head_idx, cp_chunk_idx]

        tensor_cta_coord = 0
        tensor_cta_layout = cute.make_layout(1)
        participates_in_tmem = (
            warp_idx <= self.transfer_mma_warp_id or warp_idx == self.state_mma_warp_id
        )

        if not is_valid_chunk:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if participates_in_tmem:
                tmem.wait_for_alloc()
            if warp_idx == self.compute_group_1_warp_ids[0]:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx >= 0 and warp_idx <= 3:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_0(
                tidx,
                tmem_ptr,
                tiled_mma_x,
                tiled_mma_z,
                tiled_mma_m_update,
                sX,
                sAlpha,
                alpha_m_consumer,
                (
                    m_init_producer,
                    x_acc_consumer,
                    x_ready_producer,
                    m_in_producer,
                    z_acc_consumer,
                    z_ready_producer,
                    m_acc_consumer,
                    done_producer,
                ),
                num_blocks,
                gM,
            )
        elif is_valid_chunk and warp_idx >= 4 and warp_idx <= 7:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_1(
                tidx % 128,
                tmem_ptr,
                tiled_mma_x,
                tiled_mma_y,
                tiled_mma_n_update,
                sV_tma,
                sAlpha,
                load_v_consumer,
                alpha_n_consumer,
                (
                    n_init_producer,
                    n_in_producer,
                    y_acc_consumer,
                    y_ready_producer,
                    n_acc_consumer,
                    done_consumer,
                ),
                num_blocks,
                gN,
            )
            # Drain this warp's TMEM loads, then wait for all CG1 warps before
            # warp 4 deallocates the TMEM they were reading from.
            cute.arch.fence_view_async_tmem_load()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx == self.transfer_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_t)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_transfer_mma_role(
                tmem_ptr,
                (tiled_mma_z, tiled_mma_m_update, tiled_mma_m_update_ss),
                (sK, sK_trans, sX),
                load_k_transfer_consumer,
                (
                    m_init_consumer,
                    x_ready_m_consumer,
                    m_in_consumer,
                    z_acc_producer,
                    z_ready_consumer,
                    m_acc_producer,
                ),
                num_blocks,
            )
        elif is_valid_chunk and warp_idx == self.state_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_state_mma_role(
                tmem_ptr,
                (tiled_mma_x, tiled_mma_y, tiled_mma_n_update),
                (sK, sK_trans, sT, sX),
                (load_k_state_consumer, load_t_consumer),
                (
                    n_init_consumer,
                    x_acc_producer,
                    x_ready_n_consumer,
                    n_in_consumer,
                    y_acc_producer,
                    y_ready_consumer,
                    n_acc_producer,
                ),
                num_blocks,
            )
        elif is_valid_chunk and warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            for blk in cutlass.range(num_blocks, unroll=1):
                load_k_producer = self.load_tensor_block_tma_sm100(
                    tma_atom_k,
                    tma_tensor_k,
                    sK_tma,
                    load_k_producer,
                    tok_offset,
                    t_block_start,
                    k_head_idx,
                    blk,
                    False,
                    tensor_cta_coord,
                    tensor_cta_layout,
                )
                load_v_producer = self.load_tensor_block_tma_sm100(
                    tma_atom_v,
                    tma_tensor_v,
                    sV_tma,
                    load_v_producer,
                    tok_offset,
                    t_block_start,
                    v_head_idx,
                    blk,
                    False,
                    tensor_cta_coord,
                    tensor_cta_layout,
                )
                load_t_producer = self.load_tensor_block_tma_sm100(
                    tma_atom_t,
                    tma_tensor_t,
                    sT_tma,
                    load_t_producer,
                    tok_offset,
                    t_block_start,
                    sab_head_idx,
                    blk,
                    True,
                    tensor_cta_coord,
                    tensor_cta_layout,
                )
        elif is_valid_chunk and warp_idx == self.alpha_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            alpha_producer = self.run_load_alpha_role_sm100(
                g_alpha,
                sAlpha,
                alpha_producer,
                tok_offset,
                sab_head_idx,
                valid_chunk_len,
                num_blocks,
                num_sab_heads,
                tidx,
            )
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)


class CPDeltaRuleFixupUtcmmaSm100TmemDealloc(_fi_cp.CPDeltaRuleFixupUtcmmaSm100):
    """``CPDeltaRuleFixupUtcmmaSm100`` (v0.7.2) with FlashInfer #8854's TMEM dealloc ordering."""

    def __init__(
        self,
        rows_per_cta: int = 64,
        needs_initial_state: bool = False,
        store_final_state: bool = False,
        initial_state_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        state_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cu_seqlens_dtype: Type[cutlass.Numeric] = cutlass.Int32,
        use_state_indices: bool = False,
        state_indices_dtype: Type[cutlass.Numeric] | None = None,
        initial_state_inner_strides: tuple[int, ...] | None = None,
        output_state_inner_strides: tuple[int, ...] | None = None,
        store_initial_state: bool = False,
    ):
        self.needs_initial_state = needs_initial_state
        self.store_final_state = store_final_state
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.use_state_indices = use_state_indices
        self.state_indices_dtype = state_indices_dtype
        self.initial_state_inner_strides = initial_state_inner_strides
        self.output_state_inner_strides = output_state_inner_strides
        self.store_initial_state = store_initial_state
        self.acc_dtype = cutlass.Float32
        self.mma_dtype = cutlass.TFloat32
        self.D = 128
        if rows_per_cta not in (64, 128):
            raise ValueError(f"rows_per_cta must be 64 or 128, got {rows_per_cta}")
        self.rows_per_cta = rows_per_cta
        self.row_ctas = self.D // rows_per_cta
        # A second M stage hides the exposed UTC64 TMA load; UTC128 is unchanged by it.
        self.m_stage = 2 if rows_per_cta == 64 else 1
        self.n_stage = 1
        self.threads_per_cta = 256
        self.compute_warp_group_id = 0
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.num_regs_compute = 120 if rows_per_cta == 64 else 256
        self.num_regs_other = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_acc_offset = 0
        self.tmem_opd_offset = 128
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)
        # All four compute warps must have drained their tcgen05.ld before
        # warp 0 deallocates TMEM.
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.manual_cache_key(
            "rows_per_cta",
            "row_ctas",
            "needs_initial_state",
            "store_final_state",
            "initial_state_dtype",
            "state_dtype",
            "cu_seqlens_dtype",
            "use_state_indices",
            "state_indices_dtype",
            "initial_state_inner_strides",
            "output_state_inner_strides",
            "store_initial_state",
            "m_stage",
            "n_stage",
            "num_regs_compute",
            "num_regs_other",
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma,
        tma_atom_m,
        tma_tensor_m,
        tma_atom_n,
        tma_tensor_n,
        g_initial_state_t,
        g_initial_state_workspace_t,
        g_fixed_state_t,
        g_output_state_t,
        g_state_indices_t,
        g_cu_seqlens,
        m_layout,
        n_layout,
        chunk_len,
        total_cp_chunks,
        num_seqs,
        num_heads,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, _, _ = cute.arch.block_idx()
        row_cta_idx = bx % self.row_ctas
        head_seq_idx = bx // self.row_ctas
        head_idx = head_seq_idx % num_heads
        seq_idx = head_seq_idx // num_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        num_chunks = chunks_for_len(seq_end - seq_start, chunk_len)
        chunk_start = varlen_chunk_idx(seq_idx, seq_start, 0, chunk_len)
        start = (
            cutlass.Int32(0)
            if cutlass.const_expr(self.needs_initial_state)
            else cutlass.Int32(1)
        )
        num_iters = num_chunks - start

        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sM = storage.sM.get_tensor(m_layout.outer, swizzle=m_layout.inner)
        sN = storage.sN.get_tensor(n_layout.outer, swizzle=n_layout.inner)
        sM_tma = (
            CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sM)
        )
        sN_tma = (
            CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sN)
        )
        cg_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_mma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_compute = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        cg_compute_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        m_producer, m_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.m_stage,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_m_bytes,
            barrier_storage=storage.m_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_producer, n_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.n_stage,
            producer_group=cg_tma,
            consumer_group=cg_compute_tma,
            tx_count=self.tma_n_bytes,
            barrier_storage=storage.n_mbar.data_ptr(),
            defer_sync=True,
            enable_multicast_signaling=False,
        ).make_participants()
        mma_ready_producer, mma_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_compute,
            consumer_group=cg_mma,
            barrier_storage=storage.mma_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_done_producer, mma_done_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_compute,
            barrier_storage=storage.mma_done_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cute.make_layout((1, 1)), is_relaxed=True
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, 1)))

        out_layout = cute.make_layout(
            (self.D, self.D, num_heads, total_cp_chunks),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        gFixedState = cute.make_tensor(g_fixed_state_t.iterator.align(128), out_layout)
        gFixedStateCta = cute.local_tile(
            gFixedState[None, None, head_idx, None],
            (self.rows_per_cta, self.D, total_cp_chunks),
            (row_cta_idx, 0, 0),
        )
        state_layout = cute.make_layout(
            (self.D, self.D, num_heads, num_seqs),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        state_idx = seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices_t[seq_idx])
        if cutlass.const_expr(self.needs_initial_state):
            if cutlass.const_expr(self.use_state_indices):
                initial_layout = cute.make_layout(
                    (self.D, self.D, num_heads, g_initial_state_t.shape[0]),
                    stride=(
                        g_initial_state_t.stride[2],
                        g_initial_state_t.stride[3],
                        g_initial_state_t.stride[1],
                        g_initial_state_t.stride[0],
                    ),
                )
                gInitialStateTensor = cute.make_tensor(
                    g_initial_state_t.iterator, initial_layout
                )
            else:
                gInitialStateTensor = cute.make_tensor(
                    g_initial_state_t.iterator, state_layout
                )
            gInitialState = cute.local_tile(
                gInitialStateTensor[None, None, head_idx, state_idx],
                (self.rows_per_cta, self.D),
                (row_cta_idx, 0),
            )
            gInitialStateWorkspaceTensor = cute.make_tensor(
                g_initial_state_workspace_t.iterator, state_layout
            )
            gInitialStateWorkspace = cute.local_tile(
                gInitialStateWorkspaceTensor[None, None, head_idx, seq_idx],
                (self.rows_per_cta, self.D),
                (row_cta_idx, 0),
            )
        else:
            gInitialState = gFixedStateCta[None, None, chunk_start]
            gInitialStateWorkspace = gInitialState
        if cutlass.const_expr(self.store_final_state):
            if cutlass.const_expr(self.use_state_indices):
                output_layout = cute.make_layout(
                    (self.D, self.D, num_heads, g_output_state_t.shape[0]),
                    stride=(
                        g_output_state_t.stride[2],
                        g_output_state_t.stride[3],
                        g_output_state_t.stride[1],
                        g_output_state_t.stride[0],
                    ),
                )
                gOutputStateTensor = cute.make_tensor(
                    g_output_state_t.iterator, output_layout
                )
            else:
                gOutputStateTensor = cute.make_tensor(
                    g_output_state_t.iterator, state_layout
                )
            gOutputState = cute.local_tile(
                gOutputStateTensor[None, None, head_idx, state_idx],
                (self.rows_per_cta, self.D),
                (row_cta_idx, 0),
            )
        else:
            gOutputState = gFixedStateCta[None, None, chunk_start]

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
        )
        tmem.allocate(256)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()
        if num_chunks == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if warp_idx == self.compute_warp_group_id:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif (
            warp_idx >= self.compute_warp_group_id
            and warp_idx < self.compute_warp_group_id + 4
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_role(
                tiled_mma,
                tmem_ptr,
                sN_tma,
                n_consumer,
                mma_ready_producer,
                mma_done_consumer,
                gInitialState,
                gInitialStateWorkspace,
                gFixedStateCta,
                gOutputState,
                chunk_start,
                num_chunks,
                tidx,
            )
            # Drain this warp's TMEM loads, then wait for all compute warps
            # before warp 0 deallocates the TMEM they were reading from.
            cute.arch.fence_view_async_tmem_load()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_mma_role(
                tiled_mma,
                tmem_ptr,
                sM,
                m_consumer,
                mma_ready_consumer,
                mma_done_producer,
                num_iters,
                tidx,
            )
        elif warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            cpasync.prefetch_descriptor(tma_atom_m)
            cpasync.prefetch_descriptor(tma_atom_n)
            for chunk_idx in cutlass.range(num_chunks, unroll=1):
                n_producer = self.load_tensor_tma(
                    tma_atom_n,
                    tma_tensor_n,
                    sN_tma,
                    n_producer,
                    head_idx,
                    chunk_start + chunk_idx,
                    row_cta_idx,
                    False,
                )
                if chunk_idx >= start:
                    m_producer = self.load_tensor_tma(
                        tma_atom_m,
                        tma_tensor_m,
                        sM_tma,
                        m_producer,
                        head_idx,
                        chunk_start + chunk_idx,
                        row_cta_idx,
                        True,
                    )
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)


class CPDeltaRulePrefillTcgen05Sm100TmemDealloc(_fi_cp_prefill.CPDeltaRulePrefillTcgen05Sm100):
    """``CPDeltaRulePrefillTcgen05Sm100`` (v0.7.2) with FlashInfer #8854's TMEM dealloc ordering."""

    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        state_dtype: Type[cutlass.Numeric],
        checkpoint_dtype: Type[cutlass.Numeric],
        mma_tiler_qk: Tuple[int, int, int],
        mma_tiler_qs: Tuple[int, int, int],
        mma_tiler_qkv: Tuple[int, int, int],
        mma_tiler_kv: Tuple[int, int, int],
        max_active_clusters: int,
        num_sm: int,
        is_GQA: bool,
        head_ratio: int,
        use_initial_state: bool,
        store_final_state: bool = True,
        enable_checkpoints: bool = False,
        is_persistent: bool = True,
        cu_seqlens_dtype: Type[cutlass.Numeric] = cutlass.Int32,
        checkpoint_cu_starts_dtype: Type[cutlass.Numeric] | None = None,
    ):
        self.io_dtype = io_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.checkpoint_cu_starts_dtype = checkpoint_cu_starts_dtype
        self.acc_dtype = acc_dtype
        self.state_dtype = state_dtype
        self.checkpoint_dtype = checkpoint_dtype
        self.mma_tiler_qk = mma_tiler_qk
        self.mma_tiler_qs = mma_tiler_qs
        self.mma_tiler_qkv = mma_tiler_qkv
        self.mma_tiler_kv = mma_tiler_kv
        self.max_active_clusters = max_active_clusters
        self.num_sm = num_sm
        self.is_GQA = is_GQA
        self.head_ratio = head_ratio
        self.needs_initial_state = use_initial_state
        # Every CP CTA receives an explicit state, either the user-provided
        # initial state, zero, or the fixed-up state from the previous CP chunk.
        self.use_initial_state = True
        self.store_final_state = store_final_state
        self.enable_checkpoints = enable_checkpoints
        self.is_persistent = False

        # ------------------------------------------------------------------
        # Warp assignments  (12 warps total)
        # ------------------------------------------------------------------
        # Precomputed-T transform and QK epilogue.
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # Recurrent state and output epilogues.
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_qkv_warp_id = 9
        # The second issuer owns the five state/output GEMMs.
        self.mma_cg1_warp_id = 10
        # store O
        self.epilogue_warp_id = 11
        # The lightly loaded O epilogue warp also preprocesses gate.
        self.gate_warp_id = self.epilogue_warp_id

        # Give the MMA/TMA/gate/epilogue warps enough registers to keep their
        # pipeline handles resident. Fund the increase from CG1 first while
        # preserving the existing 64,512-register CTA budget.
        self.num_regs_compute_group_0 = 224
        self.num_regs_compute_group_1 = 256
        self.num_regs_other = 24
        if not self.use_initial_state:
            # The peeled zero-state MMA carries more pipeline cursors.  Transfer
            # twenty-four registers from each CG1 warp to each lightweight warp while
            # retaining the same 64,512-register CTA allocation.
            self.num_regs_compute_group_1 = 232
            self.num_regs_other = 48

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_warp_id,
                    self.tma_qkv_warp_id,
                    self.mma_cg1_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        # ------------------------------------------------------------------
        # Named barriers - only TmemAllocator requires a NamedBarrier;
        # all other inter-warp synchronization uses mbarrier-based pipelines
        # created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_warp_id,
                    self.mma_cg1_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                )
            ),
        )
        self.t_store_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )
        self.init_state_store_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )
        # All CG1 warps must have drained their tcgen05.ld before warp 4
        # deallocates TMEM.
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )
        self.manual_cache_key(
            "io_dtype",
            "cu_seqlens_dtype",
            "checkpoint_cu_starts_dtype",
            "acc_dtype",
            "state_dtype",
            "checkpoint_dtype",
            "mma_tiler_qk",
            "mma_tiler_qs",
            "mma_tiler_qkv",
            "mma_tiler_kv",
            "is_GQA",
            "head_ratio",
            "needs_initial_state",
            "store_final_state",
            "enable_checkpoints",
        )

    @cute.kernel
    def kernel(
        self,
        # Tiled MMAs (one per logical GEMM group)
        # QK.
        tiled_mma_qk: cute.TiledMma,
        # K/Q applied to state.
        tiled_mma_qs: cute.TiledMma,
        # New-V correction and intra-block output.
        tiled_mma_qkv: cute.TiledMma,
        # New-V correction with an SMEM V operand.
        tiled_mma_qkv_ss: cute.TiledMma,
        # State update.
        tiled_mma_kv: cute.TiledMma,
        # TMA descriptors and cute tensors
        tma_q: TmaInfo,
        tma_k: TmaInfo,
        tma_v: TmaInfo,
        tma_t: TmaInfo,
        mGate: cute.Tensor,
        tma_o: TmaInfo,
        cu_seqlens: cute.Tensor,
        mFixedState: cute.Tensor,
        mInitialState: Optional[cute.Tensor],
        mStateOut: cute.Tensor,
        mStateCheckpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        cp_chunk_len_fdd: cute.FastDivmodDivisorV2,
        num_sab_heads_fdd: cute.FastDivmodDivisorV2,
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.Layout,
        t_smem_layout_staged: cute.ComposedLayout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        # TMA descriptor workspace in GMEM (one q/k/v/t/o descriptor set per CTA).
        mQ,
        mK,
        mV,
        mT,
        # used for TMA descriptor update
        mO,
        # (num_ctas, num_tensormaps, bytes_per_tensormap) Int8 workspace
        tensormap_workspace: cute.Tensor,
    ):
        """
        Process one explicitly mapped CP chunk for one sequence and SAB head.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.needs_initial_state):
            assert mInitialState is not None, (
                "mInitialState must be provided if needs_initial_state is True"
            )
        else:
            assert mInitialState is None, (
                "mInitialState must be None if needs_initial_state is False"
            )

        (
            head_idx,
            cp_chunk_idx_in_seq,
            chunk_start,
            chunk_len,
            num_cp_chunks,
            cp_chunk_idx,
            t_block_start,
            seq_len,
            seq_token_offset,
        ) = self.get_cp_work(
            cu_seqlens, bidy, bidx, num_sab_heads_fdd, cp_chunk_len_fdd
        )
        # ------------------------------------------------------------------
        # TMA descriptor GMEM workspace - one q/k/v/o descriptor set per CTA.
        # Slots: Q=0, K=1, V=2, T=3, O=4.
        # ------------------------------------------------------------------
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx

        tensormap_manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, self.bytes_per_tensormap
        )

        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 0, None)].iterator
        )
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 1, None)].iterator
        )
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 2, None)].iterator
        )
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 4, None)].iterator
        )

        # ------------------------------------------------------------------
        # 1. Allocate SMEM / TMEM, prefetch TMA descriptors
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sK_trans = storage.sK.get_tensor(
            k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        sT = storage.sT.get_tensor(
            t_smem_layout_staged.outer, swizzle=t_smem_layout_staged.inner
        )
        # A_inverse MMA input.
        sAinv = storage.sAinv.get_tensor(
            ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner
        )
        # QK output / O store  (W_qk first, then O epilogue staging)
        sQk = storage.sQk.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        # Gate scalar arrays (1D Float32, flat - no swizzle)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            cpasync.prefetch_descriptor(tma_t.atom)
            cpasync.prefetch_descriptor(tma_o.atom)

        # TMEM allocator object - CG1 will issue the actual allocation
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # CG1 owns allocation and is the last group to release TMEM state.
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )

        # ------------------------------------------------------------------
        # mbarrier-based pipelines
        # Each pipeline is created by all threads; barrier_storage points into SMEM.
        # defer_sync=True means pipeline_init_arrive() flushes all at once below.
        # ------------------------------------------------------------------
        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        # 1 thread (TMA issuer)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        # 1 warp (gate scalar load warp).
        cg_gate = _cg(self.threads_per_warp * len([self.gate_warp_id]))
        # One producer per result pipeline; K/Q are consumed by both issuers.
        cg_mma = _cg(len([self.mma_warp_id]))
        cg_mma_both = _cg(len([self.mma_warp_id, self.mma_cg1_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # One elected thread per CG0 warp releases each TMA T stage.
        cg_cg0_t = _cg(len(self.compute_group_0_warp_ids))
        # 128 threads (CG1)
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        # 4 threads (one per CG1 warp, used for V load signaling)
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        # 256 threads (CG0 + CG1)
        cg_both = _cg(
            self.threads_per_warp * len(self.compute_group_0_warp_ids)
            + self.threads_per_warp * len(self.compute_group_1_warp_ids)
        )
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # TMA load pipelines: K/Q feed MMA; V is signaled to CG1 for ALU work.
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # Gate producer -> CG0 / CG1 (software-signaled).
        # Scalar-load paths do not use TMA barriers; producer calls commit() after writes.
        load_gate_producer, load_gate_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_gate,
            consumer_group=cg_both,
            barrier_storage=storage.load_gate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_t_producer, load_t_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_t_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg0_t,
            tx_count=self.tma_t_bytes,
            barrier_storage=storage.load_t_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # QK issuer -> CG0 accumulator ring.
        cg0_shared_acc_producer, cg0_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg0_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg0,
                barrier_storage=storage.cg0_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # MMA warp -> CG1: KS/NV accumulator ring.
        cg1_shared_acc_producer, cg1_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg1_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg1,
                barrier_storage=storage.cg1_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG1 -> MMA warp: state input.
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> MMA warp: fixed-slot, ready-only input notifications.
        vks_ready_producer, vks_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.vks_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        nv_ready_producer, nv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.nv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        decay_v_ready_producer, decay_v_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.decay_v_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> epilogue warp:  output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_cg1,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        num_pairs = 2
        num_pairs_b = cute.ceil_div(chunk_len, self.b_t * num_pairs)
        num_chunks_b = num_pairs_b * num_pairs
        num_valid_chunks_b = cute.ceil_div(chunk_len, self.b_t)
        chunk_end = chunk_start + chunk_len

        # COMPUTE WARP GROUP 0 (warps 0-3): T transform and QK epilogue.
        if (
            warp_idx >= self.compute_group_0_warp_ids[0]
            and warp_idx <= self.compute_group_0_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            sQk_pisl = self._transform_to_position_independent_layout(
                sQk, qk_smem_layout_staged.inner
            )
            sAinv_pisl = self._transform_to_position_independent_layout(
                sAinv, ainv_smem_layout_staged.inner
            )
            for pair_idx in cutlass.range(num_pairs_b):
                (
                    load_gate_consumer,
                    load_t_consumer,
                    cg0_shared_acc_consumer,
                    a_inv_ready_producer,
                    qk_ready_producer,
                ) = self.compute_group_0_pair_cp(
                    tidx,
                    tmem_ptr,
                    scale,
                    (tiled_mma_qk,),
                    (sCumsumlog, sT, sAinv_pisl, sQk_pisl),
                    (
                        load_gate_consumer,
                        load_t_consumer,
                        cg0_shared_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                    ),
                    (pair_idx, num_pairs_b, num_valid_chunks_b, chunk_len),
                )
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()

        # COMPUTE WARP GROUP 1 (warps 4-7): recurrent state and output.
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if chunk_len > 0:
                kv_acc_producer = self._load_cp_initial_state(
                    tidx,
                    mFixedState,
                    mInitialState,
                    head_idx,
                    bidy,
                    cp_chunk_idx_in_seq,
                    cp_chunk_idx,
                    tmem_ptr,
                    tiled_mma_kv,
                    kv_acc_producer,
                )
                sV_pisl = self._transform_to_position_independent_layout(
                    sV, v_smem_layout_staged.inner
                )
                sO_pisl = self._transform_to_position_independent_layout(
                    sO, o_smem_layout_staged.inner
                )
                checkpoint_offset = 0
                if cutlass.const_expr(self.enable_checkpoints):
                    checkpoint_offset = cutlass.Int32(checkpoint_cu_starts[bidy])
                    checkpoint_offset += seq_token_offset // checkpoint_every_n_tokens
                is_first_chunk = True
                for chunk_idx in cutlass.range(num_chunks_b):
                    (
                        load_v_consumer,
                        load_gate_consumer,
                        cg1_shared_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        vks_ready_producer,
                        nv_ready_producer,
                        decay_v_ready_producer,
                        o_store_producer,
                        checkpoint_offset,
                    ) = self.compute_group_1_chunk(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (sV_pisl, sCumsumlog, sCumprod, sCumprod, sO_pisl),
                        (
                            mStateCheckpoints,
                            checkpoint_offset,
                            checkpoint_every_n_tokens,
                        ),
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            cg1_shared_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            vks_ready_producer,
                            nv_ready_producer,
                            decay_v_ready_producer,
                            o_store_producer,
                        ),
                        (
                            chunk_idx,
                            num_pairs_b,
                            head_idx,
                            chunk_len,
                            is_first_chunk,
                            seq_token_offset,
                            seq_len,
                        ),
                    )
                    is_first_chunk = False

                store_output_state = cp_chunk_idx_in_seq == num_cp_chunks - 1
                store_cp_boundary = False
                if cutlass.const_expr(self.enable_checkpoints):
                    cp_chunk_end = seq_token_offset + chunk_len
                    store_cp_boundary = cp_chunk_end % checkpoint_every_n_tokens == 0
                # The ending CTA owns its CP-boundary checkpoint.  This avoids
                # racing its main-kernel state against the next CTA's fixed state.
                if store_output_state or store_cp_boundary:
                    kv_acc_consumer = self._store_final_state(
                        tidx,
                        mStateOut,
                        mFixedState,
                        None,
                        head_idx,
                        bidy,
                        cp_chunk_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        kv_acc_consumer,
                        chunk_len,
                        seq_token_offset + chunk_len,
                        mStateCheckpoints,
                        checkpoint_offset,
                        checkpoint_every_n_tokens,
                        store_output_state,
                    )
                else:
                    kv_acc_handle = kv_acc_consumer.wait_and_advance()
                    kv_acc_handle.release()
            # Drain this warp's TMEM loads, then wait for all CG1 warps before
            # warp 4 deallocates the TMEM they were reading from.
            cute.arch.fence_view_async_tmem_load()
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # CG0 MMA ISSUER (warp 8): paired QK.
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            for _pair_idx in cutlass.range(num_pairs_b):
                (
                    cg0_shared_acc_producer,
                    load_k_consumer,
                    load_q_consumer,
                ) = self.mma_cg0_pair_cp(
                    tmem_ptr,
                    (tiled_mma_qk, tiled_mma_qkv),
                    (sQ, sK),
                    (
                        cg0_shared_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                    ),
                )
            cg0_shared_acc_producer.tail()

        # CG1 MMA ISSUER (warp 10): KS/QS/NV/QKV/KV.
        elif warp_idx == self.mma_cg1_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            for chunk_idx in cutlass.range(num_chunks_b):
                (
                    cg1_shared_acc_producer,
                    q_state_acc_producer,
                    kv_acc_producer,
                    load_k_consumer,
                    load_q_consumer,
                    a_inv_ready_consumer,
                    qk_ready_consumer,
                    state_inp_ready_consumer,
                    vks_ready_consumer,
                    nv_ready_consumer,
                    decay_v_ready_consumer,
                ) = self.mma_cg1_chunk(
                    tmem_ptr,
                    (
                        tiled_mma_qs,
                        tiled_mma_qkv,
                        tiled_mma_qkv_ss,
                        tiled_mma_kv,
                    ),
                    (sQ, sK, sK_trans, sV, sAinv, sQk),
                    (
                        cg1_shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        vks_ready_consumer,
                        nv_ready_consumer,
                        decay_v_ready_consumer,
                    ),
                    chunk_idx == 0,
                )
            cg1_shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # TMA LOAD WARP (warp 9): Q/K/V and precomputed T.
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tensormap_manager.init_tensormap_from_atom(
                tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id
            )
            tensormap_manager.fence_tensormap_initialization()
            bounded_q = cute.make_tensor(
                mQ.iterator,
                cute.make_layout(
                    (chunk_end, mQ.shape[1], mQ.shape[2]),
                    stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                ),
            )
            bounded_k = cute.make_tensor(
                mK.iterator,
                cute.make_layout(
                    (chunk_end, mK.shape[1], mK.shape[2]),
                    stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                ),
            )
            bounded_v = cute.make_tensor(
                mV.iterator,
                cute.make_layout(
                    (mV.shape[0], chunk_end, mV.shape[2]),
                    stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                ),
            )
            tensormap_manager.update_tensormap(
                (bounded_q, bounded_k, bounded_v),
                (tma_q.atom, tma_k.atom, tma_v.atom),
                (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                self.tma_qkv_warp_id,
                (None, None, None),
            )
            for chunk_idx in cutlass.range(num_chunks_b):
                chunk_offset = chunk_start + chunk_idx * self.b_t
                load_q_producer, load_k_producer, load_v_producer = self.tma_qkv_warp(
                    (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                    (tma_q, tma_k, tma_v),
                    (sQ, sK, sV),
                    (load_q_producer, load_k_producer, load_v_producer),
                    (chunk_offset, chunk_idx, bidy, head_idx),
                    (
                        tensormap_manager,
                        tensormap_q_ptr,
                        tensormap_k_ptr,
                        tensormap_v_ptr,
                    ),
                )
                t_chunk_idx = chunk_idx
                if chunk_idx >= num_valid_chunks_b:
                    t_chunk_idx = num_valid_chunks_b - 1
                load_t_producer = self.tma_t_warp(
                    tiled_mma_qkv,
                    tma_t,
                    sT,
                    load_t_producer,
                    t_block_start + t_chunk_idx,
                    head_idx,
                )
            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()
            load_t_producer.tail()

        # EPILOGUE WARP (warp 11): gate preprocessing and O store.
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tensormap_manager.init_tensormap_from_atom(
                tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
            )
            tensormap_manager.fence_tensormap_initialization()
            bounded_o = cute.make_tensor(
                mO.iterator,
                cute.make_layout(
                    (mO.shape[0], chunk_end, mO.shape[2]),
                    stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                ),
            )
            tensormap_manager.update_tensormap(
                (bounded_o,),
                (tma_o.atom,),
                (tensormap_o_ptr,),
                self.epilogue_warp_id,
                (None,),
            )
            tensormap_manager.fence_tensormap_update(tensormap_o_ptr)

            if chunk_len > 0:
                num_gate_prefetch = 4
                for prefetch_idx in range(2):
                    load_gate_producer = self.load_gate_warp(
                        tidx,
                        mGate,
                        (sCumsumlog, sCumprod),
                        load_gate_producer,
                        (
                            chunk_start + prefetch_idx * self.b_t,
                            head_idx,
                            prefetch_idx >= num_valid_chunks_b - 1,
                            chunk_end,
                        ),
                    )
                if num_chunks_b > 2:
                    for prefetch_idx in range(2, num_gate_prefetch):
                        load_gate_producer = self.load_gate_warp(
                            tidx,
                            mGate,
                            (sCumsumlog, sCumprod),
                            load_gate_producer,
                            (
                                chunk_start + prefetch_idx * self.b_t,
                                head_idx,
                                prefetch_idx >= num_valid_chunks_b - 1,
                                chunk_end,
                            ),
                        )
                for chunk_idx in cutlass.range(num_chunks_b):
                    prefetch_idx = chunk_idx + num_gate_prefetch
                    if prefetch_idx < num_chunks_b:
                        load_gate_producer = self.load_gate_warp(
                            tidx,
                            mGate,
                            (sCumsumlog, sCumprod),
                            load_gate_producer,
                            (
                                chunk_start + prefetch_idx * self.b_t,
                                head_idx,
                                prefetch_idx >= num_valid_chunks_b - 1,
                                chunk_end,
                            ),
                        )
                    o_store_consumer = self.epilogue_warp(
                        (sO,),
                        (tma_o,),
                        (o_store_consumer,),
                        (head_idx, chunk_start + chunk_idx * self.b_t),
                        (tensormap_manager, tensormap_o_ptr),
                    )
            load_gate_producer.tail()
