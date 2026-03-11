from sglang.srt.layers.attention.fla.gluon import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    get_tmem_reg_layout,
    gl,
    gluon,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)


@gluon.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon(
    k_desc,
    v_desc,
    w_desc,
    v_new_desc,
    g,
    h_desc,
    h0_desc,
    initial_state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    H: gl.constexpr,
    HK: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    INPLACE_UPDATE: gl.constexpr,
    SAVE_NEW_VALUE: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    TRANSPOSE_STATE: gl.constexpr,
):
    i_v, i_nh = gl.program_id(0), gl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_hk = i_h // (H // HK)

    if IS_VARLEN:
        bos, eos = gl.load(cu_seqlens + i_n).to(gl.int32), gl.load(
            cu_seqlens + i_n + 1
        ).to(gl.int32)
        T = eos - bos
        NT = gl.cdiv(T, BT)
        boh = gl.load(chunk_offsets + i_n).to(gl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = gl.cdiv(T, BT)
        boh = i_n * NT

    index = gl.load(initial_state_indices + i_n).to(gl.int32)
    NUM_WARPS: gl.constexpr = gl.num_warps()

    # Allocate shared memory for TMA loads
    dtype: gl.constexpr = k_desc.dtype
    # k, w: [1, BT, 1, BK]
    k_smem = gl.allocate_shared_memory(dtype, k_desc.block_type.shape, k_desc.layout)
    w_smem = gl.allocate_shared_memory(dtype, w_desc.block_type.shape, w_desc.layout)
    # v: [1, BT, 1, BV]
    v_smem = gl.allocate_shared_memory(dtype, v_desc.block_type.shape, v_desc.layout)
    # h0/ht: [1, 1, BK, BV] or [1, 1, BV, BK] if TRANSPOSE_STATE, dtype=fp32 (4D TMA)
    h0_smem = (
        gl.allocate_shared_memory(gl.float32, h0_desc.block_type.shape, h0_desc.layout)
        if USE_INITIAL_STATE
        else None
    )
    # h: [1, 1, 1, BK, BV], dtype=bf16/fp16 (5D TMA)
    h_smem = gl.allocate_shared_memory(dtype, h_desc.block_type.shape, h_desc.layout)

    # For varlen: use scatter layout [BT, BV]; for non-varlen: use [1, BT, 1, BV]
    if SAVE_NEW_VALUE:
        if IS_VARLEN:
            offsets_layout: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1, 4], [32, 1], [1, NUM_WARPS], [1, 0])
            )
            v_new_scatter_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
                [BT, BV], dtype
            )
            v_new_smem = gl.allocate_shared_memory(
                dtype, [BT, BV], v_new_scatter_layout
            )
        else:
            v_new_smem = gl.allocate_shared_memory(
                dtype, v_new_desc.block_type.shape, v_new_desc.layout
            )
    else:
        v_new_smem = None

    # Allocate mbarriers for TMA synchronization
    tma_bar_k = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar_k, count=1)
    tma_bar_w = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar_w, count=1)
    tma_bar_v = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar_v, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    tma_phase_k = 0
    tma_phase_w = 0
    tma_phase_v = 0
    mma_phase = 0

    # Tensor memory layout for accumulation
    v_tmem_layout: gl.constexpr = TensorMemoryLayout([BT, BV], col_stride=1)
    h_tmem_layout: gl.constexpr = TensorMemoryLayout([BK, BV], col_stride=1)
    v_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, [BT, BV], v_tmem_layout, NUM_WARPS
    )
    v_reg_layout_16: gl.constexpr = get_tmem_reg_layout(
        dtype, [BT, BV], v_tmem_layout, NUM_WARPS
    )
    h_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, [BK, BV], h_tmem_layout, NUM_WARPS
    )
    g_layout_bt: gl.constexpr = gl.SliceLayout(dim=1, parent=v_reg_layout)

    # Allocate tensor memory for MMA operations
    v_tmem = allocate_tensor_memory(gl.float32, [BT, BV], v_tmem_layout)
    kv_tmem = allocate_tensor_memory(gl.float32, [BK, BV], h_tmem_layout)

    # Initialize h accumulators
    b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=h_reg_layout)

    # Prologue: prefetch w[0] early (overlap with h0 load + transpose)
    mbarrier.expect(tma_bar_w, w_desc.block_type.nbytes)
    if IS_VARLEN:
        tma.async_copy_global_to_shared(w_desc, [0, bos, i_h, 0], tma_bar_w, w_smem)
    else:
        tma.async_copy_global_to_shared(w_desc, [i_n, 0, i_h, 0], tma_bar_w, w_smem)

    # Load initial state
    if USE_INITIAL_STATE:
        tma_bar_h0 = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(tma_bar_h0, count=1)
        tma_phase_h0 = 0
        mbarrier.expect(tma_bar_h0, h0_desc.block_type.nbytes)
        if TRANSPOSE_STATE:
            tma.async_copy_global_to_shared(
                h0_desc, [index, i_h, i_v * BV, 0], tma_bar_h0, h0_smem
            )
        else:
            tma.async_copy_global_to_shared(
                h0_desc, [index, i_h, 0, i_v * BV], tma_bar_h0, h0_smem
            )
        mbarrier.wait(tma_bar_h0, phase=tma_phase_h0)
        tma_phase_h0 ^= 1
        if TRANSPOSE_STATE:
            # smem permute: load [BV,BK] smem as [BK,BV] via permuted view
            h0_smem_2d = h0_smem.reshape([BV, BK])
            h0_smem_t = h0_smem_2d.permute((1, 0))  # [BK, BV] view
            b_h0 = h0_smem_t.load(h_reg_layout)
        else:
            h0_smem_2d = h0_smem.reshape([BK, BV])
            b_h0 = h0_smem_2d.load(h_reg_layout)
        b_h = b_h + b_h0
        mbarrier.invalidate(tma_bar_h0)

    # Main Loop
    for i_t in range(NT):
        if IS_VARLEN:
            i_b, i_t_h, i_t_kvw = 0, boh + i_t, bos + i_t * BT
        else:
            i_b, i_t_h, i_t_kvw = i_n, i_t, i_t * BT

        # Prefetch v and k early (max overlap with gate + h_store + w_wait + MMA1)
        mbarrier.expect(tma_bar_v, v_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            v_desc, [i_b, i_t_kvw, i_h, i_v * BV], tma_bar_v, v_smem
        )
        mbarrier.expect(tma_bar_k, k_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            k_desc, [i_b, i_t_kvw, i_hk, 0], tma_bar_k, k_smem
        )

        # Compute gating values (scalar ops, overlap with TMA in-flight)
        if USE_G:
            last_idx = T - 1 if i_t == NT - 1 else (i_t + 1) * BT - 1
            bg_last = gl.load(g + (bos + last_idx) * H + i_h)
            g_offset = i_t * BT + gl.arange(0, BT, layout=g_layout_bt)
            g_mask = g_offset < T
            b_g = gl.load(g + (bos + g_offset) * H + i_h, mask=g_mask, other=0)
            bg_last_exp = gl.exp(bg_last)

        if SAVE_NEW_VALUE and IS_VARLEN:
            t_limit_right = gl.minimum(T - i_t * BT, BT)
            t_offsets = gl.arange(0, BT, layout=offsets_layout)
            row_valid = t_offsets < t_limit_right
            x_offsets = gl.where(row_valid, bos + i_t * BT + t_offsets, 0x7FFFFFFF)

        # Store h_i to smem
        h_smem_2d = h_smem.reshape([BK, BV])
        h_smem_2d.store(b_h.to(dtype))

        # Wait for w (prefetched in prologue or previous iteration)
        mbarrier.wait(tma_bar_w, phase=tma_phase_w)
        tma_phase_w ^= 1
        w_smem_2d = w_smem.reshape([BT, BK])

        # TMA store h to global
        fence_async_shared()
        tma.async_copy_shared_to_global(h_desc, [i_b, i_t_h, i_h, 0, i_v * BV], h_smem)
        # w @ h: [BT, BK] @ [BK, BV] -> [BT, BV]
        tcgen05_mma(w_smem_2d, h_smem_2d, v_tmem, use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=mma_phase)
        mma_phase ^= 1

        # Prefetch w for next iteration (w_smem is free after MMA1 completes)
        if i_t < NT - 1:
            mbarrier.expect(tma_bar_w, w_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(
                w_desc, [i_b, i_t_kvw + BT, i_h, 0], tma_bar_w, w_smem
            )

        v_acc_reg = v_tmem.load(v_reg_layout)
        mbarrier.wait(tma_bar_v, phase=tma_phase_v)
        tma_phase_v ^= 1
        v_smem_2d = v_smem.reshape([BT, BV])
        v_reg = v_smem_2d.load(v_reg_layout_16)
        v_new_reg = v_reg - v_acc_reg

        # store v_new to global
        if SAVE_NEW_VALUE:
            if IS_VARLEN:
                v_new_smem.store(v_new_reg.to(dtype))
                fence_async_shared()
                tma.async_scatter(v_new_desc, x_offsets, i_h * V + i_v * BV, v_new_smem)
            else:
                v_new_smem_2d = v_new_smem.reshape([BT, BV])
                v_new_smem_2d.store(v_new_reg.to(dtype))
                fence_async_shared()
                tma.async_copy_shared_to_global(
                    v_new_desc, [i_b, i_t_kvw, i_h, i_v * BV], v_new_smem
                )

        # Apply gating
        if USE_G:
            if i_t == NT - 1:
                v_new_reg = (
                    v_new_reg * gl.where(g_mask, gl.exp(bg_last - b_g), 0)[:, None]
                )
            else:
                v_new_reg = v_new_reg * gl.exp(bg_last - b_g)[:, None]
            b_h *= bg_last_exp

        # Store gated v_new back to v_smem
        v_new_reg = v_new_reg.to(dtype)
        v_smem_2d.store(v_new_reg)

        # Wait for k
        mbarrier.wait(tma_bar_k, phase=tma_phase_k)
        tma_phase_k ^= 1
        k_smem_2d = k_smem.reshape([BT, BK])
        k_t = k_smem_2d.permute((1, 0))

        # fence v
        fence_async_shared()
        # k.T @ v_new -> kv_tmem: [BK, BT] @ [BT, BV] -> [BK, BV]
        tcgen05_mma(k_t, v_smem_2d, kv_tmem, use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=mma_phase)
        mma_phase ^= 1

        # h_i += k_i.T @ v_new
        b_kv = kv_tmem.load(h_reg_layout)
        b_h = b_h + b_kv

    if INPLACE_UPDATE:
        if TRANSPOSE_STATE:
            # smem permute: store [BK,BV] reg to [BV,BK] smem via permuted view
            h0_smem_2d = h0_smem.reshape([BV, BK])
            h0_smem_t = h0_smem_2d.permute((1, 0))  # [BK, BV] view
            h0_smem_t.store(b_h)
            fence_async_shared()
            tma.async_copy_shared_to_global(h0_desc, [index, i_h, i_v * BV, 0], h0_smem)
        else:
            h0_smem_2d = h0_smem.reshape([BK, BV])
            h0_smem_2d.store(b_h)
            fence_async_shared()
            tma.async_copy_shared_to_global(h0_desc, [index, i_h, 0, i_v * BV], h0_smem)

    mbarrier.invalidate(tma_bar_k)
    mbarrier.invalidate(tma_bar_w)
    mbarrier.invalidate(tma_bar_v)
    mbarrier.invalidate(mma_bar)
    tma.store_wait(pendings=0)
