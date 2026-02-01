import torch
import triton

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
        get_tmem_reg_layout,
        tma,
        mbarrier,
        tcgen05_mma,
        tcgen05_commit,
        fence_async_shared,
    )
except ImportError as e:
    raise ImportError(
        f">>> Failed to import Gluon in current triton version {triton.__version__} and "
        f">>> Platform {torch.cuda.get_device_capability()}.\n"
        f">>> Gluon/Blackwell features require: \n"
        f">>> 1. Triton >= 3.6.0.\n"
        f">>> 2. NVIDIA GPU (compute capability >= 10.0)\n"
        f">>> Error: {e}\n"
        f">>> Set FLA_USE_GLUON=0 to disable and continue."
    ) from e


@gluon.jit
def recompute_w_u_fwd_kernel_gluon(
    k_desc,
    v_desc,
    w_desc,
    u_desc,
    A_desc,
    beta,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: gl.constexpr,
    HK: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    i_t, i_bh = gl.program_id(0), gl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_hk = i_h // (H // HK)
    i_tg = i_t

    NUM_WARPS: gl.constexpr = gl.num_warps()
    if IS_VARLEN:
        i_n, i_t = gl.load(chunk_indices + i_t * 2).to(gl.int32), gl.load(chunk_indices + i_t * 2 + 1).to(gl.int32)
        bos, eos = gl.load(cu_seqlens + i_n).to(gl.int32), gl.load(cu_seqlens + i_n + 1).to(gl.int32)
        T = eos - bos
        # TMA coordinate: qkvo=[0, bos+i_t*BT, ...], h=[0, i_tg, ...]
        i_b, i_t_start = 0, bos + i_t * BT
    else:
        bos, eos = i_b * T, i_b * T + T
        # TMA coordinate: qkvo=[i_b, i_t*BT, ...], h=[i_b, i_t, ...]
        i_b, i_t_start = i_b, i_t * BT

    dtype: gl.constexpr = k_desc.dtype

    # allocate smem and init mbarrier state
    k_smem = gl.allocate_shared_memory(dtype, k_desc.block_type.shape, k_desc.layout)
    k_tmem_layout: gl.constexpr = TensorMemoryLayout([BT, BK], col_stride=1)
    k_tmem = allocate_tensor_memory(dtype, [BT, BK], k_tmem_layout)
    
    v_smem = gl.allocate_shared_memory(dtype, v_desc.block_type.shape, v_desc.layout)
    v_tmem_layout: gl.constexpr = TensorMemoryLayout([BT, BV], col_stride=1)
    v_tmem = allocate_tensor_memory(v_desc.dtype, [BT, BV], v_tmem_layout)
    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)

    w_tmem = allocate_tensor_memory(gl.float32, [BT, BK], k_tmem_layout)

    # fp32 for accumulator
    u_tmem = allocate_tensor_memory(gl.float32, [BT, BV], v_tmem_layout)

    A_smem = gl.allocate_shared_memory(dtype, A_desc.block_type.shape, A_desc.layout)
    A_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(A_bar, count=1)

    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)

    tma_phase = 0
    mma_phase = 0

    beta_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, NUM_WARPS], [1, 0])
    layout_1d_beta_x: gl.constexpr = gl.SliceLayout(dim=1, parent=beta_layout)
    layout_1d_beta_y: gl.constexpr = gl.SliceLayout(dim=0, parent=beta_layout)
    indices_t = i_t * BT + gl.arange(0, BT, layout=layout_1d_beta_x)
    beta += bos * H + i_h
    mask = indices_t < T
    beta_reg = gl.load(beta + indices_t * H, mask=mask, other=0.0)[:, None]

    mbarrier.expect(A_bar, A_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(A_desc, [i_b, i_t_start, i_h, 0], A_bar, A_smem)

    v_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        (BT, BV),
        v_tmem_layout,
        NUM_WARPS
    )

    if USE_G:
        g += bos * H + i_h
        g_reg = gl.load(g + indices_t * H, mask=mask, other=0.0)
        g_reg = gl.exp(g_reg)[:, None]

    k_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        (BT, BK),
        k_tmem_layout,
        NUM_WARPS
    )
    
    tma_k_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_k_bar, count=1)
    mma_k_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_k_bar, count=1)
    tma_phase_k = 0
    mma_phase_k = 0

    # G -> S
    mbarrier.expect(tma_bar, v_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(v_desc, [i_b, i_t_start, i_h, 0 * BV], tma_bar, v_smem)

    # G -> S
    mbarrier.expect(tma_k_bar, k_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(k_desc, [i_b, i_t_start, i_hk, 0 * BK], tma_k_bar, k_smem)

    NV = gl.cdiv(V, BV)
    for i_v in range(NV):
        mbarrier.wait(tma_bar, phase=tma_phase)
        # v * beta
        v_smem_2d = v_smem.reshape([BT, BV])
        v_reg = v_smem_2d.load(v_reg_layout)
        beta_reg_convert_v = gl.convert_layout(beta_reg, layout=v_reg_layout)
        vb_reg = v_reg * beta_reg_convert_v 
        vb_reg = vb_reg.to(v_desc.dtype)

        # A @ vb
        v_smem_2d.store(vb_reg)
        fence_async_shared()
        if i_v == 0:
            mbarrier.wait(A_bar, phase=0) 
        # [BT, BT] @ [BT, BV] -> [BT, BV]
        A_smem_2d = A_smem.reshape([BT, BT])
        tcgen05_mma(A_smem_2d, v_smem_2d, u_tmem, use_acc=False)
        tcgen05_commit(mma_bar)
        # prefetch next loop
        if i_v < NV - 1:
            mbarrier.expect(tma_bar, v_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(v_desc, [i_b, i_t_start, i_h, (i_v + 1) * BV], tma_bar, v_smem) 
        mbarrier.wait(mma_bar, phase=mma_phase)

        u_reg = u_tmem.load(v_reg_layout)
        if IS_VARLEN:
            t_limit_right = gl.minimum(T - i_t * BT, BT)
            offsets_u_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
            t_offsets = gl.arange(0, BT, layout=offsets_u_layout)
            mask_o = t_offsets < t_limit_right
            x_offsets = gl.where(mask_o, i_t_start + t_offsets, 0x7FFFFFFF)
            u_smem_2d = gl.allocate_shared_memory(dtype, [BT, BV], u_desc.layout)
            u_smem_2d.store(u_reg.to(dtype))
            fence_async_shared()
            tma.async_scatter(u_desc, x_offsets, i_h * V + i_v * BV, u_smem_2d)
        else:
            u_smem = gl.allocate_shared_memory(dtype, u_desc.block_type.shape, u_desc.layout)
            u_smem_2d = u_smem.reshape([BT, BV])
            u_smem_2d.store(u_reg.to(dtype))
            fence_async_shared()
            tma.async_copy_shared_to_global(u_desc, [i_b, i_t_start, i_h, i_v * BV], u_smem)
        # guarantee all tma store ops are completed
        tma.store_wait(pendings=0)

        mbarrier.wait(tma_k_bar, phase=tma_phase_k)
        # k * beta
        k_smem_2d = k_smem.reshape([BT, BK])
        k_reg = k_smem_2d.load(k_reg_layout)
        beta_reg_convert_k = gl.convert_layout(beta_reg, layout=k_reg_layout)
        kb_reg = k_reg * beta_reg_convert_k 

        if USE_G:
            g_reg_convert_k = gl.convert_layout(g_reg, layout=k_reg_layout)
            kb_reg *= g_reg_convert_k
        kb_reg = kb_reg.to(k_desc.dtype)

        k_smem_2d.store(kb_reg)
        fence_async_shared()
        tcgen05_mma(A_smem_2d, k_smem_2d, w_tmem, use_acc=False)
        tcgen05_commit(mma_k_bar)
        # prefetch next loop
        if i_v < NV - 1:
            mbarrier.expect(tma_k_bar, k_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(k_desc, [i_b, i_t_start, i_hk, (i_v + 1) * BK], tma_k_bar, k_smem)
        mbarrier.wait(mma_k_bar, phase=mma_phase_k)

        w_reg = w_tmem.load(k_reg_layout)
        if IS_VARLEN:
            t_limit_right = gl.minimum(T - i_t * BT, BT)
            offsets_w_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
            t_offsets = gl.arange(0, BT, layout=offsets_w_layout)
            mask_o = t_offsets < t_limit_right
            x_offsets = gl.where(mask_o, i_t_start + t_offsets, 0x7FFFFFFF)
            w_smem_2d = gl.allocate_shared_memory(dtype, [BT, BK], w_desc.layout)
            w_smem_2d.store(w_reg.to(dtype))
            fence_async_shared()
            tma.async_scatter(w_desc, x_offsets, i_h * K + i_v * BK, w_smem_2d)
        else:
            w_smem = gl.allocate_shared_memory(dtype, w_desc.block_type.shape, w_desc.layout)
            # store w
            w_smem_2d = w_smem.reshape([BT, BK])
            w_smem_2d.store(w_reg.to(dtype))
            fence_async_shared()
            tma.async_copy_shared_to_global(w_desc, [i_b, i_t_start, i_h, i_v * BK], w_smem)

        tma_phase ^= 1
        mma_phase ^= 1
        tma_phase_k ^= 1
        mma_phase_k ^= 1
    
        # guarantee all tma store ops are completed
        tma.store_wait(pendings=0)

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)
    mbarrier.invalidate(A_bar)
    mbarrier.invalidate(tma_k_bar)
    mbarrier.invalidate(mma_k_bar)
