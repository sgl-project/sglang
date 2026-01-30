import triton
import torch

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
def _mask_scalar(A, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return gl.where(mask_i_bit, A, 0.0)

@gluon.jit
def _apply_causal_mask(A, col_limit_right):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # ref https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    offs_n = gl.arange(0, A.shape[1])[None, :]
    s = offs_n & ~0xf
    i = offs_n & 0xf
    return gl.map_elementwise(_mask_scalar, A, col_limit_right, s, i)


@gluon.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_gluon(
    q_desc,
    k_desc,
    v_desc,
    h_desc,
    o_desc,
    g,
    g_gamma,
    cu_seqlens,
    chunk_indices,
    scale,
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
    num_warps: gl.constexpr,
):
    i_v, i_t, i_bh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_hk = i_h // (H // HK)
    i_tg = i_t
    if IS_VARLEN:
        # global chunk id
        # sequence id, chunk id of the current sequence
        i_n, i_t = gl.load(chunk_indices + i_t * 2).to(gl.int32), gl.load(chunk_indices + i_t * 2 + 1).to(gl.int32)
        bos, eos = gl.load(cu_seqlens + i_n).to(gl.int32), gl.load(cu_seqlens + i_n + 1).to(gl.int32)
        T = eos - bos
        # TMA coordinate: qkvo=[0, bos+i_t*BT, ...], h=[0, i_tg, ...]
        i_b, i_t_start = 0, bos + i_t * BT
    else:
        NT = gl.cdiv(T, BT)
        bos, eos = i_b * T, i_b * T + T
        # TMA coordinate: qkvo=[i_b, i_t*BT, ...], h=[i_b, i_t, ...]
        i_b, i_t_start = i_b, i_t * BT
    dtype: gl.constexpr = q_desc.dtype
    q_smem = gl.allocate_shared_memory(dtype, q_desc.block_type.shape, q_desc.layout)
    k_smem = gl.allocate_shared_memory(dtype, k_desc.block_type.shape, k_desc.layout)
    h_smem = gl.allocate_shared_memory(dtype, h_desc.block_type.shape, h_desc.layout)
    tma_bar_qh = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar_qh, count=1)
    tma_bar_kv = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar_kv, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    tma_phase = 0
    mma_phase = 0
    o_tmem_layout: gl.constexpr = TensorMemoryLayout([BT, BV], col_stride=1)
    o_tmem = allocate_tensor_memory(gl.float32, [BT, BV], o_tmem_layout)
    A_tmem_layout: gl.constexpr = TensorMemoryLayout([BT, BT], col_stride=1)
    A_tmem = allocate_tensor_memory(gl.float32, [BT, BT], A_tmem_layout)
    use_acc = False
    for i_k in range(gl.cdiv(K, BK)):
        # Load q and h for o computation
        mbarrier.expect(tma_bar_qh, q_desc.block_type.nbytes + h_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(q_desc, [i_b, i_t_start, i_hk, i_k * BK], tma_bar_qh, q_smem)
        tma.async_copy_global_to_shared(h_desc, [i_b, i_tg, i_h, i_k * BK, i_v * BV], tma_bar_qh, h_smem)
        # Load k for A computation
        mbarrier.expect(tma_bar_kv, k_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(k_desc, [i_b, i_t_start, i_hk, i_k * BK], tma_bar_kv, k_smem)
        # wait qh, compute o = q @ h
        mbarrier.wait(tma_bar_qh, phase=tma_phase)
        q_smem_2d = q_smem.reshape([BT, BK])  # [1, BT, 1, BK] -> [BT, BK]
        h_smem_2d = h_smem.reshape([BK, BV])  # [1, 1, 1, BK, BV] -> [BK, BV]
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        tcgen05_mma(q_smem_2d, h_smem_2d, o_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        # wait k, compute A = q @ k_t
        mbarrier.wait(tma_bar_kv, phase=tma_phase)
        tma_phase ^= 1
        k_t = k_smem.reshape([BT, BK]).permute((1, 0))  # [1, BT, 1, BK] -> [BT, BK] -> [BK, BT]
        mbarrier.wait(mma_bar, phase=mma_phase)
        mma_phase ^= 1
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        tcgen05_mma(q_smem_2d, k_t, A_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        # complete this iteration
        mbarrier.wait(mma_bar, phase=mma_phase)
        mma_phase ^= 1
        use_acc = True
    # async load v
    v_smem = gl.allocate_shared_memory(dtype, v_desc.block_type.shape, v_desc.layout)
    mbarrier.expect(tma_bar_kv, v_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(v_desc, [i_b, i_t_start, i_h, i_v * BV], tma_bar_kv, v_smem)

    o_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        [BT, BV],
        o_tmem_layout,
        num_warps,
    )
    A_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        [BT, BT],
        A_tmem_layout,
        num_warps,
    )
    o_reg = o_tmem.load(o_reg_layout)
    A_reg = A_tmem.load(A_reg_layout)
    if USE_G:
        g_layout_o: gl.constexpr = gl.SliceLayout(dim=1, parent=o_reg_layout)
        # Use the chunk ID of the current sequence.
        g_idx = i_t * BT + gl.arange(0, BT, layout=g_layout_o)
        g_offs = (bos + g_idx) * H + i_h
        b_g = gl.load(g + g_offs, mask=g_idx < T, other=0.0)
        o_reg = o_reg * gl.exp(b_g)[:, None]
        g_layout_A_row: gl.constexpr = gl.SliceLayout(dim=1, parent=A_reg_layout)
        g_layout_A_col: gl.constexpr = gl.SliceLayout(dim=0, parent=A_reg_layout)
        b_g_row = gl.convert_layout(b_g, g_layout_A_row)
        b_g_col = gl.convert_layout(b_g, g_layout_A_col)
        A_reg = A_reg * gl.exp(b_g_row[:, None] - b_g_col[None, :])
    # causal mask
    # col_limit_right[row_idx] indicates the number of visible columns in each row
    # for example: BT=64, col_limit_right = [1, 2, 3, ..., 63, 64]
    # col_limit_right = gl.minimum(gl.arange(0, BT)[:, None] + 1, T - i_t * BT)
    col_limit_right = gl.arange(0, BT)[:, None] + 1
    A_reg = _apply_causal_mask(A_reg, col_limit_right)

    A_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BT, BT], dtype)
    A_smem = gl.allocate_shared_memory(dtype, [BT, BT], A_smem_layout)
    A_smem.store(A_reg.to(dtype))
    # fench A_smem
    fence_async_shared()
    acc_tmem = allocate_tensor_memory(gl.float32, [BT, BV], o_tmem_layout)
    # wait v_sem
    mbarrier.wait(tma_bar_kv, phase=tma_phase)
    mbarrier.invalidate(tma_bar_kv)
    # intra chunk A @ v
    v_smem_2d = v_smem.reshape([BT, BV])  # [1, BT, 1, BV] -> [BT, BV]
    tcgen05_mma(A_smem, v_smem_2d, acc_tmem, use_acc=False)
    tcgen05_commit(mma_bar)
    mbarrier.wait(mma_bar, phase=mma_phase)
    mbarrier.invalidate(mma_bar)
    acc = acc_tmem.load(o_reg_layout)
    o_reg = o_reg * scale + acc * scale
    # store o to global memory
    if IS_VARLEN:
        # for example: T=126, BT=64, i_t=1 → t_limit_right = min(126-64, 64) = min(62, 64) = 62
        t_limit_right = gl.minimum(T - i_t * BT, BT)
        offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
        t_offsets = gl.arange(0, BT, layout=offsets_layout)  # [0, 1, 2, ..., 63]
        mask_o = t_offsets < t_limit_right  # [T, T, ..., T, F, F]
        # use 0x7FFFFFFF(Maximum value of a 32-bit, 2,147,483,647), when out of bounds o.view(T, H * V), TMA skips
        x_offsets = gl.where(mask_o, i_t_start + t_offsets, 0x7FFFFFFF)
        o_smem_2d = gl.allocate_shared_memory(dtype, [BT, BV], o_desc.layout)
        o_smem_2d.store(o_reg.to(dtype))
        fence_async_shared()
        tma.async_scatter(o_desc, x_offsets, i_h * V + i_v * BV, o_smem_2d)
    else:
        o_smem = gl.allocate_shared_memory(dtype, o_desc.block_type.shape, o_desc.layout)
        o_smem_2d = o_smem.reshape([BT, BV])  # [1, BT, 1, BV] -> [BT, BV]
        o_smem_2d.store(o_reg.to(dtype))  # fp32 -> bf16/fp16
        fence_async_shared()
        tma.async_copy_shared_to_global(o_desc, [i_b, i_t_start, i_h, i_v * BV], o_smem)
    tma.store_wait(pendings=0)
