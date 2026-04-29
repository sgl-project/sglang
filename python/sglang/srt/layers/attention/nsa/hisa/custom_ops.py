import torch
import tilelang
from tilelang import language as T

@tilelang.jit(
        pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def pool_mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
    dtype="bfloat16",
):
    if block_Q is None:
        block_Q = 128 // heads
    accum_dtype = "float32"
    index_dtype = "int32"

    seq_len = T.dynamic("seq_len")
    seq_len_blocked_kv = T.dynamic("seq_len_blocked_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_blocked_kv, index_dim]
    index_k_scale_shape = [seq_len_blocked_kv]
    logits_shape = [seq_len, seq_len_blocked_kv]

    @T.prim_func
    def pool_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexBlockedK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenBlockedKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenBlockedKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenBlockedKS[seq_len_i + bq_i], seq_len_blocked_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenBlockedKE[seq_len_i + bq_i], seq_len_blocked_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexBlockedK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) 

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return pool_mqa_attn_return_logits_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel

@tilelang.jit
def force_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def force_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx == cu_k_s or idx == cu_k_e - 1:
                        Logits[bx, idx] = T.infinity(dtype)

    return force_maintain_logits_kernel

@tilelang.jit
def clean_and_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_and_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx == cu_k_s or idx == cu_k_e - 1:
                        Logits[bx, idx] = T.infinity(dtype)
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_and_maintain_logits_kernel

def pool_mqa_attn_return_logits_interface(q, blocked_kv, kv_block_size, weights, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke, clean_logits=True, force_maintain=True, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_blocked_kv = blocked_kv.shape[0]

    pool_mqa_attn_return_logits_kernel = pool_mqa_attn_return_logits(heads=heads, index_dim=index_dim, dtype=dtype)
    logits = torch.empty([seq_len, seq_len_blocked_kv], device=q.device, dtype=torch.float32)
    pool_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        blocked_kv,
        logits,
        weights,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )
    if clean_logits and force_maintain:
        clean_and_maintain_logits_kernel = clean_and_maintain_logits_()
        clean_and_maintain_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
    else:
        clean_logits_kernel = clean_logits_()
        force_maintain_logits_kernel = force_maintain_logits_()
        if clean_logits:
            clean_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
        if force_maintain:
            force_maintain_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
    return logits

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def block_sparse_mqa_attn_return_logits(
    kv_block_size,
    topk,
    heads,
    index_dim,
    block_N=128,
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk * kv_block_size]

    # TODO check padded H in sparse_mla_fwd
    # does it matter here?
    H_per_block = heads
    block_N = T.min(block_N, kv_block_size)
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"

    @T.prim_func
    def block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[seq_len_i, n_i]
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N

                    T.copy(IndexK[block_s_i, 0], index_k_shared)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                    
                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0] 
    
    @T.prim_func
    def block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[seq_len_i, n_i]
                block_s_i = topk_block_id * kv_block_size

                T.copy(IndexK[block_s_i, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                
                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, n_i * kv_block_size + bn_i] = logits[bn_i, 0] 

    if kv_block_size == block_N:
        return block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size
    else:
        return block_sparse_mqa_attn_return_logits_kernel

def block_sparse_mqa_attn_return_logits_interface(q, kv, topk_block_index, kv_block_size, weights, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    topk = topk_block_index.shape[1]

    block_sparse_mqa_attn_return_logits_kernel = block_sparse_mqa_attn_return_logits(heads=heads, index_dim=index_dim, kv_block_size=kv_block_size, topk=topk)
    logits = torch.empty([seq_len, topk * kv_block_size], device=q.device, dtype=torch.float32)
    block_sparse_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        topk_block_index,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return logits

def ref_block_mean_pooling(k, k_block_size):
    seq_len_k = k.shape[0]
    num_k_blocks = (seq_len_k + k_block_size - 1) // k_block_size
    blocked_k_mean = []
    for i in range(num_k_blocks):
        start_idx = i * k_block_size
        end_idx = min((i + 1) * k_block_size, seq_len_k)
        block_kv = k[start_idx:end_idx, :]
        block_mean = block_kv.mean(dim=0, keepdim=True)  # [block_size, D] -> [1, D]
        blocked_k_mean.append(block_mean)
    # ref_blocked_k = torch.cat(blocked_k_mean, dim=0)  # [num_block, D]
    blocked_k = torch.cat(blocked_k_mean, dim=0) 
    
    return blocked_k

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def block_mean_pooling(
    max_num_pooling_blocks: int,
    pooling_block_size: int, 
    dim: int,
    block_N: int=64,
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    
    seq_len_k = T.dynamic("seq_len_k")
    k_size = [seq_len_k, dim]
    blocked_k_size = [max_num_pooling_blocks, dim]

    @T.prim_func
    def block_mean_pooling_kernel(
        K: T.Tensor(k_size, dtype=dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=accum_dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0) 

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                T.copy(K[tl_block_s:tl_block_s + block_N, :], index_k)

                cur_tl_block_size = tl_block_e - tl_block_s
                for n_i in T.parallel(block_N):
                    for d_i in T.parallel(dim):
                        if n_i >= cur_tl_block_size:
                            index_k[n_i, d_i] = T.cast(0, accum_dtype)

                T.reduce_sum(index_k, acc, dim=0, clear=False)
            
            for d_i in T.parallel(dim):
                acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)
            
            T.copy(acc, BlockedK[bx, :])
    
    return block_mean_pooling_kernel

def block_mean_pooling_interface(k, k_block_size):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=torch.float32)
    kernel = block_mean_pooling(
        max_num_pooling_blocks=max_num_pooling_blocks,
        pooling_block_size=k_block_size,
        dim=d,
    )
    kernel(
        k,
        blocked_k,
    )
    blocked_k = blocked_k.to(k.dtype)

    return blocked_k

def fp8_hierarchy_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    k_block_size: int, 
    block_topk: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    q = q.float()  # [M, H, D]
    k_fp8, k_scales = kv
    k_scales = k_scales.contiguous().view(torch.float32)
    if k_scales.ndim == 1:
        k_scales = k_scales.unsqueeze(-1)  # [N, 1]
    k = k_fp8.float() * k_scales  # [N, D]
    q = q.bfloat16()
    k = k.bfloat16()
    weights = weights.bfloat16()

    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    # TODO: still using torch mean pooling
    # blocked_k = ref_block_mean_pooling(k, k_block_size)  # [num_block, D]
    blocked_k = block_mean_pooling_interface(k, k_block_size)  # [num_block, D]

    block_k_indexer_score = pool_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks, cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)

    topk_block_indices = torch.topk(block_k_indexer_score, k=min(block_topk, block_k_indexer_score.shape[-1]), dim=-1).indices  # [M, topk]
    topk_block_indices = topk_block_indices.to(torch.int32)

    block_sparse_logits = block_sparse_mqa_attn_return_logits_interface(q=q, kv=k, topk_block_index=topk_block_indices, kv_block_size=k_block_size, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)

    return block_sparse_logits, topk_block_indices

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_mean_pooling(
    max_num_pooling_blocks: int,
    pooling_block_size: int,
    dim: int,
    block_N: int=64,
    num_stages=1,
    threads=256,
):
    """Mean-pool with fp8 re-quantization: outputs fp8 BlockedK + f32 BlockedKScale."""
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32

    seq_len_k = T.dynamic("seq_len_k")
    k_size = [seq_len_k, dim]
    scale_size = [seq_len_k]
    blocked_k_size = [max_num_pooling_blocks, dim]
    blocked_k_scale_size = [max_num_pooling_blocks]
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def fp8_native_block_mean_pooling_kernel(
        K: T.Tensor(k_size, dtype=dtype), # type: ignore
        KScale: T.Tensor(scale_size, dtype=accum_dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=dtype), # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_size, dtype=accum_dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                T.copy(K[tl_block_s:tl_block_s + block_N, :], index_k)
                # 1D KScale load via T.Parallel to avoid TMA alignment issue
                # at unaligned seq_len_k base addresses.
                for bn_i in T.Parallel(block_N):
                    scale[bn_i] = KScale[tl_block_s + bn_i]

                for bn_i, d_i in T.Parallel(block_N, dim):
                    index_k[bn_i, d_i] = index_k[bn_i, d_i] * scale[bn_i]

                cur_tl_block_size = tl_block_e - tl_block_s
                for n_i in T.parallel(block_N):
                    for d_i in T.parallel(dim):
                        if n_i >= cur_tl_block_size:
                            index_k[n_i, d_i] = T.cast(0, accum_dtype)

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
            for d_i in T.Parallel(dim):
                acc[d_i] = acc[d_i] * inv_count

            # Re-quantize the f32 mean to fp8 with a per-block scale.
            T.reduce_absmax(acc, max_abs, dim=0, clear=True)
            block_scale = T.max(max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype), T.cast(1e-10, accum_dtype))
            inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

            for d_i in T.Parallel(dim):
                BlockedK[bx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
            BlockedKScale[bx] = block_scale

    return fp8_native_block_mean_pooling_kernel

def fp8_native_block_mean_pooling_interface(k, k_scale, k_block_size):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=torch.float8_e4m3fn)
    blocked_k_scale = torch.empty((max_num_pooling_blocks,), device=k.device, dtype=torch.float32)
    kernel = fp8_native_block_mean_pooling(
        max_num_pooling_blocks=max_num_pooling_blocks,
        pooling_block_size=k_block_size,
        dim=d,
    )
    kernel(
        k,
        k_scale,
        blocked_k,
        blocked_k_scale,
    )
    return blocked_k, blocked_k_scale


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_mean_pooling_grouped(
    max_num_pooling_blocks: int,
    pooling_block_size: int,   # K, must divide block_N
    dim: int,
    block_N: int = 64,
    threads: int = 256,
):
    """Grouped variant of ``fp8_native_block_mean_pooling`` for K < block_N.

    The non-grouped kernel does ``T.copy(K[s:s+block_N=64], ...)`` per pool
    block, which over-reads ``block_N - K`` rows past each pool block end —
    fine inside the K tensor body, but at the last few pool blocks it walks
    off the end of ``seq_len_k`` and (under sustained load) eventually hits
    an unmapped page → Xid 13 → silent crash.

    This grouped variant flips the parallelism: one CTA per ``block_N``
    tokens, producing ``G = block_N // K`` pool blocks. The ``T.copy``
    reads exactly ``block_N`` rows that all live within seq_len_k except
    possibly at the very last CTA, which we handle with a row-guard
    (``T.Parallel`` + Python ``if``) instead of bulk T.copy.

    Constraint: ``block_N % pooling_block_size == 0``. Use the non-grouped
    variant when ``K >= block_N``.
    """
    assert block_N % pooling_block_size == 0, (
        f"block_N ({block_N}) must be divisible by "
        f"pooling_block_size ({pooling_block_size})"
    )
    G = block_N // pooling_block_size

    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    seq_len_k = T.dynamic("seq_len_k")
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def fp8_native_block_mean_pooling_grouped_kernel(
        K: T.Tensor([seq_len_k, dim], dtype=dtype),                            # type: ignore
        KScale: T.Tensor([seq_len_k], dtype=accum_dtype),                       # type: ignore
        BlockedK: T.Tensor([max_num_pooling_blocks, dim], dtype=dtype),         # type: ignore
        BlockedKScale: T.Tensor([max_num_pooling_blocks], dtype=accum_dtype),   # type: ignore
    ):
        # Grid: one CTA per block_N tokens (= G pool blocks).
        with T.Kernel(T.ceildiv(seq_len_k, block_N), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], accum_dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc_per_pool = T.alloc_fragment([G, dim], accum_dtype)
            max_abs_per_pool = T.alloc_fragment([G], accum_dtype)

            tl_block_s = bx * block_N
            cur_block_size = T.min(tl_block_s + block_N, seq_len_k) - tl_block_s

            # Bounds-safe row-by-row load. Avoids the OOB pattern of the
            # non-grouped kernel's bulk ``T.copy``. Slightly slower per-CTA
            # (no TMA), but only one CTA per block_N tokens vs one per pool
            # block, so total launches drop by G.
            T.fill(scale, 0.0)
            T.fill(index_k, 0.0)
            for bn_i in T.Parallel(block_N):
                if bn_i < cur_block_size:
                    scale[bn_i] = KScale[tl_block_s + bn_i]
            for bn_i, d_i in T.Parallel(block_N, dim):
                if bn_i < cur_block_size:
                    index_k[bn_i, d_i] = (
                        T.cast(K[tl_block_s + bn_i, d_i], accum_dtype)
                        * scale[bn_i]
                    )

            # Per-pool sum: build a [G, K, dim] view of index_k and reduce
            # along axis 1. The intermediate fragment costs an extra register
            # tile but lets tilelang's IR pattern-match a clean reduction
            # (avoiding the non-unit-stride / "k_i used before def" pitfalls
            # of a manual nested-loop accumulator).
            gk_view = T.alloc_fragment([G, pooling_block_size, dim], accum_dtype)
            for g_i, k_inner, d_i in T.Parallel(G, pooling_block_size, dim):
                gk_view[g_i, k_inner, d_i] = index_k[
                    g_i * pooling_block_size + k_inner, d_i
                ]
            T.reduce_sum(gk_view, acc_per_pool, dim=1, clear=True)

            # Per-pool mean: divide by actual valid token count (handles the
            # partial trailing pool block when seq_len_k isn't a multiple of
            # K). ``T.if_then_else`` guards the divide-by-zero case for pool
            # blocks that fall entirely past seq_len_k (they will be masked
            # out at store time anyway).
            for g_i, d_i in T.Parallel(G, dim):
                pool_start = tl_block_s + g_i * pooling_block_size
                pool_end_c = T.min(pool_start + pooling_block_size, seq_len_k)
                pc = pool_end_c - pool_start
                inv_count = T.if_then_else(
                    pc > 0,
                    T.cast(1.0, accum_dtype) / T.cast(pc, accum_dtype),
                    T.cast(0.0, accum_dtype),
                )
                acc_per_pool[g_i, d_i] = acc_per_pool[g_i, d_i] * inv_count

            # Per-pool fp8 max-abs scale.
            T.reduce_absmax(acc_per_pool, max_abs_per_pool, dim=1, clear=True)

            # Quantize + masked store: skip pool blocks past num_pool_total
            # (last CTA's trailing slots).
            for g_i, d_i in T.Parallel(G, dim):
                out_idx = bx * G + g_i
                if out_idx < max_num_pooling_blocks:
                    bs = T.max(
                        max_abs_per_pool[g_i] * T.cast(FP8_MAX_INV, accum_dtype),
                        T.cast(1e-10, accum_dtype),
                    )
                    BlockedK[out_idx, d_i] = T.cast(
                        acc_per_pool[g_i, d_i] / bs, dtype,
                    )
            for g_i in T.Parallel(G):
                out_idx = bx * G + g_i
                if out_idx < max_num_pooling_blocks:
                    bs = T.max(
                        max_abs_per_pool[g_i] * T.cast(FP8_MAX_INV, accum_dtype),
                        T.cast(1e-10, accum_dtype),
                    )
                    BlockedKScale[out_idx] = bs

    return fp8_native_block_mean_pooling_grouped_kernel


def fp8_native_block_mean_pooling_grouped_interface(k, k_scale, k_block_size, block_N=64):
    """Tilelang grouped mean-pool for K < block_N. Same I/O contract as
    ``fp8_native_block_mean_pooling_interface``."""
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=torch.float8_e4m3fn)
    blocked_k_scale = torch.empty((max_num_pooling_blocks,), device=k.device, dtype=torch.float32)
    kernel = fp8_native_block_mean_pooling_grouped(
        max_num_pooling_blocks=max_num_pooling_blocks,
        pooling_block_size=k_block_size,
        dim=d,
        block_N=block_N,
    )
    kernel(k, k_scale, blocked_k, blocked_k_scale)
    return blocked_k, blocked_k_scale


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def pool_mqa_attn_return_logits_fp8(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    """Prefill block-MQA with fp8 Q + fp8 BlockedK + per-block f32 scale.

    Uses native fp8×fp8→f32 Tensor Core GEMM.  Per-block scale applied
    post-GEMM (it varies per K-block so it affects cross-block ranking).
    """
    if block_Q is None:
        block_Q = 128 // heads
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = "float32"
    index_dtype = "int32"

    seq_len = T.dynamic("seq_len")
    seq_len_blocked_kv = T.dynamic("seq_len_blocked_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_blocked_kv, index_dim]
    index_k_scale_shape = [seq_len_blocked_kv]
    logits_shape = [seq_len, seq_len_blocked_kv]

    @T.prim_func
    def pool_mqa_attn_return_logits_fp8_kernel(
        IndexQ: T.Tensor(index_q_shape, fp8_dtype),                     # type: ignore
        IndexBlockedK: T.Tensor(index_k_shape, fp8_dtype),              # type: ignore
        IndexBlockedKScale: T.Tensor(index_k_scale_shape, accum_dtype), # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),                    # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),               # type: ignore
        CuSeqLenBlockedKS: T.Tensor([seq_len], index_dtype),            # type: ignore
        CuSeqLenBlockedKE: T.Tensor([seq_len], index_dtype),            # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], fp8_dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
            # Scale lives in register fragment — post-GEMM `s * scale` is
            # register×register, ~15% faster than smem at large M.
            index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)
            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenBlockedKS[seq_len_i + bq_i], seq_len_blocked_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenBlockedKE[seq_len_i + bq_i], seq_len_blocked_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexBlockedK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexBlockedKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * index_k_scale_fragment[bn_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return pool_mqa_attn_return_logits_fp8_kernel


def pool_mqa_attn_return_logits_fp8_interface(q_fp8, blocked_kv_fp8, blocked_kv_scale, kv_block_size, weights_f32, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke, clean_logits=True, force_maintain=True):
    """Prefill block-MQA interface: fp8 Q + fp8 BlockedK + f32 scale + f32 Weights."""
    seq_len, heads, index_dim = q_fp8.shape
    seq_len_blocked_kv = blocked_kv_fp8.shape[0]

    kernel = pool_mqa_attn_return_logits_fp8(heads=heads, index_dim=index_dim)
    logits = torch.empty([seq_len, seq_len_blocked_kv], device=q_fp8.device, dtype=torch.float32)
    kernel(
        q_fp8.view(seq_len * heads, index_dim),
        blocked_kv_fp8,
        blocked_kv_scale,
        logits,
        weights_f32,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )
    if clean_logits and force_maintain:
        clean_and_maintain_logits_kernel = clean_and_maintain_logits_()
        clean_and_maintain_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
    else:
        if clean_logits:
            clean_logits_()(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
        if force_maintain:
            force_maintain_logits_()(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
    return logits


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_sparse_mqa_attn_return_logits(
    kv_block_size,
    topk,
    heads,
    index_dim,
    block_N=128,
    num_stages=1,
    threads=256,
):
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    # TopKBlockIndex is int64 (torch.topk's native output) — avoids one
    # int64→int32 cast kernel between hierarchy_caller and this op.
    topk_index_dtype = T.int64

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk * kv_block_size]

    H_per_block = heads
    block_N = T.min(block_N, kv_block_size)
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"

    @T.prim_func
    def fp8_native_block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, fp8_dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, fp8_dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype), # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], topk_index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            # fp8 Q/K smem: 8KB + 16KB = ~25KB total vs prior 112KB → 4 blocks/SM.
            # GEMM is native fp8×fp8→f32 (no f32 intermediate smem).
            index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
            # Scale stays in shared here (not fragment): sparse_mqa has an
            # outer `for n_i in serial(topk)` loop × Pipelined inner loop,
            # and fragment reload across serial iterations measured ~15%
            # slower end-to-end (shared load cheap in this loop pattern).
            scale_shared = T.alloc_shared([block_N], accum_dtype)

            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads:seq_len_i * heads + H_per_block, :], index_q_shared)
            T.copy(Weights[seq_len_i, :], weights)

            for n_i in T.serial(topk):
                # Cast to int32 here: downstream slices (T.copy bounds) must
                # match smem's int32 extent, otherwise TileLang's tile-op
                # lowering fails with a StructuralEqual extent check.
                topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i], index_dtype)
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N

                    T.copy(IndexK[block_s_i:block_s_i + block_N, :], index_k_shared)
                    # 1D scale load via T.Parallel to avoid TMA MISALIGNED_ADDRESS
                    # (CUDA err 715) at unaligned seq_len_kv base addresses.
                    for bn_i in T.Parallel(block_N):
                        scale_shared[bn_i] = IndexKScale[block_s_i + bn_i]

                    # fp8 × fp8 → f32 GEMM (K scale applied post-GEMM below).
                    # FullRow: M=block_N=128 / 8 warps = 16 rows/warp = exact
                    # WMMA m=16 tile.  FullCol N=64/8=8 < WMMA n min of 16 →
                    # inefficient.
                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * scale_shared[bn_i], 0) * weights[bq_i, h_i])
                    
                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0] 
    
    @T.prim_func
    def fp8_native_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size(
        IndexQ: T.Tensor(index_q_shape, fp8_dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, fp8_dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype), # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], topk_index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            # fp8 Q/K smem: ~25KB total → 4 blocks/SM (vs prior 112KB / 2 blocks).
            # fp8×fp8 GEMM; per-token K scale applied in the post-GEMM step.
            index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
            # Scale stays in shared (see note in the non-small variant above).
            scale_shared = T.alloc_shared([block_N], accum_dtype)

            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads:seq_len_i * heads + H_per_block, :], index_q_shared)
            T.copy(Weights[seq_len_i, :], weights)

            for n_i in T.serial(topk):
                # Cast to int32: T.copy slice extents must be int32 to match
                # smem shape during TileLang lowering.
                topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i], index_dtype)
                block_s_i = topk_block_id * kv_block_size

                T.copy(IndexK[block_s_i:block_s_i + block_N, :], index_k_shared)
                # 1D scale load via T.Parallel to avoid TMA MISALIGNED_ADDRESS
                # (CUDA err 715) at unaligned seq_len_kv base addresses.
                for bn_i in T.Parallel(block_N):
                    scale_shared[bn_i] = IndexKScale[block_s_i + bn_i]

                # FullRow: M=block_N=128/8 warps=16 = WMMA m-tile.  FullCol would
                # give N=H_per_block=64/8=8 < WMMA n-min (16), fragmenting tiles.
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * scale_shared[bn_i], 0) * weights[bq_i, h_i])
                
                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, n_i * kv_block_size + bn_i] = logits[bn_i, 0] 

    if kv_block_size == block_N:
        return fp8_native_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size
    else:
        return fp8_native_block_sparse_mqa_attn_return_logits_kernel

def fp8_native_block_sparse_mqa_attn_return_logits_interface(q, k, k_scale, topk_block_index, kv_block_size, weights, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = k.shape[0]
    topk = topk_block_index.shape[1]

    block_sparse_mqa_attn_return_logits_kernel = fp8_native_block_sparse_mqa_attn_return_logits(heads=heads, index_dim=index_dim, kv_block_size=kv_block_size, topk=topk)
    logits = torch.empty([seq_len, topk * kv_block_size], device=q.device, dtype=torch.float32)
    block_sparse_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        k,
        k_scale,
        topk_block_index,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return logits


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_sparse_mqa_attn_return_logits_grouped(
    kv_block_size,    # K, must be < block_N and divide block_N
    topk,             # must be divisible by G = block_N // K
    heads,
    index_dim,
    block_N=128,      # GEMM tile / parallelism dial.
    threads=256,
):
    """Grouped variant of ``fp8_native_block_sparse_mqa_attn_return_logits``
    for K < block_N.

    Direct copy of the ``..._kernel_for_small_pooling_size`` vanilla
    kernel above (which works at K=block_N=64), with only the per-iter
    K-tensor load and output-column indexing widened from ``K`` rows
    (1 topk index) to ``block_N = G*K`` rows (G consecutive topk
    indices). Same shared-memory layout, same warp-spec friendly
    load+gemm+post-process pattern, same mask shape — just
    Python-unrolled over the G groups so the IR stays at one nesting
    level (no T.serial(G) inside T.serial(num_chunks), which would
    fragment producer/consumer pairing across loop boundaries).

    Constraints: ``block_N % K == 0`` and ``topk % G == 0``. Use the
    non-grouped variant when K >= block_N.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    topk_index_dtype = T.int64

    assert block_N % kv_block_size == 0, (
        f"block_N ({block_N}) must be divisible by kv_block_size ({kv_block_size})"
    )
    G = block_N // kv_block_size
    assert topk % G == 0, (
        f"topk ({topk}) must be divisible by G ({G})"
    )
    num_chunks = topk // G

    H_per_block = heads

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk * kv_block_size]

    @T.prim_func
    def fp8_native_block_sparse_mqa_attn_return_logits_grouped_kernel(
        IndexQ: T.Tensor(index_q_shape, fp8_dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, fp8_dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], topk_index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
            scale_shared = T.alloc_shared([block_N], accum_dtype)

            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads:seq_len_i * heads + H_per_block, :], index_q_shared)
            T.copy(Weights[seq_len_i, :], weights)

            for n_i in T.serial(num_chunks):
                n_i_start = n_i * G

                # CHANGE vs vanilla: load G K-blocks into the [block_N, D]
                # tile (vanilla loads block_N=K rows from one topk
                # index). G is constexpr → Python-unroll keeps each
                # T.copy at the same nesting level as the GEMM consumer
                # (a T.serial(G) inner loop would split warp-spec
                # producer/consumer scopes).
                for g_i in range(G):
                    topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i_start + g_i], index_dtype)
                    block_s_i = topk_block_id * kv_block_size
                    T.copy(
                        IndexK[block_s_i:block_s_i + kv_block_size, :],
                        index_k_shared[g_i * kv_block_size:(g_i + 1) * kv_block_size, :],
                    )

                # CHANGE vs vanilla: scale source addr depends on
                # ``bn_i // K`` (which group), not a single ``block_s_i``.
                # ``T.Parallel(block_N)`` matches the existing 256-thread
                # extent (``T.Parallel(K)`` would be too small at K<64
                # → loop-layout error).
                for bn_i in T.Parallel(block_N):
                    g_i = bn_i // kv_block_size
                    b_i = bn_i - g_i * kv_block_size
                    topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i_start + g_i], index_dtype)
                    scale_shared[bn_i] = IndexKScale[topk_block_id * kv_block_size + b_i]

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * scale_shared[bn_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                # CHANGE vs vanilla: per-row k_i uses its group's
                # ``block_s_i`` (same lookup as scale write).
                for i_i in T.Parallel(block_N):
                    g_i = i_i // kv_block_size
                    b_i = i_i - g_i * kv_block_size
                    topk_block_id = T.cast(TopKBlockIndex[seq_len_i, n_i_start + g_i], index_dtype)
                    k_i = topk_block_id * kv_block_size + b_i
                    if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                # CHANGE vs vanilla: output offset uses ``n_i_start``
                # (G groups stored back-to-back).
                for bn_i in T.Parallel(block_N):
                    Logits[seq_len_i, n_i_start * kv_block_size + bn_i] = logits[bn_i, 0]

    return fp8_native_block_sparse_mqa_attn_return_logits_grouped_kernel


def fp8_native_block_sparse_mqa_attn_return_logits_grouped_interface(
    q, k, k_scale, topk_block_index, kv_block_size, weights,
    cu_seqlen_ks, cu_seqlen_ke, block_N=128,
):
    """Tilelang grouped block-sparse MQA for K < block_N.

    Same I/O contract as ``fp8_native_block_sparse_mqa_attn_return_logits_interface``,
    but pads the topk dim up to a multiple of ``G = block_N // K`` so the
    grouped kernel's ``topk % G == 0`` constraint holds. Padding entries
    use a sentinel ``-1`` that the kernel masks to ``-inf``; we slice the
    output back to the original ``topk * K`` width.
    """
    seq_len, heads, index_dim = q.shape
    topk = topk_block_index.shape[1]
    assert block_N % kv_block_size == 0, (
        f"block_N ({block_N}) must be divisible by kv_block_size ({kv_block_size})"
    )
    G = block_N // kv_block_size
    pad = (G - (topk % G)) % G
    topk_padded = topk + pad

    if pad > 0:
        topk_pad = torch.full(
            (seq_len, pad), -1,
            device=topk_block_index.device, dtype=topk_block_index.dtype,
        )
        topk_block_index_padded = torch.cat([topk_block_index, topk_pad], dim=1)
    else:
        topk_block_index_padded = topk_block_index

    kernel = fp8_native_block_sparse_mqa_attn_return_logits_grouped(
        heads=heads, index_dim=index_dim,
        kv_block_size=kv_block_size, topk=topk_padded,
        block_N=block_N,
    )
    logits_padded = torch.empty(
        [seq_len, topk_padded * kv_block_size],
        device=q.device, dtype=torch.float32,
    )
    kernel(
        q.view(seq_len * heads, index_dim),
        k, k_scale,
        topk_block_index_padded,
        logits_padded,
        weights,
        cu_seqlen_ks, cu_seqlen_ke,
    )
    if pad > 0:
        return logits_padded[:, : topk * kv_block_size].contiguous()
    return logits_padded


def fp8_native_hierarchy_mqa_logits_tilelang_legacy(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    k_block_size: int, 
    block_topk: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, topk * k_block_size], dtype `torch.float32`.
    """
    k_fp8, k_scales = kv
    k_scales = k_scales.view(torch.float32)

    if k_scales.ndim == 2:
        assert k_scales.shape[1] == 1, "k_scales should have shape [N] or [N, 1], but got shape {}".format(k_scales.shape)
        k_scales = k_scales.squeeze(1)

    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    # Mean pool now outputs fp8 + per-block f32 scale (re-quantized in kernel).
    blocked_k, blocked_k_scale = fp8_native_block_mean_pooling_interface(k_fp8, k_scales, k_block_size)

    # pool_mqa_fp8 uses fp8×fp8 GEMM; no Python-level cast for Q or weights.
    block_k_indexer_score = pool_mqa_attn_return_logits_fp8_interface(q_fp8=q, blocked_kv_fp8=blocked_k, blocked_kv_scale=blocked_k_scale, kv_block_size=k_block_size, weights_f32=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks, cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)

    # bf16 + sorted=False: bf16 topk is ~40% faster than f32 on [M, num_blocks],
    # and downstream sparse_mqa doesn't depend on n_i ordering.  Validated with
    # longbench_samsum (--limit 32): rouge 0.4544 ± 0.028 (> 0.40 baseline).
    # torch.topk returns int64; sparse_mqa kernel now accepts int64 directly,
    # avoiding an otherwise-needed int64→int32 cast.
    topk_block_indices = torch.topk(block_k_indexer_score.bfloat16(), k=min(block_topk, block_k_indexer_score.shape[-1]), dim=-1, sorted=False).indices  # [M, topk] int64

    block_sparse_logits = fp8_native_block_sparse_mqa_attn_return_logits_interface(q=q, k=k_fp8, k_scale=k_scales, topk_block_index=topk_block_indices, kv_block_size=k_block_size, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)

    return block_sparse_logits, topk_block_indices


def fp8_native_hierarchy_mqa_logits_with_pool_cache(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    k_block_size: int,
    block_topk: int,
    blocked_k_cache: torch.Tensor,         # [max_pool_blocks, D] fp8  IN-OUT
    blocked_k_scale_cache: torch.Tensor,   # [max_pool_blocks]    f32  IN-OUT
    new_k_start: int,                      # start token index (inclusive) of the NEW K slice in k_fp8
    new_k_end: int,                        # end   token index (exclusive) of the NEW K slice
    current_num_pool: int,                 # number of valid pool blocks in blocked_k_cache for this step
):
    """Chunked-prefill variant of ``fp8_native_hierarchy_mqa_logits_tilelang_legacy``.

    Caller maintains a persistent pool-K cache across chunks. Each call:

      1. Pools only the NEW K slice ``k_fp8[new_k_start:new_k_end]`` (the
         tokens added in this chunk) and writes the resulting pool blocks
         into ``blocked_k_cache[new_k_start // k_block_size :
         new_k_start // k_block_size + num_new_blocks]``. Blocks that were
         already pooled by earlier chunks are left untouched — reuse saves
         their mean-pool cost.
      2. Runs pool_mqa on ``blocked_k_cache[:current_num_pool]``, followed
         by top-k over blocks and fine-grained sparse MQA on the raw K
         (same as the no-cache path).

    ``new_k_start`` must be aligned to ``k_block_size`` so pool-block writes
    don't overlap earlier chunks' valid ranges. ``new_k_end`` may be
    unaligned (the last pool block of this chunk is then a ragged tail;
    correct pool-of-partial-block semantics are handled by the flat mean
    pool kernel).
    """
    assert new_k_start % k_block_size == 0, (
        f"new_k_start ({new_k_start}) must be a multiple of k_block_size "
        f"({k_block_size}) so cache writes land on pool-block boundaries."
    )
    assert new_k_end > new_k_start, (
        f"new_k_end ({new_k_end}) must be > new_k_start ({new_k_start})"
    )

    k_fp8, k_scales = kv
    k_scales = k_scales.view(torch.float32)
    if k_scales.ndim == 2:
        assert k_scales.shape[1] == 1
        k_scales = k_scales.squeeze(1)

    # 1) Pool only the new slice of K, paste into the persistent cache.
    new_blocked_k, new_blocked_k_scale = fp8_native_block_mean_pooling_interface(
        k_fp8[new_k_start:new_k_end],
        k_scales[new_k_start:new_k_end],
        k_block_size,
    )
    new_pool_start = new_k_start // k_block_size
    new_pool_end = new_pool_start + new_blocked_k.shape[0]
    blocked_k_cache[new_pool_start:new_pool_end] = new_blocked_k
    blocked_k_scale_cache[new_pool_start:new_pool_end] = new_blocked_k_scale

    # 2) pool_mqa on cached pool up to current_num_pool.
    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size
    block_k_indexer_score = pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q,
        blocked_kv_fp8=blocked_k_cache[:current_num_pool],
        blocked_kv_scale=blocked_k_scale_cache[:current_num_pool],
        kv_block_size=k_block_size,
        weights_f32=weights,
        cu_seqlen_blocked_ks=cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke=cu_seqlen_blocked_ke,
    )
    topk_block_indices = torch.topk(
        block_k_indexer_score.bfloat16(),
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices

    # 3) sparse_mqa on raw K (unchanged from no-cache pipeline).
    block_sparse_logits = fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q=q, k=k_fp8, k_scale=k_scales,
        topk_block_index=topk_block_indices,
        kv_block_size=k_block_size, weights=weights,
        cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke,
    )
    return block_sparse_logits, topk_block_indices


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def batch_decode_pool_mqa_attn_return_logits(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    block_H: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype: str = "bfloat16",
):
    """
    Decode 专用：q_len 固定为 1，不在 Q 维分块，只在 H 和 Nb 上分块。

    Shapes:
      Q:          [B, 1, H, D]
      BlockedK:   [B, Nb, D]
      Logits:     [B, 1, Nb] fp32
      Weights:    [B, 1, H]
      ContextLens:[B]  (有效 Nb)
    """
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    nb = T.dynamic("seq_len_blocked_kv")

    q_shape = [batch, heads, index_dim]
    k_shape = [batch, nb, index_dim]
    logits_shape = [batch, nb]
    w_shape = [batch, heads]

    # padding 到 16 对齐，避免 gemm 列维过小/不合法
    block_H_pad = T.ceildiv(block_H, 16) * 16
    assert block_H_pad == heads

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        BlockedK: T.Tensor(k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(w_shape, dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, 1, threads=threads) as (bx, by):
            # shared tiles
            k_shared = T.alloc_shared([block_N, index_dim], dtype)
            q_shared = T.alloc_shared([block_H_pad, index_dim], dtype)

            # fragments
            s = T.alloc_fragment([block_N, block_H_pad], accum_dtype)
            w = T.alloc_fragment([block_H_pad], accum_dtype)
            logits_accum = T.alloc_fragment([block_N], accum_dtype)

            # valid kv range
            k_e = T.min(ContextLens[bx], nb)
            T.copy(Q[bx, 0, 0], q_shared)
            T.copy(Weights[bx, 0], w)

            for k_i in T.Pipelined(T.ceildiv(nb, block_N), num_stages=num_stages):
                k_start = k_i * block_N
                T.copy(BlockedK[bx, k_start, 0], k_shared)

                T.gemm(
                    k_shared,
                    q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for kn_i, hn_i in T.Parallel(block_N, block_H_pad):
                    s[kn_i, hn_i] = T.max(s[kn_i, hn_i], 0) * w[hn_i]

                T.reduce_sum(s, logits_accum, dim=1, clear=True)

                for kn_i in T.Parallel(block_N):
                    k_col = k_start + kn_i
                    if k_col < k_e:
                        Logits[bx, k_col] = logits_accum[kn_i]
                    else:
                        Logits[bx, k_col] = -T.infinity(accum_dtype)

    return kernel


def batch_pool_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    blocked_kv: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    kv_block_size: int,
    clean_logits: bool = True,
    force_maintain: bool = True,
    dtype: str = "bfloat16",
    block_N: int = 64,
):
    """
    Decode 接口：
      q:          [B, 1, H, D]
      blocked_kv: [B, Nb, D]
      weights:    [B,1,H]
      context_lens:[B] (有效 Nb)
    Return:
      logits: [B, Nb] fp32
    """

    assert len(q.shape) == 4
    B, seq_len_q, H, D = q.shape
    B, seq_len_kv, D = blocked_kv.shape

    assert seq_len_q == 1, "decode expects q_len=1"

    q = q.squeeze(1)
    weights = weights.squeeze(1)

    logits = torch.empty((B, seq_len_kv), device=q.device, dtype=torch.float32)

    kernel = batch_decode_pool_mqa_attn_return_logits(
        heads=H,
        index_dim=D,
        block_N=block_N,
        block_H=H,
        dtype=dtype,
    )
    kernel(
        q,
        blocked_kv,
        logits,
        weights,
        context_lens.to(torch.int32),
    )

    if clean_logits:
        n = torch.arange(seq_len_kv, device=q.device)[None, :]
        valid = n < context_lens.to(torch.int64)[:, None]
        logits = logits.masked_fill(~valid[:, :], float("-inf"))

    if force_maintain:
        # ctx = context_lens.to(torch.int64).clamp(min=0, max=seq_len_kv)
        # b_idx = torch.arange(B, device=q.device)
        # last = (ctx - 1).clamp(min=0, max=seq_len_kv - 1)
        # logits[b_idx, 0] = float("inf")
        # logits[b_idx, last] = float("inf")

        ctx = context_lens.to(torch.int64).clamp(min=0, max=seq_len_kv)
        last = (ctx - 1).clamp(min=0, max=seq_len_kv - 1)
        logits[:, 0].fill_(float("inf"))
        logits.scatter_(dim=1, index=last.unsqueeze(1), value=float("inf"))

    logits = logits.unsqueeze(1)  # [B,1,Nb]

    return logits


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def batch_decode_pool_mqa_attn_return_logits_fp8_legacy(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    block_H: int = 64,
    num_stages: int = 3,
    threads: int = 128,
):
    """Decode block-MQA with fp8 Q + fp8 BlockedK + per-block f32 scale + f32 Weights.

    Mirrors batch_decode_pool_mqa_attn_return_logits but uses native fp8×fp8
    GEMM (saves ~2× over bf16) and applies the per-K-block scale post-GEMM.
    threads=128 → 4 warps → block_N/4 = 16 (FullRow) and block_H/4 = 16
    (FullCol) both meet WMMA m/n-tile minimum.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    nb = T.dynamic("seq_len_blocked_kv")

    q_shape = [batch, heads, index_dim]            # fp8
    k_shape = [batch, nb, index_dim]               # fp8
    k_scale_shape = [batch, nb]                    # f32 per-block scale
    logits_shape = [batch, nb]
    w_shape = [batch, heads]                       # f32

    block_H_pad = T.ceildiv(block_H, 16) * 16
    assert block_H_pad == heads

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, fp8_dtype),                    # type: ignore
        BlockedK: T.Tensor(k_shape, fp8_dtype),             # type: ignore
        BlockedKScale: T.Tensor(k_scale_shape, accum_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),        # type: ignore
        Weights: T.Tensor(w_shape, accum_dtype),            # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),        # type: ignore
    ):
        with T.Kernel(batch, 1, threads=threads) as (bx, by):
            k_shared = T.alloc_shared([block_N, index_dim], fp8_dtype)
            k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            q_shared = T.alloc_shared([block_H_pad, index_dim], fp8_dtype)

            s = T.alloc_fragment([block_N, block_H_pad], accum_dtype)
            w = T.alloc_fragment([block_H_pad], accum_dtype)
            logits_accum = T.alloc_fragment([block_N], accum_dtype)

            k_e = T.min(ContextLens[bx], nb)
            T.copy(Q[bx, 0, 0], q_shared)
            T.copy(Weights[bx, 0], w)

            for k_i in T.Pipelined(T.ceildiv(nb, block_N), num_stages=num_stages):
                k_start = k_i * block_N
                # 2D fp8 tile copy from contiguous [batch, nb, dim] mean-pool output.
                T.copy(BlockedK[bx, k_start, 0], k_shared)
                T.copy(BlockedKScale[bx, k_start], k_scale_fragment)

                T.gemm(
                    k_shared,
                    q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # Apply per-K-block scale + relu + per-head weights, fused.
                for kn_i, hn_i in T.Parallel(block_N, block_H_pad):
                    s[kn_i, hn_i] = T.max(s[kn_i, hn_i] * k_scale_fragment[kn_i], 0) * w[hn_i]

                T.reduce_sum(s, logits_accum, dim=1, clear=True)

                for kn_i in T.Parallel(block_N):
                    k_col = k_start + kn_i
                    # Fused mask + force_maintain: position 0 and last valid
                    # position get +inf (matches Python force_maintain logic).
                    if k_col == 0 or k_col == k_e - 1:
                        Logits[bx, k_col] = T.infinity(accum_dtype)
                    elif k_col < k_e:
                        Logits[bx, k_col] = logits_accum[kn_i]
                    else:
                        Logits[bx, k_col] = -T.infinity(accum_dtype)

    return kernel


def batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
    q_fp8: torch.Tensor,
    blocked_kv_fp8: torch.Tensor,
    blocked_kv_scale: torch.Tensor,
    weights_f32: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    kv_block_size: int,
    block_N: int = 64,
):
    """Decode block-MQA: fp8 Q + fp8 BlockedK + per-block f32 scale + f32 weights.

    The kernel handles BOTH masking (-inf for invalid positions) AND
    force_maintain (+inf at position 0 and last valid position) inline,
    so no Python-side post-processing is needed.
    """
    assert len(q_fp8.shape) == 4
    B, seq_len_q, H, D = q_fp8.shape
    _B, seq_len_kv, _D = blocked_kv_fp8.shape
    assert seq_len_q == 1, "decode expects q_len=1"

    q_2d = q_fp8.squeeze(1)                  # [B, H, D] fp8
    w_2d = weights_f32.view(B, H)            # [B, H] f32

    logits = torch.empty((B, seq_len_kv), device=q_fp8.device, dtype=torch.float32)
    kernel = batch_decode_pool_mqa_attn_return_logits_fp8_legacy(
        heads=H, index_dim=D, block_N=block_N, block_H=H,
    )
    assert context_lens.dtype == torch.int32, f"context_lens must be int32, got {context_lens.dtype}"
    kernel(q_2d, blocked_kv_fp8, blocked_kv_scale, logits, w_2d, context_lens)
    return logits.unsqueeze(1)  # [B, 1, Nb]

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def paged_block_sparse_mqa_attn_return_logits(
    paged_block_size,
    kv_block_size,
    topk,
    heads,
    index_dim,
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_blocks = T.dynamic("max_blocks")
    num_phys_blocks = T.dynamic("num_phys_blocks")

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_shape = [num_phys_blocks, paged_block_size, 1, index_dim]
    logits_shape = [batch, seq_len, topk * kv_block_size]
    weights_shape = [batch, seq_len, heads]

    H_per_block = heads
    block_N = paged_block_size
    assert block_N > 0, "block_N must be positive"
    assert kv_block_size >= block_N and kv_block_size % block_N == 0, "block_N must divide kv_block_size"
    assert paged_block_size >= block_N and paged_block_size % block_N == 0, "block_N must divide paged_block_size"
    assert paged_block_size == block_N, "for simplicity we require paged_block_size == block_N in this kernel"

    @T.prim_func
    def paged_block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            T.copy(Weights[b, seq_len_i, :], weights)

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[b, seq_len_i, n_i]
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N

                    if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                        phys = BlockTables[b, block_s_i // paged_block_size]
                        T.copy(KvCache[phys, :, 0, :], index_k_shared)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                    
                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        p = k_i // paged_block_size
                        if (k_i < cu_k_s_min) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[b, seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0]
    
    @T.prim_func
    def paged_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            T.copy(Weights[b, seq_len_i, :], weights)

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[b, seq_len_i, n_i]
                block_s_i = topk_block_id * kv_block_size

                if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                    phys = BlockTables[b, block_s_i // paged_block_size]
                    T.copy(KvCache[phys, :, 0, :], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                
                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    p = k_i // paged_block_size
                    if (k_i < cu_k_s_min) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[b, seq_len_i, n_i * kv_block_size + bn_i] = logits[bn_i, 0]

    if kv_block_size == block_N:
        return paged_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size
    else:
        return paged_block_sparse_mqa_attn_return_logits_kernel

def paged_block_sparse_mqa_attn_return_logits_interface(
    q,
    kv_cache,
    topk_block_index,
    kv_block_size,
    weights,
    context_lens,
    block_tables,
    dtype="bfloat16",
):
    batch, seq_len, heads, index_dim = q.shape
    topk = int(topk_block_index.shape[-1])
    paged_block_size = int(kv_cache.shape[1])

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    logits = torch.empty(
        (batch, seq_len, topk * kv_block_size),
        device=q.device,
        dtype=torch.float32,
    )

    kernel = paged_block_sparse_mqa_attn_return_logits(
        paged_block_size=paged_block_size,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        dtype=dtype,
    )
    kernel(
        q,
        kv_cache,
        topk_block_index.to(torch.int32),
        logits,
        weights,
        context_lens.to(torch.int32),
        block_tables.to(torch.int32),
    )
    return logits

def ref_paged_mean_pooling(
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        k_block_size: int,
    ):
        """
        Args:
            kv_cache: [num_blocks, block_size, 1, D]
            context_lens: [B]
            block_tables: [B, max_blocks]  (逻辑 paged block -> 物理 block)
        Returns:
            blocked_k: [B, max_num_pooling_blocks, D]  (已 padding)
            num_pooling_blocks: [B]  (每个样本真实的 pooling block 数，用于 mask)
        """
        device = kv_cache.device
        B = int(context_lens.shape[0])
        block_size = int(kv_cache.shape[1])        # paged block size，通常 64（修复原来的错误）
        D = int(kv_cache.shape[-1])

        # 每个样本需要的 k-block 数不同，先 pad 到 batch 内最大值
        assert B > 0
        max_seqlen = int(context_lens.max().item())
        max_num_pooling_blocks = (max_seqlen + k_block_size - 1) // k_block_size

        blocked_k = torch.zeros(
            (B, max_num_pooling_blocks, D),
            device=device,
            dtype=kv_cache.dtype,
        )
        num_pooling_blocks = torch.empty((B,), device=device, dtype=torch.int32)

        for b in range(B):
            seqlen = int(context_lens[b].item())
            nblocks = (seqlen + k_block_size - 1) // k_block_size
            num_pooling_blocks[b] = nblocks

            for n in range(nblocks):
                pooling_block_start = n * k_block_size
                pooling_block_end = min((n + 1) * k_block_size, seqlen)
                pooling_block_len = pooling_block_end - pooling_block_start
                if pooling_block_len <= 0:
                    continue

                # 计算覆盖该 pooling block 所需的 paged blocks（注意：用 block_size，不是 block_tables.shape[1]）
                paged_block_start = pooling_block_start // block_size
                paged_block_end = (pooling_block_end + block_size - 1) // block_size  # exclusive
                paged_block_indices = block_tables[b, paged_block_start:paged_block_end].to(torch.long)

                # [num_paged_blocks, block_size, 1, D] -> [num_paged_blocks*block_size, D]
                paged_blocks = kv_cache.index_select(0, paged_block_indices)
                tokens = paged_blocks.reshape(-1, 1, D).reshape(-1, D)

                # pooling_block_start 可能落在 paged block 中间，切片到准确的 token 范围
                offset = pooling_block_start - paged_block_start * block_size  # == pooling_block_start % block_size
                tokens = tokens[offset : offset + pooling_block_len]           # [pooling_block_len, D]

                blocked_k[b, n] = tokens.mean(dim=0)

        return blocked_k, num_pooling_blocks

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def paged_mean_pooling(
    paged_block_size: int,
    pooling_block_size: int,
    max_num_pooling_blocks: int,
    dim: int,
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")

    kv_cache_shape = [num_blocks, paged_block_size, 1, dim]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    blocked_k_shape = [batch, max_num_pooling_blocks, dim]

    block_N = paged_block_size
    assert pooling_block_size % block_N == 0, "For simplicity, we require pooling_block_size to be a multiple of paged_block_size"

    @T.prim_func
    def paged_mean_pooling_kernel(
        KvCache: T.Tensor(kv_cache_shape, dtype), # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype), # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_shape, accum_dtype), # type: ignore
    ):
        with T.Kernel(batch, max_num_pooling_blocks, threads=threads) as (bx, by):
            b = bx
            seq_len = ContextLens[b]
            k_start = by * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len)
            cur_pooling_block_size = k_end - k_start

            index_k_shared = T.alloc_fragment([block_N, dim], dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            T.fill(acc, 0.0)

            if cur_pooling_block_size > 0:
                for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                    paged_block_s = k_start + b_i * block_N
                    T.fill(index_k_shared, 0.0)

                    if paged_block_s // paged_block_size < max_blocks:
                        paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                        T.copy(KvCache[paged_block_phys_id, :, 0, :], index_k_shared)

                    for n_i, d_i in T.Parallel(block_N, dim):
                        tl_block_idx = paged_block_s + n_i
                        if tl_block_idx >= k_end:
                            index_k_shared[n_i, d_i] = T.cast(0, accum_dtype)

                    T.reduce_sum(index_k_shared, acc, dim=0, clear=False)

                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] / T.cast(cur_pooling_block_size, accum_dtype)

            T.copy(acc, BlockedK[b, by, :])

    return paged_mean_pooling_kernel

def paged_mean_pooling_interface(
        max_num_pooling_blocks: int,
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        k_block_size: int,
    ):
    """
    Args:
        kv_cache: [num_blocks, block_size, 1, D]
        context_lens: [B]
        block_tables: [B, max_blocks]  (逻辑 paged block -> 物理 block)
    Returns:
        blocked_k: [B, max_num_pooling_blocks, D]  (已 padding)
        num_pooling_blocks: [B]  (每个样本真实的 pooling block 数，用于 mask)
    """
    num_blocks, paged_block_size, head, dim = kv_cache.shape
    batch, max_blocks = block_tables.shape
    assert head == 1, "Only support head=1 for now"

    blocked_k = torch.empty((batch, max_num_pooling_blocks, dim), device=kv_cache.device, dtype=torch.float32)

    kernel = paged_mean_pooling(paged_block_size=paged_block_size, pooling_block_size=k_block_size, max_num_pooling_blocks=max_num_pooling_blocks, dim=dim)
    kernel(
        kv_cache,
        block_tables,
        context_lens,
        blocked_k,
    )

    blocked_k = blocked_k.to(kv_cache.dtype)
    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size
    return blocked_k, num_pooling_blocks

def fp8_hierarchy_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    max_seq_len: int,
    k_block_size: int, 
    block_topk: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    # TODO 
    # 1. convert q, kv and weights to bf16
    # 2. call the paged version of tilelang kernels

    q = q_fp8.float()

    fp8_dtype = q_fp8.dtype
    dim = q.shape[-1]
    num_blocks, block_size, _, D_plus_4 = kv_cache_fp8.shape
    kv_cache = kv_cache_fp8.view(num_blocks, -1)

    scale = kv_cache[:, block_size * dim:]  # [num_blocks]
    kv_cache = kv_cache[:, :block_size * dim].view(fp8_dtype)      # [num_blocks, block_size * dim]
    scale = scale.contiguous().view(torch.float32) 

    kv_cache = kv_cache.view(num_blocks, block_size, 1, dim)  # [num_blocks, block_size, dim]
    scale = scale.view(num_blocks, block_size)

    kv_cache = kv_cache.float() * scale[:, :, None, None]  # [num_blocks, block_size, 1, D]
    # TODO bfloat16 compute
    # kv_cache = kv_cache.to(torch.bfloat16)
    # scale = scale.to(torch.bfloat16)
    # kv_cache.mul_(scale[:, :, None, None])

    q = q.bfloat16()
    kv_cache = kv_cache.bfloat16()
    weights = weights.bfloat16()
    
    max_num_pooling_blocks = (max_seq_len + k_block_size - 1) // k_block_size
    blocked_k, num_pooling_blocks = paged_mean_pooling_interface(max_num_pooling_blocks, kv_cache, context_lens, block_tables, k_block_size)  # [B, num_pooling_blocks, D], [B]

    block_k_indexer_score = batch_pool_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, context_lens=num_pooling_blocks)  # [B, next_n, num_pooling_blocks]
    topk_block_indices = torch.topk(block_k_indexer_score, k=min(block_topk, block_k_indexer_score.shape[-1]), dim=-1).indices  # [B, next_n, topk]
    topk_block_indices = topk_block_indices.to(torch.int32)

    block_sparse_k_indexer_score = paged_block_sparse_mqa_attn_return_logits_interface(q=q, kv_cache=kv_cache, topk_block_index=topk_block_indices, kv_block_size=k_block_size, weights=weights, context_lens=context_lens, block_tables=block_tables)  # [B, next_n, topk*kv_block_size]

    return block_sparse_k_indexer_score, topk_block_indices

def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv, scale = kv
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    if scale.ndim == 2:
        assert scale.shape[-1] == 1
        scale = scale.squeeze(-1)   # [N]

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits

def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache (CUDA fallback).

    This is a pure PyTorch fallback for CUDA when DeepGEMM is not available.
    Handles head_dim = 132 (128 + 4 for RoPE).

    Args:
        q: Query tensor of shape [B, next_n, H, D].
        kv_cache: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm.platforms import current_platform

    fp8_dtype = current_platform.fp8_dtype()
    # fp8_dtype = q.dtype
    batch_size, next_n, heads, dim = q.size()
    # kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    num_blocks, block_size, _, _ = kv_cache.size()
    kv_cache = kv_cache.view(num_blocks, -1)
    
    
    scale = kv_cache[:, block_size * dim:] # [num_blocks, block_size]
    kv_cache = kv_cache[:, :block_size * dim].view(fp8_dtype) # [num_blocks, block_size*dim]

    kv_cache = kv_cache.view(num_blocks, block_size, 1, dim)  # [num_blocks, block_size, dim]
    scale = scale.contiguous().view(torch.float32)  # [num_blocks, block_size]

    kv_cache = kv_cache.float() * scale[:, :, None, None]  # [num_blocks, block_size, 1, dim]

    q = q.float()
    # scale = scale.contiguous().view(torch.float32)
    # kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_blocks, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i].item()
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_idx in range((context_len - 1) // block_size + 1):
            block_id = block_tables[i][block_idx]
            qx, kx = q[i], kv_cache[block_id]
            k_offsets = torch.arange(
                block_idx * block_size, (block_idx + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_idx * block_size : (block_idx + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_paged_mean_pooling(
    paged_block_size: int,
    pooling_block_size: int,
    max_num_pooling_blocks: int,
    dim: int,
    num_stages=1,
    threads=128,
):
    """Paged mean pooling: outputs fp8 + per-block f32 scale (re-quantized)."""
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")

    kv_cache_fp8_shape = [num_blocks, paged_block_size * (dim + 4)]
    kv_cache_fp32_shape = [num_blocks, paged_block_size * (dim + 4) // 4]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    blocked_k_shape = [batch, max_num_pooling_blocks, dim]
    blocked_k_scale_shape = [batch, max_num_pooling_blocks]

    fp8_end = paged_block_size * dim
    scale_offset = paged_block_size * dim // 4
    FP8_MAX_INV = 1.0 / 448.0

    block_N = paged_block_size
    assert pooling_block_size % block_N == 0, f"For simplicity, we require pooling_block_size {pooling_block_size} to be a multiple of paged_block_size {block_N}"

    @T.prim_func
    def fp8_native_paged_mean_pooling_kernel(
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, dtype), # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype), # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype), # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype), # type: ignore
        BlockedK: T.Tensor(blocked_k_shape, dtype), # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_shape, accum_dtype), # type: ignore
    ):
        with T.Kernel(batch, max_num_pooling_blocks, threads=threads) as (bx, by):
            b = bx
            seq_len = ContextLens[b]
            k_start = by * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len)
            cur_pooling_block_size = k_end - k_start

            index_k_shared = T.alloc_fragment([block_N * dim], accum_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, dim])
            scale_shared = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            if cur_pooling_block_size > 0:
                for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                    paged_block_s = k_start + b_i * block_N
                    T.fill(index_k_shared, 0.0)

                    if paged_block_s // paged_block_size < max_blocks:
                        paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                        T.copy(KvCacheFP8View[paged_block_phys_id, :fp8_end], index_k_shared)
                        T.copy(KvCacheFP32View[paged_block_phys_id, scale_offset:], scale_shared)

                    for n_i, d_i in T.Parallel(block_N, dim):
                        tl_block_idx = paged_block_s + n_i
                        index_k_reshaped[n_i, d_i] = T.cast(index_k_reshaped[n_i, d_i], accum_dtype) * scale_shared[n_i]
                        if tl_block_idx >= k_end:
                            index_k_reshaped[n_i, d_i] = T.cast(0, accum_dtype)

                    T.reduce_sum(index_k_reshaped, acc, dim=0, clear=False)

                inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] * inv_count

            # Re-quantize f32 mean to fp8 with per-block scale.
            T.reduce_absmax(acc, max_abs, dim=0, clear=True)
            block_scale = T.max(max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype), T.cast(1e-10, accum_dtype))
            inv_block_scale = T.cast(1.0, accum_dtype) / block_scale
            for d_i in T.Parallel(dim):
                BlockedK[b, by, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
            BlockedKScale[b, by] = block_scale

    return fp8_native_paged_mean_pooling_kernel

def fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks: int,
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        k_block_size: int,
    ):
    """Returns (blocked_k fp8, blocked_k_scale f32, num_pooling_blocks)."""
    num_blocks, paged_block_size, head, DPlus4 = kv_cache.shape
    batch, max_blocks = block_tables.shape
    assert head == 1, "Only support head=1 for now"

    dim = DPlus4 - 4
    kv_cache = kv_cache.view(num_blocks, paged_block_size * (dim + 4))

    blocked_k = torch.empty((batch, max_num_pooling_blocks, dim), device=kv_cache.device, dtype=torch.float8_e4m3fn)
    blocked_k_scale = torch.empty((batch, max_num_pooling_blocks), device=kv_cache.device, dtype=torch.float32)

    kernel = fp8_native_paged_mean_pooling(paged_block_size=paged_block_size, pooling_block_size=k_block_size, max_num_pooling_blocks=max_num_pooling_blocks, dim=dim)
    kernel(
        kv_cache.view(torch.float8_e4m3fn),
        kv_cache.view(torch.float32),
        block_tables,
        context_lens,
        blocked_k,
        blocked_k_scale,
    )

    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size
    return blocked_k, blocked_k_scale, num_pooling_blocks


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_paged_mean_pooling_tail_only_legacy(
    paged_block_size: int,
    pooling_block_size: int,
    max_num_pooling_blocks: int,
    dim: int,
    num_stages=1,
    threads=128,
):
    """Paged mean pooling — tail block only.

    Like ``fp8_native_paged_mean_pooling`` but per batch entry, writes only
    the last pool block (index ``num_pool - 1`` where ``num_pool =
    ceildiv(seq_len, pooling_block_size)``). Non-tail indices of the output
    buffer are left untouched — the caller is expected to populate them
    from a pre-computed pool-K cache before invoking the kernel.

    Grid is ``(batch,)`` (one program per batch entry, not per pool block),
    so the kernel does O(1) pool work per batch regardless of context length.
    If ``seq_len == 0`` the kernel is a no-op for that batch entry.
    """
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")

    kv_cache_fp8_shape = [num_blocks, paged_block_size * (dim + 4)]
    kv_cache_fp32_shape = [num_blocks, paged_block_size * (dim + 4) // 4]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    blocked_k_shape = [batch, max_num_pooling_blocks, dim]
    blocked_k_scale_shape = [batch, max_num_pooling_blocks]

    fp8_end = paged_block_size * dim
    scale_offset = paged_block_size * dim // 4
    FP8_MAX_INV = 1.0 / 448.0

    block_N = paged_block_size
    assert pooling_block_size % block_N == 0, (
        "For simplicity, we require pooling_block_size to be a multiple of paged_block_size"
    )

    @T.prim_func
    def fp8_native_paged_mean_pooling_tail_only_kernel(
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, dtype),  # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype),  # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype),  # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype),  # type: ignore
        BlockedK: T.Tensor(blocked_k_shape, dtype),  # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch, threads=threads) as bx:
            b = bx
            seq_len = ContextLens[b]
            # num_pool = ceildiv(seq_len, pooling_block_size). Tail = num_pool - 1.
            num_pool = T.ceildiv(seq_len, pooling_block_size)
            tail_idx = num_pool - 1
            k_start = tail_idx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len)
            cur_pooling_block_size = k_end - k_start

            index_k_shared = T.alloc_fragment([block_N * dim], accum_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, dim])
            scale_shared = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            if num_pool > 0:
                if cur_pooling_block_size > 0:
                    for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                        paged_block_s = k_start + b_i * block_N
                        T.fill(index_k_shared, 0.0)

                        if paged_block_s // paged_block_size < max_blocks:
                            paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                            T.copy(KvCacheFP8View[paged_block_phys_id, :fp8_end], index_k_shared)
                            T.copy(KvCacheFP32View[paged_block_phys_id, scale_offset:], scale_shared)

                        for n_i, d_i in T.Parallel(block_N, dim):
                            tl_block_idx = paged_block_s + n_i
                            index_k_reshaped[n_i, d_i] = T.cast(index_k_reshaped[n_i, d_i], accum_dtype) * scale_shared[n_i]
                            if tl_block_idx >= k_end:
                                index_k_reshaped[n_i, d_i] = T.cast(0, accum_dtype)

                        T.reduce_sum(index_k_reshaped, acc, dim=0, clear=False)

                    inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
                    for d_i in T.Parallel(dim):
                        acc[d_i] = acc[d_i] * inv_count

                # Re-quantize the tail's f32 mean to fp8 with a per-block scale.
                T.reduce_absmax(acc, max_abs, dim=0, clear=True)
                block_scale = T.max(max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype), T.cast(1e-10, accum_dtype))
                inv_block_scale = T.cast(1.0, accum_dtype) / block_scale
                for d_i in T.Parallel(dim):
                    BlockedK[b, tail_idx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
                BlockedKScale[b, tail_idx] = block_scale

    return fp8_native_paged_mean_pooling_tail_only_kernel


def fp8_native_paged_mean_pooling_tail_only_legacy_interface(
    blocked_k: torch.Tensor,           # [B, max_num_pooling_blocks, D]  fp8   IN-OUT
    blocked_k_scale: torch.Tensor,     # [B, max_num_pooling_blocks]     f32   IN-OUT
    kv_cache: torch.Tensor,            # [num_blocks, paged_block_size, 1, D+4]
    context_lens: torch.Tensor,        # [B] int32
    block_tables: torch.Tensor,        # [B, max_blocks] int32
    k_block_size: int,
):
    """Simulates a cache-based decode: caller has already populated all
    non-tail entries of ``blocked_k`` / ``blocked_k_scale`` (from a pool-K
    cache). The kernel pools the tail block from the raw KV-cache and writes
    it back *in place*. Returns the input tensors for chaining.
    """
    num_blocks, paged_block_size, head, DPlus4 = kv_cache.shape
    assert head == 1, "Only support head=1 for now"
    batch, max_num_pooling_blocks, dim = blocked_k.shape
    assert dim == DPlus4 - 4, f"dim mismatch: blocked_k.dim={dim} vs DPlus4-4={DPlus4 - 4}"
    assert blocked_k_scale.shape == (batch, max_num_pooling_blocks)

    kv_cache_flat = kv_cache.view(num_blocks, paged_block_size * (dim + 4))
    kernel = fp8_native_paged_mean_pooling_tail_only_legacy(
        paged_block_size=paged_block_size,
        pooling_block_size=k_block_size,
        max_num_pooling_blocks=max_num_pooling_blocks,
        dim=dim,
    )
    kernel(
        kv_cache_flat.view(torch.float8_e4m3fn),
        kv_cache_flat.view(torch.float32),
        block_tables,
        context_lens,
        blocked_k,
        blocked_k_scale,
    )
    return blocked_k, blocked_k_scale


# =====================================================================
# Phase-2 v3: pool K stored PAGED (like main KV cache)
# =====================================================================
#
# pool_k_pages[num_pool_pages_global, pool_page_size=64, D+4] uint8 —
# mirrors main KV cache layout. Each pool "token" is the mean-pool of
# k_block_size=128 real tokens; 64 pool tokens per page = 8192 real
# tokens of context per page. pool_page_tables[R, max_pool_pages_per_req]
# maps (request, logical_pool_page) to physical pool_page.
#
# Advantage over v2b: block_mqa reads pool_k_pages directly with TMA
# (same pattern as baseline paged main-KV block_mqa) — no gather pass,
# no blocked_k scratch, no tail_scratch. Completed blocks + tail are
# written IN PLACE into pool_k_pages[phys, slot_in_page, :D].


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_paged_mean_pooling_tail_only(
    paged_block_size: int,
    pooling_block_size: int,
    pool_page_size: int,
    dim: int,
    num_stages=1,
    threads=128,
):
    """Tail pool-block mean-pool, written in place into ``pool_k_pages``.

    Grid is ``(batch,)``. Per request:
      * compute ``tail_pblk = ceildiv(seq_len, K) - 1``
      * ``phys = pool_page_tables[b, tail_pblk // pool_page_size]``
      * ``slot = tail_pblk % pool_page_size``
      * mean-pool K=128 tokens from main KV cache (via BlockTables)
      * write fp8 + f32 scale into pool_k_pages[phys, slot, :]

    No separate tail scratch tensor — the downstream block_mqa reads the
    tail row from pool_k_pages just like any other pool row.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    num_blocks = T.dynamic("num_blocks")
    max_blocks = T.dynamic("max_blocks")
    batch = T.dynamic("batch")
    num_pool_pages_global = T.dynamic("num_pool_pages_global")
    max_pool_pages = T.dynamic("max_pool_pages")

    kv_cache_fp8_shape = [num_blocks, paged_block_size * (dim + 4)]
    kv_cache_fp32_shape = [num_blocks, paged_block_size * (dim + 4) // 4]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]
    pool_page_tables_shape = [batch, max_pool_pages]
    pool_k_pages_fp8_shape = [num_pool_pages_global, pool_page_size * (dim + 4)]
    pool_k_pages_fp32_shape = [num_pool_pages_global, pool_page_size * (dim + 4) // 4]

    fp8_end = paged_block_size * dim
    scale_offset = paged_block_size * dim // 4
    FP8_MAX_INV = 1.0 / 448.0

    block_N = paged_block_size  # 64
    assert pooling_block_size % block_N == 0, (
        "pooling_block_size must be a multiple of paged_block_size"
    )

    @T.prim_func
    def kernel(
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, fp8_dtype),             # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype),         # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype),              # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype),              # type: ignore
        PoolPageTables: T.Tensor(pool_page_tables_shape, index_dtype),       # type: ignore
        PoolKPagesFP8View: T.Tensor(pool_k_pages_fp8_shape, fp8_dtype),      # type: ignore
        PoolKPagesFP32View: T.Tensor(pool_k_pages_fp32_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch, threads=threads) as bx:
            b = bx
            seq_len = ContextLens[b]
            num_pool = T.ceildiv(seq_len, pooling_block_size)
            tail_pblk = num_pool - 1
            k_start = tail_pblk * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len)
            cur_pooling_block_size = k_end - k_start

            index_k_shared = T.alloc_fragment([block_N * dim], accum_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, dim])
            scale_shared = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            if num_pool > 0:
                if cur_pooling_block_size > 0:
                    for b_i in T.Serial(T.ceildiv(cur_pooling_block_size, block_N)):
                        paged_block_s = k_start + b_i * block_N
                        T.fill(index_k_shared, 0.0)

                        if paged_block_s // paged_block_size < max_blocks:
                            paged_block_phys_id = BlockTables[b, paged_block_s // paged_block_size]
                            T.copy(KvCacheFP8View[paged_block_phys_id, :fp8_end], index_k_shared)
                            T.copy(KvCacheFP32View[paged_block_phys_id, scale_offset:], scale_shared)

                        for n_i, d_i in T.Parallel(block_N, dim):
                            tl_block_idx = paged_block_s + n_i
                            index_k_reshaped[n_i, d_i] = T.cast(index_k_reshaped[n_i, d_i], accum_dtype) * scale_shared[n_i]
                            if tl_block_idx >= k_end:
                                index_k_reshaped[n_i, d_i] = T.cast(0, accum_dtype)

                        T.reduce_sum(index_k_reshaped, acc, dim=0, clear=False)

                    inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
                    for d_i in T.Parallel(dim):
                        acc[d_i] = acc[d_i] * inv_count

                T.reduce_absmax(acc, max_abs, dim=0, clear=True)
                block_scale = T.max(
                    max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype),
                    T.cast(1e-10, accum_dtype),
                )
                inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

                # Write into pool_k_pages[phys, slot, :].
                # Within a pool page (layout: [pool_page_size * (dim + 4)] bytes):
                #   fp8 row `slot` occupies bytes [slot * dim, (slot + 1) * dim)
                #   f32 scale for row `slot` at f32 index (pool_page_size * dim) // 4 + slot
                logical_page = tail_pblk // pool_page_size
                slot = tail_pblk - logical_page * pool_page_size  # == tail_pblk % pool_page_size
                phys = PoolPageTables[b, logical_page]
                fp8_row_off = slot * dim
                scale_f32_idx = pool_page_size * dim // 4 + slot
                for d_i in T.Parallel(dim):
                    PoolKPagesFP8View[phys, fp8_row_off + d_i] = T.cast(
                        acc[d_i] * inv_block_scale, fp8_dtype
                    )
                PoolKPagesFP32View[phys, scale_f32_idx] = block_scale

    return kernel


def fp8_native_paged_mean_pooling_tail_only_interface(
    kv_cache: torch.Tensor,              # [num_blocks, paged_block_size, 1, D+4] uint8
    context_lens: torch.Tensor,          # [B] int32
    block_tables: torch.Tensor,          # [B, max_blocks] int32
    pool_page_tables: torch.Tensor,      # [B, max_pool_pages] int32
    pool_k_pages: torch.Tensor,          # [N_pool_pages, pool_page_size * (D+4)] uint8 IN-OUT
    k_block_size: int,
    pool_page_size: int,
):
    num_blocks, paged_block_size, head, DPlus4 = kv_cache.shape
    assert head == 1, "Only support head=1"
    D = DPlus4 - 4
    kv_cache_flat = kv_cache.view(num_blocks, paged_block_size * DPlus4)
    assert pool_k_pages.shape[1] == pool_page_size * DPlus4

    kernel = fp8_native_paged_mean_pooling_tail_only(
        paged_block_size=paged_block_size,
        pooling_block_size=k_block_size,
        pool_page_size=pool_page_size,
        dim=D,
    )
    kernel(
        kv_cache_flat.view(torch.float8_e4m3fn),
        kv_cache_flat.view(torch.float32),
        block_tables,
        context_lens,
        pool_page_tables,
        pool_k_pages.view(torch.float8_e4m3fn),
        pool_k_pages.view(torch.float32),
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_paged_mean_pooling_completed_blocks(
    paged_block_size: int,
    pooling_block_size: int,
    pool_page_size: int,
    dim: int,
    max_pool_per_req_grid: int,
    num_stages=1,
    threads=128,
):
    """Paged-output variant of completed_blocks — writes into ``pool_k_pages``.

    Grid: ``(batch, max_pool_per_req_grid)``. Cell ``(b, pblk_rel)`` is a
    potential new-completion at absolute pool-block index
    ``pblk_abs = prev_complete + pblk_rel``. If ``pblk_rel < n_new``, the
    kernel mean-pools the covering K=128 tokens and writes:
      * ``logical_pool_page = pblk_abs // pool_page_size``
      * ``slot = pblk_abs %  pool_page_size``
      * ``phys = pool_page_tables[req_idx, logical_pool_page]``
      * ``pool_k_pages[phys, slot, :D]``  (fp8)  and scale slot.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    req_pool_idx_dtype = T.int64

    num_kv_blocks = T.dynamic("num_kv_blocks")
    batch = T.dynamic("batch")
    max_ctx = T.dynamic("max_ctx")
    max_pool_pages = T.dynamic("max_pool_pages")
    num_pool_pages_global = T.dynamic("num_pool_pages_global")
    max_running_req = T.dynamic("max_running_req")

    kv_cache_fp8_shape = [num_kv_blocks, paged_block_size * (dim + 4)]
    kv_cache_fp32_shape = [num_kv_blocks, paged_block_size * (dim + 4) // 4]
    req_to_token_shape = [max_running_req, max_ctx]
    pool_page_tables_shape = [max_running_req, max_pool_pages]
    seq_len_shape = [batch]
    pool_k_pages_fp8_shape = [num_pool_pages_global, pool_page_size * (dim + 4)]
    pool_k_pages_fp32_shape = [num_pool_pages_global, pool_page_size * (dim + 4) // 4]

    fp8_end = paged_block_size * dim
    scale_offset = paged_block_size * dim // 4
    FP8_MAX_INV = 1.0 / 448.0
    K = pooling_block_size

    block_N = paged_block_size  # 64
    assert K % block_N == 0

    @T.prim_func
    def kernel(
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, fp8_dtype),              # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype),          # type: ignore
        ReqToToken: T.Tensor(req_to_token_shape, index_dtype),                # type: ignore
        PoolPageTables: T.Tensor(pool_page_tables_shape, index_dtype),        # type: ignore
        ReqPoolIndices: T.Tensor(seq_len_shape, req_pool_idx_dtype),          # type: ignore
        PrevSeqLens: T.Tensor(seq_len_shape, index_dtype),                    # type: ignore
        NewSeqLens: T.Tensor(seq_len_shape, index_dtype),                     # type: ignore
        PoolKPagesFP8View: T.Tensor(pool_k_pages_fp8_shape, fp8_dtype),       # type: ignore
        PoolKPagesFP32View: T.Tensor(pool_k_pages_fp32_shape, accum_dtype),   # type: ignore
    ):
        with T.Kernel(batch, max_pool_per_req_grid, threads=threads) as (bx, by):
            b = bx
            pblk_rel = by

            index_k_shared = T.alloc_fragment([block_N * dim], accum_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, dim])
            scale_shared = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)

            prev_len = PrevSeqLens[b]
            new_len = NewSeqLens[b]
            prev_complete = prev_len // K
            new_complete = new_len // K
            n_new = new_complete - prev_complete

            if pblk_rel < n_new:
                pblk_abs = prev_complete + pblk_rel
                req_idx = T.cast(ReqPoolIndices[b], index_dtype)
                logical_page = pblk_abs // pool_page_size
                slot = pblk_abs - logical_page * pool_page_size
                phys = PoolPageTables[req_idx, logical_page]
                logical_start = pblk_abs * K

                T.fill(acc, 0.0)
                for b_i in T.serial(K // block_N):
                    chunk_logical_start = logical_start + b_i * block_N
                    T.fill(index_k_shared, 0.0)

                    buf_pos = ReqToToken[req_idx, chunk_logical_start]
                    phys_page = buf_pos // paged_block_size

                    T.copy(KvCacheFP8View[phys_page, :fp8_end], index_k_shared)
                    T.copy(KvCacheFP32View[phys_page, scale_offset:], scale_shared)

                    for n_i, d_i in T.Parallel(block_N, dim):
                        index_k_reshaped[n_i, d_i] = T.cast(
                            index_k_reshaped[n_i, d_i], accum_dtype
                        ) * scale_shared[n_i]

                    T.reduce_sum(index_k_reshaped, acc, dim=0, clear=False)

                inv_count = T.cast(1.0, accum_dtype) / T.cast(K, accum_dtype)
                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] * inv_count

                T.reduce_absmax(acc, max_abs, dim=0, clear=True)
                block_scale = T.max(
                    max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype),
                    T.cast(1e-10, accum_dtype),
                )
                inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

                fp8_row_off = slot * dim
                scale_f32_idx = pool_page_size * dim // 4 + slot
                for d_i in T.Parallel(dim):
                    PoolKPagesFP8View[phys, fp8_row_off + d_i] = T.cast(
                        acc[d_i] * inv_block_scale, fp8_dtype
                    )
                PoolKPagesFP32View[phys, scale_f32_idx] = block_scale

    return kernel


def fp8_native_paged_mean_pooling_completed_blocks_interface(
    kv_cache_flat: torch.Tensor,           # [num_pages, page_size * (D + 4)] uint8
    req_to_token: torch.Tensor,            # [R, T] int32
    pool_page_tables: torch.Tensor,        # [R, max_pool_pages] int32
    req_pool_indices: torch.Tensor,        # [B] int64
    prev_seq_lens: torch.Tensor,           # [B] int32
    new_seq_lens: torch.Tensor,            # [B] int32
    pool_k_pages: torch.Tensor,            # [N_pool_pages, pool_page_size * (D+4)] uint8 IN-OUT
    k_block_size: int,
    paged_block_size: int,
    pool_page_size: int,
    max_pool_per_req_grid: int,
):
    _, DPlus4_times_P = kv_cache_flat.shape
    D = DPlus4_times_P // paged_block_size - 4
    assert pool_k_pages.shape[1] == pool_page_size * (D + 4)

    kernel = fp8_native_paged_mean_pooling_completed_blocks(
        paged_block_size=paged_block_size,
        pooling_block_size=k_block_size,
        pool_page_size=pool_page_size,
        dim=D,
        max_pool_per_req_grid=max_pool_per_req_grid,
    )
    kernel(
        kv_cache_flat.view(torch.float8_e4m3fn),
        kv_cache_flat.view(torch.float32),
        req_to_token,
        pool_page_tables,
        req_pool_indices,
        prev_seq_lens,
        new_seq_lens,
        pool_k_pages.view(torch.float8_e4m3fn),
        pool_k_pages.view(torch.float32),
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def batch_decode_pool_mqa_attn_return_logits_fp8(
    heads: int,
    index_dim: int,
    pool_page_size: int = 64,
    block_H: int = 64,
    num_stages: int = 3,
    threads: int = 128,
):
    """Decode block-MQA reading pool_k_pages in a paged, TMA-friendly way.

    Grid: ``(batch, 1)``. Inner pipeline over ``max_pool_pages`` — each
    iteration TMA-reads one pool page (``pool_page_size=64`` rows) from
    ``pool_k_pages[phys, :]`` and performs the fp8×fp8 GEMM. Same pattern
    as baseline paged block_mqa on the main KV cache.

    No gather, no scratch — the pool rows are already laid out contiguously
    within a page because the v3 allocator allocates pages, not individual
    block IDs.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    num_pool_pages_global = T.dynamic("num_pool_pages_global")
    max_pool_pages = T.dynamic("max_pool_pages")

    block_N = pool_page_size

    q_shape = [batch, heads, index_dim]
    pool_k_pages_fp8_shape = [num_pool_pages_global, pool_page_size * (index_dim + 4)]
    pool_k_pages_fp32_shape = [num_pool_pages_global, pool_page_size * (index_dim + 4) // 4]
    pool_page_tables_shape = [batch, max_pool_pages]
    logits_shape = [batch, max_pool_pages * pool_page_size]
    w_shape = [batch, heads]

    fp8_end = pool_page_size * index_dim
    scale_offset = pool_page_size * index_dim // 4

    block_H_pad = T.ceildiv(block_H, 16) * 16
    assert block_H_pad == heads

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, fp8_dtype),                                    # type: ignore
        PoolKPagesFP8View: T.Tensor(pool_k_pages_fp8_shape, fp8_dtype),     # type: ignore
        PoolKPagesFP32View: T.Tensor(pool_k_pages_fp32_shape, accum_dtype), # type: ignore
        PoolPageTables: T.Tensor(pool_page_tables_shape, index_dtype),      # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),                        # type: ignore
        Weights: T.Tensor(w_shape, accum_dtype),                            # type: ignore
        ContextLensPool: T.Tensor([batch], index_dtype),                    # type: ignore
    ):
        with T.Kernel(batch, 1, threads=threads) as (bx, by):
            k_shared = T.alloc_shared([block_N * index_dim], fp8_dtype)
            k_reshaped = T.reshape(k_shared, [block_N, index_dim])
            k_scale_shared = T.alloc_shared([block_N], accum_dtype)
            q_shared = T.alloc_shared([block_H_pad, index_dim], fp8_dtype)

            s = T.alloc_fragment([block_N, block_H_pad], accum_dtype)
            w = T.alloc_fragment([block_H_pad], accum_dtype)
            logits_accum = T.alloc_fragment([block_N], accum_dtype)

            k_e = ContextLensPool[bx]  # num pool blocks for this request
            T.copy(Q[bx, 0, 0], q_shared)
            T.copy(Weights[bx, 0], w)

            for lp in T.Pipelined(max_pool_pages, num_stages=num_stages):
                phys = PoolPageTables[bx, lp]
                # Bulk LDG of one pool page's fp8 K rows. disable_tma=True
                # matches baseline sparse_paged pattern — 1D shared + paged
                # indirection isn't a TMA-compatible layout in tilelang.
                T.copy(PoolKPagesFP8View[phys, :fp8_end], k_shared, disable_tma=True)
                for bn_i in T.Parallel(block_N):
                    k_scale_shared[bn_i] = PoolKPagesFP32View[phys, scale_offset + bn_i]

                T.gemm(
                    k_reshaped,
                    q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for kn_i, hn_i in T.Parallel(block_N, block_H_pad):
                    s[kn_i, hn_i] = T.max(s[kn_i, hn_i] * k_scale_shared[kn_i], 0) * w[hn_i]

                T.reduce_sum(s, logits_accum, dim=1, clear=True)

                for kn_i in T.Parallel(block_N):
                    pool_idx = lp * pool_page_size + kn_i
                    # Fused mask + force_maintain: pos 0 and last valid pos get +inf.
                    if pool_idx == 0 or pool_idx == k_e - 1:
                        Logits[bx, pool_idx] = T.infinity(accum_dtype)
                    elif pool_idx < k_e:
                        Logits[bx, pool_idx] = logits_accum[kn_i]
                    else:
                        Logits[bx, pool_idx] = -T.infinity(accum_dtype)

    return kernel


def batch_pool_mqa_attn_return_logits_fp8_interface(
    q_fp8: torch.Tensor,                 # [B, 1, H, D] fp8
    pool_k_pages: torch.Tensor,          # [N_pool_pages, pool_page_size * (D+4)] uint8
    pool_page_tables: torch.Tensor,      # [B, max_pool_pages] int32
    weights_f32: torch.Tensor,           # [B, H] f32 (or [B*1, H])
    context_lens_pool: torch.Tensor,     # [B] int32 — num pool blocks per req
    *,
    pool_page_size: int = 64,
):
    """v3 decode block-MQA interface. Returns [B, 1, max_pool_pages * pool_page_size] f32."""
    assert len(q_fp8.shape) == 4
    B, seq_len_q, H, D = q_fp8.shape
    assert seq_len_q == 1

    max_pool_pages = pool_page_tables.shape[-1]
    assert pool_page_tables.shape == (B, max_pool_pages)
    assert pool_k_pages.shape[1] == pool_page_size * (D + 4), (
        f"pool_k_pages row bytes {pool_k_pages.shape[1]} != "
        f"pool_page_size * (D + 4) = {pool_page_size * (D + 4)}"
    )

    q_2d = q_fp8.squeeze(1)                  # [B, H, D] fp8
    w_2d = weights_f32.view(B, H)            # [B, H] f32

    pool_k_fp8_view = pool_k_pages.view(torch.float8_e4m3fn)
    pool_k_f32_view = pool_k_pages.view(torch.float32)

    logits = torch.empty(
        (B, max_pool_pages * pool_page_size), device=q_fp8.device, dtype=torch.float32,
    )
    kernel = batch_decode_pool_mqa_attn_return_logits_fp8(
        heads=H, index_dim=D, pool_page_size=pool_page_size, block_H=H,
    )
    assert context_lens_pool.dtype == torch.int32
    assert pool_page_tables.dtype == torch.int32
    kernel(
        q_2d, pool_k_fp8_view, pool_k_f32_view,
        pool_page_tables, logits, w_2d, context_lens_pool,
    )
    return logits.unsqueeze(1)  # [B, 1, max_pool_pages * pool_page_size]


def fp8_native_hierarchy_paged_mqa_logits_tilelang_with_pool_cache(
    q_fp8: torch.Tensor,                # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,         # [num_blocks, paged_block_size, 1, D+4] uint8
    pool_k_pages: torch.Tensor,         # [N_pool_pages, pool_page_size, D+4] uint8 (cached)
    pool_page_tables: torch.Tensor,     # [B, max_pool_pages] int32
    weights: torch.Tensor,              # [B*1, H] f32
    context_lens: torch.Tensor,         # [B] int32 — raw seq_len per request
    block_tables: torch.Tensor,         # [B, max_kv_blocks] int32
    k_block_size: int,
    pool_page_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end v3 paged hierarchy-MQA. No gather, no blocked_k scratch.

    Pool rows are stored IN PAGES (layout identical to main KV cache),
    so block_mqa reads them via TMA just like baseline paged attention.
    Tail is refreshed in-place into pool_k_pages[phys_last, tail_slot, :].
    """
    # 1) Refresh tail pool block in place.
    fp8_native_paged_mean_pooling_tail_only_interface(
        kv_cache=kv_cache_fp8, context_lens=context_lens,
        block_tables=block_tables,
        pool_page_tables=pool_page_tables,
        pool_k_pages=pool_k_pages,
        k_block_size=k_block_size,
        pool_page_size=pool_page_size,
    )

    # 2) Block-MQA directly on pool_k_pages (paged, TMA-friendly).
    num_pool_blocks_per_req = (context_lens + k_block_size - 1) // k_block_size
    block_k_indexer_score = batch_pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q_fp8,
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        weights_f32=weights,
        context_lens_pool=num_pool_blocks_per_req,
        pool_page_size=pool_page_size,
    )  # [B, 1, max_pool_pages * pool_page_size]

    # 3) Top-k over pool blocks.
    topk_block_indices = torch.topk(
        block_k_indexer_score,
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices

    # 4) Sparse paged MQA on the selected blocks (identical to v1 / v2b).
    block_sparse_k_indexer_score = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
        q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices, kv_block_size=k_block_size,
        weights=weights, context_lens=context_lens, block_tables=block_tables,
    )
    return block_sparse_k_indexer_score, topk_block_indices


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_paged_block_sparse_mqa_attn_return_logits(
    paged_block_size,
    kv_block_size,
    topk,
    heads,
    index_dim,
    num_stages=1,
    threads=128,
    dtype="bfloat16",
):
    """Decode paged sparse-MQA with fp8 Q/K smem + fp8×fp8 GEMM + FullRow.

    block_N = paged_block_size = 64 → with threads=128 (4 warps), M/4 = 16 =
    WMMA m-tile.  K scale applied post-GEMM.  1D scale load via T.Parallel
    to avoid TMA alignment issues seen in cudagraph capture with B=1.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    # TopKBlockIndex is int64 (torch.topk's native output) — avoids one
    # int64→int32 cast kernel between hierarchy_caller and this op.
    topk_index_dtype = T.int64

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_blocks = T.dynamic("max_blocks")
    num_phys_blocks = T.dynamic("num_phys_blocks")

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_fp8_shape = [num_phys_blocks, paged_block_size * (index_dim + 4)]
    kv_cache_fp32_shape = [num_phys_blocks, paged_block_size * (index_dim + 4) // 4]
    logits_shape = [batch, seq_len, topk * kv_block_size]
    weights_shape = [batch, seq_len, heads]

    fp8_end = paged_block_size * index_dim
    scale_offset = paged_block_size * index_dim // 4

    H_per_block = heads
    block_N = paged_block_size
    assert block_N > 0, "block_N must be positive"
    assert kv_block_size >= block_N and kv_block_size % block_N == 0, "block_N must divide kv_block_size"
    assert paged_block_size >= block_N and paged_block_size % block_N == 0, "block_N must divide paged_block_size"
    assert paged_block_size == block_N, "for simplicity we require paged_block_size == block_N in this kernel"

    @T.prim_func
    def fp8_native_paged_block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, fp8_dtype),  # type: ignore
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, fp8_dtype),  # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], topk_index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, accum_dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            # fp8 Q/K smem: 8KB + 8KB = 16KB vs prior 32+32=64KB (f32).
            index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
            # 1D fp8 smem matches 1D K source slice (SPLIT layout); reshape to
            # 2D for the GEMM op.  T.copy uses disable_tma=True to force LDG.
            index_k_shared = T.alloc_shared([block_N * index_dim], fp8_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, index_dim])
            # Scale stays in shared here: the load happens inside a conditional
            # (out-of-range paged block is skipped). Shared is zero-initialized
            # at kernel launch so a skipped load gracefully leaves 0 and the
            # post-GEMM product is 0. A fragment would hold uninitialized
            # registers on skip, causing non-deterministic decode output.
            index_k_scale_shared = T.alloc_shared([block_N], accum_dtype)

            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            T.copy(Weights[b, seq_len_i, :], weights)

            for n_i in T.serial(topk):
                # Cast to int32: downstream uses (BlockTables indexing, T.copy
                # slice bounds) require int32 to match their operand types in
                # TileLang's tile-op lowering.
                topk_block_id = T.cast(TopKBlockIndex[b, seq_len_i, n_i], index_dtype)
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N

                    if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                        phys = BlockTables[b, block_s_i // paged_block_size]
                        # disable_tma forces vectorized LDG (TMA needs same-dtype
                        # 2D layout that's incompatible with this 1D K slice).
                        T.copy(KvCacheFP8View[phys, :fp8_end], index_k_shared, disable_tma=True)
                        # 1D scale load via T.Parallel (TMA 1D alignment unsafe).
                        for bn_i in T.Parallel(block_N):
                            index_k_scale_shared[bn_i] = KvCacheFP32View[phys, scale_offset + bn_i]

                    # fp8 × fp8 → f32 GEMM; K scale applied post-GEMM.
                    T.gemm(
                        index_k_reshaped,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * index_k_scale_shared[bn_i], 0) * weights[bq_i, h_i])

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        p = k_i // paged_block_size
                        if (k_i < cu_k_s_min) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[b, seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0]

    @T.prim_func
    def fp8_native_paged_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size(
        IndexQ: T.Tensor(index_q_shape, fp8_dtype),  # type: ignore
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, fp8_dtype),  # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], topk_index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, accum_dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, seq_len, threads=threads) as (bx, by):
            b = bx
            seq_len_i = by

            # fp8 Q/K smem.
            index_q_shared = T.alloc_shared([H_per_block, index_dim], fp8_dtype)
            # 1D fp8 smem matches 1D K source slice; reshape for GEMM.
            index_k_shared = T.alloc_shared([block_N * index_dim], fp8_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, index_dim])
            # Scale stays in shared (zero-init'd) because the load is guarded
            # by an out-of-range conditional — fragment would hold garbage on
            # skip and break decode determinism.
            index_k_scale_shared = T.alloc_shared([block_N], accum_dtype)

            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            cu_k_s_min = T.cast(0, index_dtype)
            cu_k_e_max = ContextLens[b]

            T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            T.copy(Weights[b, seq_len_i, :], weights)

            for n_i in T.serial(topk):
                # Cast to int32: downstream uses (BlockTables indexing, T.copy
                # slice bounds) require int32 to match their operand types in
                # TileLang's tile-op lowering.
                topk_block_id = T.cast(TopKBlockIndex[b, seq_len_i, n_i], index_dtype)
                block_s_i = topk_block_id * kv_block_size

                if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                    phys = BlockTables[b, block_s_i // paged_block_size]
                    T.copy(KvCacheFP8View[phys, :fp8_end], index_k_shared, disable_tma=True)
                    for bn_i in T.Parallel(block_N):
                        index_k_scale_shared[bn_i] = KvCacheFP32View[phys, scale_offset + bn_i]

                T.gemm(
                    index_k_reshaped,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block // heads, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i] * index_k_scale_shared[bn_i], 0) * weights[bq_i, h_i])

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for i_i in T.Parallel(block_N):
                    k_i = block_s_i + i_i
                    p = k_i // paged_block_size
                    if (k_i < cu_k_s_min) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                        logits[i_i, 0] = -T.infinity(accum_dtype)

                for bn_i in T.Parallel(block_N):
                    Logits[b, seq_len_i, n_i * kv_block_size + bn_i] = logits[bn_i, 0]

    if kv_block_size == block_N:
        return fp8_native_paged_block_sparse_mqa_attn_return_logits_kernel_for_small_pooling_size
    else:
        return fp8_native_paged_block_sparse_mqa_attn_return_logits_kernel

def fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
    q_fp8,
    kv_cache_fp8,
    topk_block_index,
    kv_block_size,
    weights,
    context_lens,
    block_tables,
    dtype="bfloat16",
):
    batch, seq_len, heads, index_dim = q_fp8.shape
    topk = int(topk_block_index.shape[-1])

    num_blocks, paged_block_size, _, D_plus_4 = kv_cache_fp8.shape
    assert _ == 1, "Only support head=1 for k in indexer"
    assert D_plus_4 - 4 == index_dim, f"Expected kv_cache last dim to be index_dim+4, but got {D_plus_4} vs {index_dim+4}"

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    kv_cache_flat = kv_cache_fp8.view(num_blocks, -1)
    kv_cache_fp8_2d = kv_cache_flat.view(torch.float8_e4m3fn)
    kv_cache_f32 = kv_cache_flat.view(torch.float32)
    logits = torch.empty(
        (batch, seq_len, topk * kv_block_size),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    kernel = fp8_native_paged_block_sparse_mqa_attn_return_logits(
        paged_block_size=paged_block_size,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        dtype=dtype,
    )
    # topk_block_index is int64 (torch.topk's native output); other indices are
    # already int32 from vLLM.  Skipping .to() saves Python-level no-op casts
    # (~5-15μs).  Asserts catch caller bugs early.
    assert topk_block_index.dtype == torch.int64, f"topk_block_index must be int64, got {topk_block_index.dtype}"
    assert context_lens.dtype == torch.int32, f"context_lens must be int32, got {context_lens.dtype}"
    assert block_tables.dtype == torch.int32, f"block_tables must be int32, got {block_tables.dtype}"
    kernel(
        q_fp8,
        kv_cache_fp8_2d,
        kv_cache_f32,
        topk_block_index,
        logits,
        weights,
        context_lens,
        block_tables,
    )
    return logits

def fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    max_seq_len: int,
    k_block_size: int, 
    block_topk: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    fp8_dtype = q_fp8.dtype
    dim = q_fp8.shape[-1]
    num_blocks, block_size, _, D_plus_4 = kv_cache_fp8.shape

    max_num_pooling_blocks = (max_seq_len + k_block_size - 1) // k_block_size
    # Mean pool outputs fp8 + per-block f32 scale; pool_mqa_fp8 consumes
    # them directly via fp8×fp8 GEMM (no Python dequant, no bf16 cast).
    blocked_k_fp8, blocked_k_scale, num_pooling_blocks = fp8_native_paged_mean_pooling_interface(max_num_pooling_blocks, kv_cache_fp8, context_lens, block_tables, k_block_size)

    block_k_indexer_score = batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
        q_fp8=q_fp8,
        blocked_kv_fp8=blocked_k_fp8,
        blocked_kv_scale=blocked_k_scale,
        weights_f32=weights,
        context_lens=num_pooling_blocks,
        kv_block_size=k_block_size,
    )  # [B, next_n, num_pooling_blocks]
    # sorted=False skips internal sort (sparse_mqa doesn't need order); bf16 cast
    # not worth it here because block_k_indexer_score is small (B × Nb, B ≤ 64).
    # torch.topk returns int64; paged sparse_mqa now accepts int64 directly.
    topk_block_indices = torch.topk(block_k_indexer_score, k=min(block_topk, block_k_indexer_score.shape[-1]), dim=-1, sorted=False).indices  # [B, next_n, topk] int64

    block_sparse_k_indexer_score = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8, topk_block_index=topk_block_indices, kv_block_size=k_block_size, weights=weights, context_lens=context_lens, block_tables=block_tables)  # [B, next_n, topk*kv_block_size]

    return block_sparse_k_indexer_score, topk_block_indices


def fp8_native_hierarchy_paged_mqa_logits_with_pool_cache(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    blocked_k_cache: torch.Tensor,         # [B, max_num_pooling_blocks, D] fp8   IN-OUT
    blocked_k_scale_cache: torch.Tensor,   # [B, max_num_pooling_blocks]    f32   IN-OUT
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    k_block_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache-based variant of ``fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy``.

    Caller maintains ``blocked_k_cache`` + ``blocked_k_scale_cache`` across
    decode steps. On entry, positions ``[b, :num_pool[b]-1]`` must hold
    valid pooled values from prior steps; the tail slot
    ``[b, num_pool[b]-1]`` is overwritten here. Saves the full paged mean
    pool cost — only the tail block (up to ``k_block_size`` tokens) is
    recomputed each decode step.
    """
    # 1) Refresh tail block from raw KV cache (in place).
    fp8_native_paged_mean_pooling_tail_only_legacy_interface(
        blocked_k_cache, blocked_k_scale_cache,
        kv_cache_fp8, context_lens, block_tables, k_block_size,
    )
    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size

    # 2) block-MQA on the (now-fresh) cached pooled K.
    block_k_indexer_score = batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
        q_fp8=q_fp8,
        blocked_kv_fp8=blocked_k_cache,
        blocked_kv_scale=blocked_k_scale_cache,
        weights_f32=weights,
        context_lens=num_pooling_blocks,
        kv_block_size=k_block_size,
    )

    # 3) top-k over blocks.
    topk_block_indices = torch.topk(
        block_k_indexer_score,
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices

    # 4) fine-grained paged sparse MQA on the selected blocks.
    block_sparse_k_indexer_score = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
        q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices, kv_block_size=k_block_size,
        weights=weights, context_lens=context_lens, block_tables=block_tables,
    )

    return block_sparse_k_indexer_score, topk_block_indices
