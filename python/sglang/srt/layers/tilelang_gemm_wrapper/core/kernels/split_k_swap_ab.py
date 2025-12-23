"""Split-K + SwapAB FP8 Blockwise GEMM Kernel.

A_scale: (M//128, K//128) per-block, B_scale: (N, K//128) per-token-group
Output: C (N, M) transposed
"""
import tilelang
import tilelang.language as T


@tilelang.jit
def kernel_factory(
    M,
    N,
    K,
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    threads=None,
    split_k=1,
    out_dtype="bfloat16",
    accum_dtype="float32",
    c_scale_local=False,
    b_scale_shm=False,
):
    group_size = 128
    A_scale_shape = (T.ceildiv(M, group_size), T.ceildiv(K, group_size))
    B_scale_shape = (N, T.ceildiv(K, group_size))
    K_per_split = K // split_k
    c_scale_alloc = T.alloc_fragment if c_scale_local else T.alloc_shared

    @T.macro
    def split_gemm(
        A: T.Tensor((M, K), "float8_e4m3"),
        B: T.Tensor((N, K), "float8_e4m3"),
        C_partial: T.Tensor((split_k, N, M), accum_dtype),
        a_scale: T.Tensor(A_scale_shape, "float32"),
        b_scale: T.Tensor(B_scale_shape, "float32"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=threads
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), "float8_e4m3")
            B_shared = T.alloc_shared((block_N, block_K), "float8_e4m3")
            C_shared = T.alloc_shared((block_N, block_M), accum_dtype)
            C_scale = c_scale_alloc((block_N,), "float32")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_N, block_M), accum_dtype)

            if b_scale_shm:
                B_scale_shared = T.alloc_shared((block_N,), "float32")

            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K_per_split, block_K)
            for ko in T.Pipelined(K_iters, num_stages=num_stages):
                k_offset = bz * K_per_split + ko * block_K
                scale_k = bz * (K_per_split // group_size) + ko

                T.copy(A[by * block_M, k_offset], A_shared)
                T.copy(B[bx * block_N, k_offset], B_shared)

                if b_scale_shm:
                    for i in T.Parallel(block_N):
                        B_scale_shared[i] = b_scale[bx * block_N + i, scale_k]
                    A_scale = a_scale[by * block_M // group_size, scale_k]
                    for i in T.Parallel(block_N):
                        C_scale[i] = B_scale_shared[i] * A_scale
                else:
                    A_scale = a_scale[by * block_M // group_size, scale_k]
                    for i in T.Parallel(block_N):
                        C_scale[i] = b_scale[bx * block_N + i, scale_k] * A_scale

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[j, i] += C_local[i, j] * C_scale[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C_partial[bz, bx * block_N, by * block_M])

    @T.macro
    def combine(
        C_partial: T.Tensor((split_k, N, M), accum_dtype),
        C: T.Tensor((N, M), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            C_local = T.alloc_fragment((block_N, block_M), accum_dtype)
            C_partial_local = T.alloc_fragment((block_N, block_M), accum_dtype)
            C_shared = T.alloc_shared((block_N, block_M), out_dtype)

            T.clear(C_local)
            for s in range(split_k):
                T.copy(C_partial[s, bx * block_N, by * block_M], C_partial_local)
                for i, j in T.Parallel(block_N, block_M):
                    C_local[i, j] += C_partial_local[i, j]

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_N, by * block_M])

    @T.prim_func
    def tilelang_fp8_blockwise_splitk_swapab(
        A: T.Tensor((M, K), "float8_e4m3"),
        B: T.Tensor((N, K), "float8_e4m3"),
        C_partial: T.Tensor((split_k, N, M), accum_dtype),
        C: T.Tensor((N, M), out_dtype),
        a_scale: T.Tensor(A_scale_shape, "float32"),
        b_scale: T.Tensor(B_scale_shape, "float32"),
    ):
        split_gemm(A, B, C_partial, a_scale, b_scale)
        combine(C_partial, C)

    return tilelang_fp8_blockwise_splitk_swapab
