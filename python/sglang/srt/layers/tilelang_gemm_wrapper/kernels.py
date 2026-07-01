"""TileLang kernels for FP8 W8A8 blockwise GEMM."""

import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

FP8_DTYPE = "float8_e4m3"
BF16_DTYPE = "bfloat16"
FP32_DTYPE = "float32"
GROUP_SIZE = 128

# Keep these kernel families as separate TileLang entrypoints. The JIT specializes
# function arguments into generated code, and the base/swapAB/splitK variants have
# different tensor signatures and launch orders.


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def fp8_blockwise_gemm_base_kernel(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: str = BF16_DTYPE,
    accum_dtype: str = FP32_DTYPE,
    c_scale_local: bool = False,
    a_scale_shm: bool = False,
    swizzle_panel: int = 0,
    swizzle_order: str = "row",
):
    M = T.symbolic("M")
    c_scale_alloc = T.alloc_fragment if c_scale_local else T.alloc_shared

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), FP8_DTYPE),
        A_scale: T.Tensor((M, T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE),
        B: T.Tensor((N, K), FP8_DTYPE),
        B_scale: T.Tensor(
            (T.ceildiv(N, GROUP_SIZE), T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE
        ),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (pid_n, pid_m):
            if swizzle_panel > 0:
                T.use_swizzle(swizzle_panel, order=swizzle_order)

            A_shared = T.alloc_shared((block_M, block_K), FP8_DTYPE)
            B_shared = T.alloc_shared((block_N, block_K), FP8_DTYPE)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_scale = c_scale_alloc((block_M,), FP32_DTYPE)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            if a_scale_shm:
                A_scale_shared = T.alloc_shared((block_M,), FP32_DTYPE)

            T.clear(C_local)
            T.clear(C_local_accum)

            for k_iter in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[pid_m * block_M, k_iter * block_K], A_shared)
                T.copy(B[pid_n * block_N, k_iter * block_K], B_shared)

                if a_scale_shm:
                    for i in T.Parallel(block_M):
                        A_scale_shared[i] = A_scale[pid_m * block_M + i, k_iter]
                    b_scale = B_scale[pid_n * block_N // GROUP_SIZE, k_iter]
                    for i in T.Parallel(block_M):
                        C_scale[i] = A_scale_shared[i] * b_scale
                else:
                    b_scale = B_scale[pid_n * block_N // GROUP_SIZE, k_iter]
                    for i in T.Parallel(block_M):
                        C_scale[i] = A_scale[pid_m * block_M + i, k_iter] * b_scale

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * C_scale[i]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])

    return kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def fp8_blockwise_gemm_swap_ab_kernel(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: str = BF16_DTYPE,
    accum_dtype: str = FP32_DTYPE,
    c_scale_local: bool = False,
    b_scale_shm: bool = False,
):
    M = T.symbolic("M")
    c_scale_alloc = T.alloc_fragment if c_scale_local else T.alloc_shared

    @T.prim_func
    def kernel(
        A: T.Tensor((N, K), FP8_DTYPE),
        A_scale: T.Tensor(
            (T.ceildiv(N, GROUP_SIZE), T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE
        ),
        B: T.Tensor((M, K), FP8_DTYPE),
        B_scale: T.Tensor((M, T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_N), T.ceildiv(N, block_M), threads=threads
        ) as (pid_m, pid_n):
            A_shared = T.alloc_shared((block_M, block_K), FP8_DTYPE)
            B_shared = T.alloc_shared((block_N, block_K), FP8_DTYPE)
            C_shared = T.alloc_shared((block_N, block_M), out_dtype)
            C_scale = c_scale_alloc((block_N,), FP32_DTYPE)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_N, block_M), accum_dtype)

            if b_scale_shm:
                B_scale_shared = T.alloc_shared((block_N,), FP32_DTYPE)

            T.clear(C_local)
            T.clear(C_local_accum)

            for k_iter in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[pid_n * block_M, k_iter * block_K], A_shared)
                T.copy(B[pid_m * block_N, k_iter * block_K], B_shared)

                if b_scale_shm:
                    for i in T.Parallel(block_N):
                        B_scale_shared[i] = B_scale[pid_m * block_N + i, k_iter]
                    a_scale = A_scale[pid_n * block_M // GROUP_SIZE, k_iter]
                    for i in T.Parallel(block_N):
                        C_scale[i] = B_scale_shared[i] * a_scale
                else:
                    a_scale = A_scale[pid_n * block_M // GROUP_SIZE, k_iter]
                    for i in T.Parallel(block_N):
                        C_scale[i] = B_scale[pid_m * block_N + i, k_iter] * a_scale

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[j, i] += C_local[i, j] * C_scale[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[pid_m * block_N, pid_n * block_M])

    return kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def fp8_blockwise_gemm_split_k_kernel(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    split_k: int = 2,
    out_dtype: str = BF16_DTYPE,
    accum_dtype: str = FP32_DTYPE,
    c_scale_local: bool = False,
    a_scale_shm: bool = False,
):
    M = T.symbolic("M")
    k_per_split = K // split_k
    c_scale_alloc = T.alloc_fragment if c_scale_local else T.alloc_shared

    @T.macro
    def split_gemm(
        A: T.Tensor((M, K), FP8_DTYPE),
        A_scale: T.Tensor((M, T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE),
        B: T.Tensor((N, K), FP8_DTYPE),
        B_scale: T.Tensor(
            (T.ceildiv(N, GROUP_SIZE), T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE
        ),
        C_partial: T.Tensor((split_k, M, N), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=threads
        ) as (pid_n, pid_m, pid_k):
            A_shared = T.alloc_shared((block_M, block_K), FP8_DTYPE)
            B_shared = T.alloc_shared((block_N, block_K), FP8_DTYPE)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_scale = c_scale_alloc((block_M,), FP32_DTYPE)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            if a_scale_shm:
                A_scale_shared = T.alloc_shared((block_M,), FP32_DTYPE)

            T.clear(C_local)
            T.clear(C_local_accum)

            for k_iter in T.Pipelined(
                T.ceildiv(k_per_split, block_K), num_stages=num_stages
            ):
                k_offset = pid_k * k_per_split + k_iter * block_K
                scale_k = pid_k * (k_per_split // GROUP_SIZE) + k_iter

                T.copy(A[pid_m * block_M, k_offset], A_shared)
                T.copy(B[pid_n * block_N, k_offset], B_shared)

                if a_scale_shm:
                    for i in T.Parallel(block_M):
                        A_scale_shared[i] = A_scale[pid_m * block_M + i, scale_k]
                    b_scale = B_scale[pid_n * block_N // GROUP_SIZE, scale_k]
                    for i in T.Parallel(block_M):
                        C_scale[i] = A_scale_shared[i] * b_scale
                else:
                    b_scale = B_scale[pid_n * block_N // GROUP_SIZE, scale_k]
                    for i in T.Parallel(block_M):
                        C_scale[i] = A_scale[pid_m * block_M + i, scale_k] * b_scale

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * C_scale[i]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C_partial[pid_k, pid_m * block_M, pid_n * block_N])

    @T.macro
    def combine(
        C_partial: T.Tensor((split_k, M, N), accum_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (pid_n, pid_m):
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_partial_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.clear(C_local)
            for split_idx in range(split_k):
                T.copy(
                    C_partial[split_idx, pid_m * block_M, pid_n * block_N],
                    C_partial_local,
                )
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] += C_partial_local[i, j]

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), FP8_DTYPE),
        A_scale: T.Tensor((M, T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE),
        B: T.Tensor((N, K), FP8_DTYPE),
        B_scale: T.Tensor(
            (T.ceildiv(N, GROUP_SIZE), T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE
        ),
        C_partial: T.Tensor((split_k, M, N), accum_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        split_gemm(A, A_scale, B, B_scale, C_partial)
        combine(C_partial, C)

    return kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def fp8_blockwise_gemm_split_k_swap_ab_kernel(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    split_k: int = 2,
    out_dtype: str = BF16_DTYPE,
    accum_dtype: str = FP32_DTYPE,
    c_scale_local: bool = False,
    b_scale_shm: bool = False,
):
    M = T.symbolic("M")
    k_per_split = K // split_k
    c_scale_alloc = T.alloc_fragment if c_scale_local else T.alloc_shared

    @T.macro
    def split_gemm(
        A: T.Tensor((N, K), FP8_DTYPE),
        A_scale: T.Tensor(
            (T.ceildiv(N, GROUP_SIZE), T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE
        ),
        B: T.Tensor((M, K), FP8_DTYPE),
        B_scale: T.Tensor((M, T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE),
        C_partial: T.Tensor((split_k, M, N), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_N), T.ceildiv(N, block_M), split_k, threads=threads
        ) as (pid_m, pid_n, pid_k):
            A_shared = T.alloc_shared((block_M, block_K), FP8_DTYPE)
            B_shared = T.alloc_shared((block_N, block_K), FP8_DTYPE)
            C_shared = T.alloc_shared((block_N, block_M), accum_dtype)
            C_scale = c_scale_alloc((block_N,), FP32_DTYPE)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_N, block_M), accum_dtype)

            if b_scale_shm:
                B_scale_shared = T.alloc_shared((block_N,), FP32_DTYPE)

            T.clear(C_local)
            T.clear(C_local_accum)

            for k_iter in T.Pipelined(
                T.ceildiv(k_per_split, block_K), num_stages=num_stages
            ):
                k_offset = pid_k * k_per_split + k_iter * block_K
                scale_k = pid_k * (k_per_split // GROUP_SIZE) + k_iter

                T.copy(A[pid_n * block_M, k_offset], A_shared)
                T.copy(B[pid_m * block_N, k_offset], B_shared)

                if b_scale_shm:
                    for i in T.Parallel(block_N):
                        B_scale_shared[i] = B_scale[pid_m * block_N + i, scale_k]
                    a_scale = A_scale[pid_n * block_M // GROUP_SIZE, scale_k]
                    for i in T.Parallel(block_N):
                        C_scale[i] = B_scale_shared[i] * a_scale
                else:
                    a_scale = A_scale[pid_n * block_M // GROUP_SIZE, scale_k]
                    for i in T.Parallel(block_N):
                        C_scale[i] = B_scale[pid_m * block_N + i, scale_k] * a_scale

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[j, i] += C_local[i, j] * C_scale[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C_partial[pid_k, pid_m * block_N, pid_n * block_M])

    @T.macro
    def combine(
        C_partial: T.Tensor((split_k, M, N), accum_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_N), T.ceildiv(N, block_M), threads=threads
        ) as (pid_m, pid_n):
            C_local = T.alloc_fragment((block_N, block_M), accum_dtype)
            C_partial_local = T.alloc_fragment((block_N, block_M), accum_dtype)
            C_shared = T.alloc_shared((block_N, block_M), out_dtype)

            T.clear(C_local)
            for split_idx in range(split_k):
                T.copy(
                    C_partial[split_idx, pid_m * block_N, pid_n * block_M],
                    C_partial_local,
                )
                for i, j in T.Parallel(block_N, block_M):
                    C_local[i, j] += C_partial_local[i, j]

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[pid_m * block_N, pid_n * block_M])

    @T.prim_func
    def kernel(
        A: T.Tensor((N, K), FP8_DTYPE),
        A_scale: T.Tensor(
            (T.ceildiv(N, GROUP_SIZE), T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE
        ),
        B: T.Tensor((M, K), FP8_DTYPE),
        B_scale: T.Tensor((M, T.ceildiv(K, GROUP_SIZE)), FP32_DTYPE),
        C_partial: T.Tensor((split_k, M, N), accum_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        split_gemm(A, A_scale, B, B_scale, C_partial)
        combine(C_partial, C)

    return kernel
