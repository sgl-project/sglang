"""TileLang kernels for FP8 W8A8 blockwise GEMM."""

import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def fp8_blockwise_gemm_kernel(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    M = T.symbolic("M")
    group_size = 128

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float8_e4m3"),
        A_scale: T.Tensor((M, T.ceildiv(K, group_size)), "float32"),
        B: T.Tensor((N, K), "float8_e4m3"),
        B_scale: T.Tensor(
            (T.ceildiv(N, group_size), T.ceildiv(K, group_size)), "float32"
        ),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (pid_n, pid_m):
            A_shared = T.alloc_shared((block_M, block_K), "float8_e4m3")
            B_shared = T.alloc_shared((block_N, block_K), "float8_e4m3")
            C_shared = T.alloc_shared((block_M, block_N), "bfloat16")
            C_scale = T.alloc_shared((block_M,), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_local_accum = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            T.clear(C_local_accum)

            for k_iter in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[pid_m * block_M, k_iter * block_K], A_shared)
                T.copy(B[pid_n * block_N, k_iter * block_K], B_shared)

                b_scale = B_scale[pid_n * block_N // group_size, k_iter]
                for i in T.Parallel(block_M):
                    C_scale[i] = A_scale[pid_m * block_M + i, k_iter] * b_scale

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * C_scale[i]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])

    return kernel
