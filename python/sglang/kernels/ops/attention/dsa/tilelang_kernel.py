import functools
from functools import lru_cache
from typing import Any, Optional

import tilelang
import tilelang.language as T
import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported, is_hip

tilelang.set_log_level("WARNING")

# Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
# `out_idx` list in place when normalising negative indices to positive ones.
# That breaks any @tilelang.jit factory that compiles two prim_funcs with
# different param counts (e.g. our unified single/dual partial kernel) — the
# second compile sees indices already-converted for the first's len(params)
# and silently builds the wrong adapter, leading to IndexError at call time.
# Patch once on import to copy the list before mutation.
from tilelang.jit.adapter.base import (  # noqa: E402
    BaseKernelAdapter as _BaseKernelAdapter,
)

if not getattr(_BaseKernelAdapter, "_legalize_result_idx_patched", False):
    _orig_legalize = _BaseKernelAdapter._legalize_result_idx

    def _legalize_result_idx_safe(self, result_idx):
        if isinstance(result_idx, list):
            result_idx = list(result_idx)
        return _orig_legalize(self, result_idx)

    _BaseKernelAdapter._legalize_result_idx = _legalize_result_idx_safe
    _BaseKernelAdapter._legalize_result_idx_patched = True

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
# TL_DISABLE_FAST_MATH has deprecated in v0.1.7.post1 tilelang
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_fp8_fnuz = is_fp8_fnuz()

BF16 = "bfloat16"
FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"
FP8_DTYPE = torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
FP32 = "float32"
INT32 = "int32"
UINT8 = "uint8"


@lru_cache(maxsize=8)
def _pick_inner_iter(seq: int, ni: int, cu: int, block_per_cu: int) -> int:
    """
    Pick the largest valid inner_iter (power-of-two divisor of ni) that keeps
    enough work per CU (seq * ni / inner_iter / cu >= block_per_cu), so we avoid
    under-utilization while minimizing the number of partial groups.
    """

    max_it = int(seq * ni / (cu * block_per_cu))
    it = ni
    while it >= 2:
        if it <= max_it and ni % it == 0:
            return it
        it //= 2
    return 1


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd_kernel_v1(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim) or dim % 64 == 0, (
        f"dim={dim} must be a power of 2 or a multiple of 64"
    )
    assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"tail_dim={tail_dim} must be 0 or a power of 2"
    )
    has_tail = tail_dim > 0
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            if has_tail:
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            if has_tail:
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            if has_tail:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                if has_tail:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i,
                            Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                            g_i,
                            D + d_i,
                        ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                if has_tail:
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

    return main


@tilelang.jit(
    out_idx=[-1],
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)  # type: ignore
def sparse_attention_fwd_kernel_v2(
    num_heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    *,
    kv_group: int = 1,
    sm_scale: Optional[float] = None,
    block_I: int = 64,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)
    threads = 384

    batch = T.symbolic("batch")
    qo_len = T.symbolic("seq_len")
    num_pages = T.symbolic("num_pages")

    q_shape = [batch, qo_len, num_heads, dim + tail_dim]
    kv_shape = [batch, num_pages, kv_group, dim + tail_dim]
    o_shape = [batch, qo_len, num_heads, dim]
    indices_shape = [batch, qo_len, kv_group, topk]

    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = num_heads
    padded_H = max(tilelang.math.next_power_of_2(num_heads), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, "NI should be a multiple of 2"
    D = dim
    D_tail = tail_dim
    if num_heads > 64:
        assert num_heads % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = num_heads // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        """
        Q: [b, qo_len, H, D + D_tail] (bfloat16)
        KV: [b, num_pages, kv_group, D + D_tail] (bfloat16)
        Indices: [b, qo_len, kv_group, topk] (int32)
        """

        with T.Kernel(qo_len * REPLICATE_H, batch, 1, threads=threads) as (bx, by, bz):  # type: ignore
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r
            is_kv_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_kv_valid_1 = T.alloc_shared([BI], "bool", scope="shared")

            acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
            alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_local([1], indices_dtype)
            indices_tmp = T.alloc_local([1], indices_dtype)

            bar_q = T.alloc_barrier(arrive_count=384)
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_k_1_free = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

            bar_0_128 = T.alloc_barrier(arrive_count=128)
            bar_1_128 = T.alloc_barrier(arrive_count=128)
            bar_2_128 = T.alloc_barrier(arrive_count=128)
            bar_final = T.alloc_barrier(arrive_count=128)

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else bx // REPLICATE_H

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            T.barrier_arrive(bar_q)

            if tx < 128:
                T.set_max_nreg(240, 1)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                T.fill(acc_o_l, 0)
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    # with sync_at(bar_0_128, 0):
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))
                    T.barrier_arrive(bar_0_128)
                    T.barrier_wait(bar_0_128, 0)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            is_kv_valid_0[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True)
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared_0,
                        acc_s,
                        transpose_B=True,
                    )
                    if i_i != 0:
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(
                        acc_s, sumexp_i, dim=1
                    )  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))
                    T.barrier_arrive(bar_0_128)
                    T.barrier_wait(bar_0_128, 1)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            is_kv_valid_1[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True)
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared_1,
                        acc_s,
                        transpose_B=True,
                    )
                    T.barrier_arrive(bar_sScale_and_sS_free)
                    T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(
                        acc_s, sumexp_i, dim=1
                    )  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_1_free[0])

                # Rescale
                for h_i in T.Parallel(H_per_block):
                    sum_exp_shared[h_i] = sumexp[h_i]
                T.barrier_arrive(bar_final)
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_l[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0 : D // 2])
            elif tx >= 128 and tx < 256:
                # T.set_max_nreg(168, 1)
                T.fill(acc_o_r, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                    T.barrier_arrive(bar_1_128)
                    T.barrier_wait(bar_1_128, 0)
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                    T.barrier_arrive(bar_k_0_free[0])
                    T.barrier_arrive(bar_sScale_and_sS_free)

                    # Buffer 1
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                    T.barrier_arrive(bar_1_128)
                    T.barrier_wait(bar_1_128, 1)
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                    T.barrier_arrive(bar_k_1_free[0])
                    if i_i != T.ceildiv(NI, 2) - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free)

                # Rescale
                T.barrier_wait(bar_final, 0)
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2 : D])
            elif tx >= 256:
                # producer
                T.set_max_nreg(80, 0)
                indices_local[0] = 0
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    T.barrier_arrive(bar_2_128)
                    T.barrier_wait(bar_2_128, 0)

                    for r in T.serial(4):
                        indices_tmp[0] = Indices[
                            b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 256) // 8
                        ]
                        is_kv_valid_0[r * 16 + (tx - 256) // 8] = indices_tmp[0] >= 0
                        if is_kv_valid_0[r * 16 + (tx - 256) // 8]:
                            indices_local[0] = indices_tmp[0]

                        with T.attr("default", "async_scope", 1):  # type: ignore
                            for u in T.serial(4):
                                for v in T.vectorized(8):
                                    KV_shared_0_l[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                                    KV_shared_0_r[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        D // 2 + 64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                        with T.attr("default", "async_scope", 1):  # type: ignore
                            for v in T.vectorized(8):
                                K_tail_shared_0[
                                    r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v
                                ] = KV[
                                    b_i,
                                    indices_local[0],
                                    g_i,
                                    D + (tx - 256) % 8 * 8 + v,
                                ]

                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    T.barrier_arrive(bar_2_128)
                    T.barrier_wait(bar_2_128, 1)

                    for r in T.serial(4):
                        indices_tmp[0] = Indices[
                            b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8
                        ]
                        is_kv_valid_1[r * 16 + (tx - 256) // 8] = indices_tmp[0] >= 0
                        if is_kv_valid_1[r * 16 + (tx - 256) // 8]:
                            indices_local[0] = indices_tmp[0]

                        with T.attr("default", "async_scope", 1):  # type: ignore
                            for u in T.serial(4):
                                for v in T.vectorized(8):
                                    KV_shared_1_l[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                                    KV_shared_1_r[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        D // 2 + 64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                        with T.attr("default", "async_scope", 1):  # type: ignore
                            for v in T.vectorized(8):
                                K_tail_shared_1[
                                    r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v
                                ] = KV[
                                    b_i,
                                    indices_local[0],
                                    g_i,
                                    D + (tx - 256) % 8 * 8 + v,
                                ]

                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

    return main


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_partial(
    heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    inner_iter=1,
    num_stages=1,
    threads=256,
):
    """
    grid: (seq_len * REPLICATE_H, top_k / block_I / inner_iter)
    Each GPU block processes `inner_iter` consecutive KV tiles and writes one (partial_o, partial_lse) entry.
    """

    assert is_causal == True, "non-causal is not supported"
    assert kv_group == 1
    assert topk % block_I == 0
    assert topk % (block_I * inner_iter) == 0, (
        f"topk ({topk}) must be divisible by block_I * inner_iter = "
        f"{block_I} * {inner_iter}"
    )

    # log2(e) = 1.44269504
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = 1
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    REPLICATE_H = (head_kv // 64) if head_kv > 64 else 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64
    N_GROUPS = topk // (block_I * inner_iter)
    BI = block_I
    D = dim
    D_tail = tail_dim
    has_tail = D_tail > 0

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    partial_o_shape = [batch, seq_len, N_GROUPS, heads, dim]
    partial_lse_shape = [batch, seq_len, N_GROUPS, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    _q_in_shared = inner_iter == 1

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Partial_O: T.Tensor(partial_o_shape, dtype),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, N_GROUPS, threads=threads) as (bx, by):
            if _q_in_shared:
                Q_buf = T.alloc_shared([H_per_block, D], dtype)
                if has_tail:
                    Q_tail_buf = T.alloc_shared([H_per_block, D_tail], dtype)
            else:
                Q_buf = T.alloc_fragment([H_per_block, D], dtype)
                if has_tail:
                    Q_tail_buf = T.alloc_fragment([H_per_block, D_tail], dtype)

            KV_shared = T.alloc_shared([BI, D], dtype)
            if has_tail:
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            mask = T.alloc_fragment([BI], T.bool)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = 0, 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            group_i = by
            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_buf)
            if has_tail:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_buf)

            for k_i in T.Pipelined(inner_iter, num_stages=num_stages):
                topk_block_i = group_i * inner_iter + k_i

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i] >= 0
                for bi_i, d_i in T.Parallel(BI, D):
                    idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    KV_shared[bi_i, d_i] = KV[
                        b_i, T.if_then_else(idx >= 0, idx, 0), g_i, d_i
                    ]
                if has_tail:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i, T.if_then_else(idx >= 0, idx, 0), g_i, D + d_i
                        ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                T.gemm(
                    Q_buf,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                if has_tail:
                    T.gemm(
                        Q_tail_buf,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] *= alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # sumexp==0 (all masked), divide by 1 to get 0 and avoid nan
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            # sumexp==0 (all masked), use large negative so combine ignores this split
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )

            T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :])
            T.copy(sumexp, Partial_Lse[b_i, s_i, group_i, H0:H1])

    return main


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_combine(
    heads,
    dim,
    topk,
    head_per_block,
    *,
    block_I=64,
    threads=256,
):
    """
    grid: (seq_len * REPLICATE_H). batch=1, kv_group=1.
    Each block does one tile of heads (e.g. 4 or 8 for decode).
    """

    assert heads % head_per_block == 0, f"head_per_block must divide heads"

    batch = 1
    seq_len = T.dynamic("seq_len")

    NI = topk // block_I
    H_per_block = head_per_block
    REPLICATE_H = heads // H_per_block

    partial_o_shape = [batch, seq_len, NI, heads, dim]
    partial_lse_shape = [batch, seq_len, NI, heads]
    o_shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Partial_O: T.Tensor(partial_o_shape, dtype),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, threads=threads) as (bx,):
            shared_lse = T.alloc_shared([NI, H_per_block], accum_dtype)

            lse_max = T.alloc_fragment([H_per_block], accum_dtype)
            lse_sum = T.alloc_fragment([H_per_block], accum_dtype)
            scale = T.alloc_fragment([H_per_block, NI], accum_dtype)
            acc_o = T.alloc_fragment([H_per_block, dim], accum_dtype)

            b_i = 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * H_per_block
            H1 = H0 + H_per_block

            for k in T.serial(NI):
                T.copy(Partial_Lse[b_i, s_i, k, H0:H1], shared_lse[k, :])

            T.fill(lse_max, -(2**30))
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    lse_max[h_i] = T.max(lse_max[h_i], shared_lse[k, h_i])
            T.fill(lse_sum, 0)
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    lse_sum[h_i] = lse_sum[h_i] + T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i]
                    )
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    scale[h_i, k] = T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i] - T.log2(lse_sum[h_i])
                    )

            T.fill(acc_o, 0)
            for k in T.serial(NI):
                for h_i, d_i in T.Parallel(H_per_block, dim):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] + scale[h_i, k] * Partial_O[
                        b_i, s_i, k, H0 + h_i, d_i
                    ].astype(accum_dtype)

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

    return main


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def sparse_mla_fwd_decode_partial_fp8(
    num_heads: int,
    d_v: int,
    d_tail: int,
    topk: int,
    *,
    sm_scale=None,
    block_I=64,
    inner_iter=1,
    threads=256,
):
    assert d_v in (256, 512), f"only support d_v=256 or d_v=512, got {d_v}"
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )

    # Softmax scores are in [0, 1]. We scale by fp8_max_val before FP8 cast
    # to better utilize FP8 dynamic range, then apply the inverse scale after GEMM.
    # This is numerically safe because softmax output is bounded by 1.
    fp8_dtype = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"
    fp8_max_val = 240.0 if _is_fp8_fnuz else 448.0
    s_inv_scale_const = fp8_max_val
    s_scale_const = 1.0 / fp8_max_val

    BI = block_I
    group_size = 128
    assert d_v % group_size == 0
    assert d_tail in (0, 64), f"only support d_tail=0 or d_tail=64, got {d_tail}"
    num_main_tiles = d_v // group_size
    has_tail = d_tail > 0
    dim_quant_fp8 = d_v + d_tail
    rope_offset_fp8 = d_v
    n_groups = topk // (BI * inner_iter)

    if sm_scale is None:
        sm_scale = (1.0 / (d_v + d_tail)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    h_per_block = 16
    # Match bf16 partial behavior: keep fixed 16-head tiles and use
    # sliced T.copy on H0:H1 for tail handling.
    assert num_heads <= h_per_block or num_heads % h_per_block == 0, (
        "num_heads must be <=16 or divisible by 16"
    )
    head_blocks_per_seq = (num_heads + h_per_block - 1) // h_per_block

    batch = 1
    kv_group = 1
    seq_len = T.symbolic("seq_len")
    num_pages = T.symbolic("num_pages")

    q_fp8_shape = [batch, seq_len, num_heads, d_v + d_tail]
    kv_fp8_shape = [batch, num_pages, kv_group, dim_quant_fp8]
    idx_shape = [batch, seq_len, kv_group, topk]
    partial_o_shape = [batch, seq_len, n_groups, num_heads, d_v]
    partial_lse_shape = [batch, seq_len, n_groups, num_heads]

    accum_dtype = T.float32
    dtype_bf16 = T.bfloat16

    @T.prim_func
    def main(
        q_fp8: T.Tensor(q_fp8_shape, fp8_dtype),
        kv_fp8: T.Tensor(kv_fp8_shape, fp8_dtype),
        indices: T.Tensor(idx_shape, T.int32),
        partial_o: T.Tensor(partial_o_shape, dtype_bf16),
        partial_lse: T.Tensor(partial_lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * head_blocks_per_seq, n_groups, threads=threads) as (
            bx,
            by,
        ):
            b_i, g_i = 0, 0
            s_i = bx // head_blocks_per_seq
            group_i = by
            H0 = (bx % head_blocks_per_seq) * h_per_block
            H1 = H0 + h_per_block

            # Split the main-dimension GEMM into 128-wide tiles.
            # Although this adds extra intermediate memory traffic,
            # it shortens the MFMA accumulation dependency chain and improves performance.
            q_tile0 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile1 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            kv_tile0 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile1 = T.alloc_shared([BI, group_size], fp8_dtype)
            if num_main_tiles == 4:
                q_tile2 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
                q_tile3 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
                kv_tile2 = T.alloc_shared([BI, group_size], fp8_dtype)
                kv_tile3 = T.alloc_shared([BI, group_size], fp8_dtype)
            if has_tail:
                q_tail_buf = T.alloc_shared([h_per_block, d_tail], fp8_dtype)
                k_tail_shared = T.alloc_shared([BI, d_tail], fp8_dtype)
            s_fp8_shared = T.alloc_shared([h_per_block, BI], fp8_dtype)
            page_idx_shared = T.alloc_shared([BI], T.int32)

            mask = T.alloc_fragment([BI], T.bool)
            acc_s = T.alloc_fragment([h_per_block, BI], accum_dtype)
            acc_tile = T.alloc_fragment([h_per_block, BI], accum_dtype)
            sv_tile = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            sumexp = T.alloc_fragment([h_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([h_per_block], accum_dtype)
            alpha = T.alloc_fragment([h_per_block], accum_dtype)
            m_i = T.alloc_fragment([h_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([h_per_block], accum_dtype)
            inv_denom = T.alloc_fragment([h_per_block], accum_dtype)

            acc_o_tile0 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile1 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            if num_main_tiles == 4:
                acc_o_tile2 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
                acc_o_tile3 = T.alloc_fragment([h_per_block, group_size], accum_dtype)

            T.fill(acc_o_tile0, 0)
            T.fill(acc_o_tile1, 0)
            if num_main_tiles == 4:
                T.fill(acc_o_tile2, 0)
                T.fill(acc_o_tile3, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            if has_tail:
                T.copy(q_fp8[b_i, s_i, H0:H1, d_v:], q_tail_buf)
            T.copy(q_fp8[b_i, s_i, H0:H1, 0:group_size], q_tile0)
            T.copy(q_fp8[b_i, s_i, H0:H1, group_size : 2 * group_size], q_tile1)
            if num_main_tiles == 4:
                T.copy(
                    q_fp8[b_i, s_i, H0:H1, 2 * group_size : 3 * group_size],
                    q_tile2,
                )
                T.copy(
                    q_fp8[b_i, s_i, H0:H1, 3 * group_size : 4 * group_size],
                    q_tile3,
                )

            for k_i in T.serial(inner_iter):
                topk_block_i = group_i * inner_iter + k_i

                for bi_i in T.Parallel(BI):
                    idx = indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    valid = idx >= 0
                    page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                    mask[bi_i] = valid

                for bi_i, j in T.Parallel(BI, group_size):
                    page = page_idx_shared[bi_i]
                    kv_tile0[bi_i, j] = kv_fp8[b_i, page, g_i, j]
                    kv_tile1[bi_i, j] = kv_fp8[b_i, page, g_i, group_size + j]
                    if num_main_tiles == 4:
                        kv_tile2[bi_i, j] = kv_fp8[b_i, page, g_i, 2 * group_size + j]
                        kv_tile3[bi_i, j] = kv_fp8[b_i, page, g_i, 3 * group_size + j]

                if has_tail:
                    for bi_i, j in T.Parallel(BI, d_tail):
                        page = page_idx_shared[bi_i]
                        k_tail_shared[bi_i, j] = kv_fp8[
                            b_i, page, g_i, rope_offset_fp8 + j
                        ]

                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                T.gemm(q_tile0, kv_tile0, acc_s, transpose_B=True, clear_accum=False)
                T.gemm(q_tile1, kv_tile1, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                if num_main_tiles == 4:
                    T.gemm(
                        q_tile2,
                        kv_tile2,
                        acc_tile,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                    T.gemm(
                        q_tile3,
                        kv_tile3,
                        acc_tile,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                if has_tail:
                    T.gemm(
                        q_tail_buf,
                        k_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(h_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(h_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile0[h_i, j] *= alpha[h_i]
                    acc_o_tile1[h_i, j] *= alpha[h_i]
                    if num_main_tiles == 4:
                        acc_o_tile2[h_i, j] *= alpha[h_i]
                        acc_o_tile3[h_i, j] *= alpha[h_i]

                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    s_fp8_shared[h_i, bi_i] = T.clamp(
                        acc_s[h_i, bi_i] * s_inv_scale_const,
                        -fp8_max_val,
                        fp8_max_val,
                    )
                T.gemm(s_fp8_shared, kv_tile0, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile0[h_i, j] += sv_tile[h_i, j] * s_scale_const
                T.gemm(s_fp8_shared, kv_tile1, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile1[h_i, j] += sv_tile[h_i, j] * s_scale_const
                if num_main_tiles == 4:
                    T.gemm(s_fp8_shared, kv_tile2, sv_tile, clear_accum=True)
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile2[h_i, j] += sv_tile[h_i, j] * s_scale_const
                    T.gemm(s_fp8_shared, kv_tile3, sv_tile, clear_accum=True)
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile3[h_i, j] += sv_tile[h_i, j] * s_scale_const

            for h_i in T.Parallel(h_per_block):
                denom = T.if_then_else(sumexp[h_i] == 0.0, 1.0, sumexp[h_i])
                inv_denom[h_i] = 1.0 / denom
            for h_i, j in T.Parallel(h_per_block, group_size):
                acc_o_tile0[h_i, j] *= inv_denom[h_i]
                acc_o_tile1[h_i, j] *= inv_denom[h_i]
                if num_main_tiles == 4:
                    acc_o_tile2[h_i, j] *= inv_denom[h_i]
                    acc_o_tile3[h_i, j] *= inv_denom[h_i]

            for h_i in T.Parallel(h_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )

            T.copy(acc_o_tile0, partial_o[b_i, s_i, group_i, H0:H1, 0:group_size])
            T.copy(
                acc_o_tile1,
                partial_o[b_i, s_i, group_i, H0:H1, group_size : 2 * group_size],
            )
            if num_main_tiles == 4:
                T.copy(
                    acc_o_tile2,
                    partial_o[
                        b_i, s_i, group_i, H0:H1, 2 * group_size : 3 * group_size
                    ],
                )
                T.copy(
                    acc_o_tile3,
                    partial_o[
                        b_i, s_i, group_i, H0:H1, 3 * group_size : 4 * group_size
                    ],
                )

            T.copy(sumexp, partial_lse[b_i, s_i, group_i, H0:H1])

    return main


def tilelang_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3
    num_heads = q.shape[1]
    dim = q.shape[2]
    tail_dim = dim - d_v
    topk = indices.shape[-1]
    assert topk % 64 == 0, "topk must be padded to a multiple of 64"

    is_fp8_kv = kv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if _is_hip or is_fp8_kv:
        if is_fp8_kv:
            if q.dtype != kv.dtype:
                q = q.to(kv.dtype)
            if not _is_hip:
                block_I, threads, block_per_cu = 32, 128, 1
                cu = torch.cuda.get_device_properties(q.device).multi_processor_count
            elif _is_gfx95_supported:
                block_I, threads, block_per_cu, cu = 64, 256, 2, 256
            else:
                block_I, threads, block_per_cu, cu = 64, 256, 1, 304
            ni = topk // block_I
            inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
            kernel_partial = sparse_mla_fwd_decode_partial_fp8(
                num_heads,
                d_v,
                tail_dim,
                topk,
                sm_scale=sm_scale,
                block_I=block_I,
                inner_iter=inner_iter,
                threads=threads,
            )
        else:
            if _is_gfx95_supported:
                block_I, threads, block_per_cu, cu = 64, 256, 2, 256
            else:
                block_I, threads, block_per_cu, cu = 32, 128, 1, 304
            ni = topk // block_I
            inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
            kernel_partial = sparse_mla_fwd_decode_partial(
                num_heads,
                d_v,
                tail_dim,
                topk,
                sm_scale=sm_scale,
                block_I=block_I,
                inner_iter=inner_iter,
                threads=threads,
            )
        partial_o_batched, partial_lse_batched = kernel_partial(
            q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
        )
        n_groups = ni // inner_iter
        kernel_combine = sparse_mla_fwd_decode_combine(
            num_heads,
            d_v,
            n_groups * block_I,
            head_per_block=4,
            block_I=block_I,
            threads=threads,
        )
        out = kernel_combine(partial_o_batched, partial_lse_batched)
    else:
        kernel_factory = (
            sparse_attention_fwd_kernel_v1
            if tail_dim == 0
            else sparse_attention_fwd_kernel_v2
        )
        kernel = kernel_factory(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale)
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
    return out


@functools.cache
def fp8_paged_mqa_logits_kernel(
    head_dim: int = 128,
    num_heads: int = 64,
    block_size: int = 64,
    clear_accum: bool = True,
    split_kv: int = 1,
) -> Any:
    N = T.symbolic("batch_size")
    L = T.symbolic("max_table_length")
    S = T.symbolic("max_seq_len")
    C = T.symbolic("num_blocks")
    B = block_size
    D = head_dim
    H = num_heads
    SK = int(split_kv)
    BLOCK_BYTES = B * (D + 4)
    SCALE_OFFSET = B * D

    assert D % 4 == 0
    assert H % 4 == 0
    assert D == 128
    assert SK >= 1

    @tilelang.jit(
        pass_configs={
            **pass_configs,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        }
    )
    def fp8_paged_mqa_logits(
        q: T.Tensor[(N, H, D), FP8],
        kvcache_u8: T.Tensor[(C, BLOCK_BYTES), UINT8],
        weight: T.Tensor[(N, H), FP32],
        seq_lens: T.Tensor[(N,), INT32],
        page_table: T.Tensor[(N, L), INT32],
        o: T.Tensor[(N, S), FP32],
    ) -> None:
        _ = N, L, S, C, D, H, B
        with T.Kernel(N * SK) as bxs:
            bx = bxs % N
            pid_split = bxs // N
            seq_len = seq_lens[bx]
            np_total = T.ceildiv(seq_len, B)
            stride = T.ceildiv(np_total, SK)
            i_start = pid_split * stride
            n_iters = T.max(0, T.min(stride, np_total - i_start))

            q_smem = T.alloc_shared((H, D), FP8)
            q_s_frag = T.alloc_fragment((H,), FP32)
            T.copy(q[bx, 0, 0], q_smem)
            T.copy(weight[bx, 0], q_s_frag)

            for j in T.Pipelined(n_iters, num_stages=2):
                i = i_start + j
                page = page_table[bx, i]
                k_smem_u8 = T.alloc_shared((1, B * D), UINT8)
                T.copy(kvcache_u8[page : page + 1, 0:SCALE_OFFSET], k_smem_u8)
                k_smem = T.view(k_smem_u8, (B, D), FP8)
                k_s_smem_u8 = T.alloc_shared((1, B * 4), UINT8)
                T.copy(
                    kvcache_u8[page : page + 1, SCALE_OFFSET:BLOCK_BYTES],
                    k_s_smem_u8,
                )
                k_s_smem = T.view(k_s_smem_u8, (B,), FP32)
                k_s_frag = T.alloc_fragment((B,), FP32)
                T.copy(k_s_smem, k_s_frag)

                logits = T.alloc_fragment((B, H), FP32)
                if not clear_accum:
                    T.fill(logits, 0.0)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )

                # post processing
                for h, j2 in T.Parallel(H, B):
                    logits[j2, h] = T.max(logits[j2, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                for j2 in T.Parallel(B):
                    logits_sum[j2] *= k_s_frag[j2]
                T.copy(logits_sum, o[bx, i * B])

    return fp8_paged_mqa_logits


def tilelang_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)

    NUM_CU = 256
    split_kv = split_kv = max(1, min(max_seq_len // block_size, NUM_CU // batch_size))
    kernel = fp8_paged_mqa_logits_kernel(
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=block_size,
        clear_accum=clean_logits,
        split_kv=split_kv,
    )
    q_fp8 = q_fp8.view(batch_size, num_heads, head_dim)
    kvcache_u8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    kernel(q_fp8, kvcache_u8, weight, seq_lens, page_table, logits)
    return logits
