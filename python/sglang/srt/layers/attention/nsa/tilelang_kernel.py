from functools import lru_cache
from typing import Any, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
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
FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3"
FP8_DTYPE = torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
FP32 = "float32"
INT32 = "int32"


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


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


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    M = T.symbolic("M")
    fp8_min = -224.0 if _is_fp8_fnuz else -448.0
    fp8_max = 224.0 if _is_fp8_fnuz else 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    if _is_fp8_fnuz:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    else:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int, clear_accum=True):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                if not clear_accum:
                    T.fill(logits, 0)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    if _is_hip:
        return fp8_index_kernel(q.shape[2], q.shape[3], False)(q, q_s, k, k_s)
    else:
        return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)


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
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
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
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
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
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
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
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
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
                    T.gemm(
                        Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared_0,
                        acc_s,
                        transpose_B=True,
                        wg_wait=-1,
                    )

                    T.wait_wgmma(0)

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
                    T.gemm(
                        Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared_1,
                        acc_s,
                        transpose_B=True,
                        wg_wait=-1,
                    )

                    T.wait_wgmma(0)

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
                Q_tail_buf = T.alloc_shared([H_per_block, D_tail], dtype)
            else:
                Q_buf = T.alloc_fragment([H_per_block, D], dtype)
                Q_tail_buf = T.alloc_fragment([H_per_block, D_tail], dtype)

            KV_shared = T.alloc_shared([BI, D], dtype)
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
    assert d_v == 512, f"only support d_v=512"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"

    # Softmax scores are in [0, 1]. We scale by fp8_max_val before FP8 cast
    # to better utilize FP8 dynamic range, then apply the inverse scale after GEMM.
    # This is numerically safe because softmax output is bounded by 1.
    fp8_dtype = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"
    fp8_max_val = 240.0 if _is_fp8_fnuz else 448.0
    s_inv_scale_const = fp8_max_val
    s_scale_const = 1.0 / fp8_max_val

    BI = block_I
    group_size = 128
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
    assert (
        num_heads <= h_per_block or num_heads % h_per_block == 0
    ), "num_heads must be <=16 or divisible by 16"
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

            # We intentionally split the K=512 GEMM into 4x128 tiles.
            # Although this adds extra intermediate memory traffic,
            # it shortens the MFMA accumulation dependency chain and improves performance.
            q_tile0 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile1 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile2 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile3 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            kv_tile0 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile1 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile2 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile3 = T.alloc_shared([BI, group_size], fp8_dtype)
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
            acc_o_tile2 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile3 = T.alloc_fragment([h_per_block, group_size], accum_dtype)

            T.fill(acc_o_tile0, 0)
            T.fill(acc_o_tile1, 0)
            T.fill(acc_o_tile2, 0)
            T.fill(acc_o_tile3, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            T.copy(q_fp8[b_i, s_i, H0:H1, d_v:], q_tail_buf)
            T.copy(q_fp8[b_i, s_i, H0:H1, 0 * group_size : 1 * group_size], q_tile0)
            T.copy(q_fp8[b_i, s_i, H0:H1, 1 * group_size : 2 * group_size], q_tile1)
            T.copy(q_fp8[b_i, s_i, H0:H1, 2 * group_size : 3 * group_size], q_tile2)
            T.copy(q_fp8[b_i, s_i, H0:H1, 3 * group_size : 4 * group_size], q_tile3)

            for k_i in T.serial(inner_iter):
                topk_block_i = group_i * inner_iter + k_i

                for bi_i in T.Parallel(BI):
                    idx = indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    valid = idx >= 0
                    page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                    mask[bi_i] = valid

                for bi_i, j in T.Parallel(BI, group_size):
                    page = page_idx_shared[bi_i]
                    kv_tile0[bi_i, j] = kv_fp8[b_i, page, g_i, 0 * group_size + j]
                    kv_tile1[bi_i, j] = kv_fp8[b_i, page, g_i, 1 * group_size + j]
                    kv_tile2[bi_i, j] = kv_fp8[b_i, page, g_i, 2 * group_size + j]
                    kv_tile3[bi_i, j] = kv_fp8[b_i, page, g_i, 3 * group_size + j]

                for bi_i, j in T.Parallel(BI, d_tail):
                    page = page_idx_shared[bi_i]
                    k_tail_shared[bi_i, j] = kv_fp8[b_i, page, g_i, rope_offset_fp8 + j]

                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                T.gemm(q_tile0, kv_tile0, acc_s, transpose_B=True, clear_accum=False)
                T.gemm(q_tile1, kv_tile1, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                T.gemm(q_tile2, kv_tile2, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                T.gemm(q_tile3, kv_tile3, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
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
                    acc_o_tile0[h_i, j] = acc_o_tile0[h_i, j] * alpha[h_i]
                    acc_o_tile1[h_i, j] = acc_o_tile1[h_i, j] * alpha[h_i]
                    acc_o_tile2[h_i, j] = acc_o_tile2[h_i, j] * alpha[h_i]
                    acc_o_tile3[h_i, j] = acc_o_tile3[h_i, j] * alpha[h_i]

                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    s_fp8_shared[h_i, bi_i] = T.clamp(
                        acc_s[h_i, bi_i] * s_inv_scale_const,
                        -fp8_max_val,
                        fp8_max_val,
                    )
                T.gemm(s_fp8_shared, kv_tile0, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile0[h_i, j] = (
                        acc_o_tile0[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

                T.gemm(s_fp8_shared, kv_tile1, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile1[h_i, j] = (
                        acc_o_tile1[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

                T.gemm(s_fp8_shared, kv_tile2, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile2[h_i, j] = (
                        acc_o_tile2[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

                T.gemm(s_fp8_shared, kv_tile3, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile3[h_i, j] = (
                        acc_o_tile3[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

            for h_i in T.Parallel(h_per_block):
                denom = T.if_then_else(sumexp[h_i] == 0.0, 1.0, sumexp[h_i])
                inv_denom[h_i] = 1.0 / denom
            for h_i, j in T.Parallel(h_per_block, group_size):
                acc_o_tile0[h_i, j] = acc_o_tile0[h_i, j] * inv_denom[h_i]
                acc_o_tile1[h_i, j] = acc_o_tile1[h_i, j] * inv_denom[h_i]
                acc_o_tile2[h_i, j] = acc_o_tile2[h_i, j] * inv_denom[h_i]
                acc_o_tile3[h_i, j] = acc_o_tile3[h_i, j] * inv_denom[h_i]

            for h_i in T.Parallel(h_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )

            T.copy(
                acc_o_tile0,
                partial_o[b_i, s_i, group_i, H0:H1, 0 * group_size : 1 * group_size],
            )
            T.copy(
                acc_o_tile1,
                partial_o[b_i, s_i, group_i, H0:H1, 1 * group_size : 2 * group_size],
            )
            T.copy(
                acc_o_tile2,
                partial_o[b_i, s_i, group_i, H0:H1, 2 * group_size : 3 * group_size],
            )
            T.copy(
                acc_o_tile3,
                partial_o[b_i, s_i, group_i, H0:H1, 3 * group_size : 4 * group_size],
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
    assert topk == 2048

    if _is_hip:
        is_fp8_kv = kv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if is_fp8_kv:
            if q.dtype != kv.dtype:
                q = q.to(kv.dtype)
            if _is_gfx95_supported:
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
        kernel = sparse_attention_fwd_kernel_v2(
            num_heads, d_v, tail_dim, topk, sm_scale=sm_scale
        )
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
    return out


def _build_fp8_combined_view(k_cache: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Reinterpret a MODEL1_FP8Sparse KV cache as a contiguous uint32 view.
    Input:  k_cache (num_blocks, block_size, 1, d_qk)  fp8/uint8
            — per-block storage also holds scales + padding past d_qk.
    Output: (num_blocks, block_pad_u32) uint32 covering the full block
            stride. Same storage ashe input, no copy.
    """
    k_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    num_blocks = k_u8.shape[0]
    block_size = k_u8.shape[1]
    block_pad_u32 = k_u8.stride(0) // 4
    storage = k_u8.untyped_storage()
    flat_u32 = torch.empty(0, dtype=torch.uint32, device=k_u8.device).set_(
        storage, 0, (storage.nbytes() // 4,), (1,)
    )
    k_combined = torch.as_strided(
        flat_u32,
        size=(num_blocks, block_pad_u32),
        stride=(block_pad_u32, 1),
        storage_offset=k_u8.storage_offset() // 4,
    )
    return k_combined, num_blocks, block_size


_TOPK_LEN_SENTINEL_CACHE: dict = {}
_INT32_MAX = 2**31 - 1


def _topk_length_sentinel(device: torch.device, batch: int) -> torch.Tensor:
    """Cached `(batch,) int32 INT_MAX` tensor used when `topk_length` is None."""
    cur = _TOPK_LEN_SENTINEL_CACHE.get(device)
    if cur is None or cur.numel() < batch:
        cur = torch.full(
            (max(batch, 256),), _INT32_MAX, dtype=torch.int32, device=device
        )
        _TOPK_LEN_SENTINEL_CACHE[device] = cur
    return cur[:batch]


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def dpsk_v4_fp8_partial_kernel(
    num_heads: int,
    topk1: int,
    block_size_kv_1: int,
    topk2: int = 0,
    block_size_kv_2: int = 0,
    *,
    dim: int = 448,
    tail_dim: int = 64,
    sm_scale: float = 0.0,
    block_I: int = 64,
    inner_iter: int = 1,
    inner_iter_2: int = 0,
    num_stages: int = 0,
    threads: int = 512,
) -> Any:
    """
    Read FP8 K cache directly, dequantise to BF16 in-kernel, do flash-attn
    online softmax with split-K. Supports a second cache (`topk2>0`) and
    `attn_sink` is folded later by the combine kernel.
    """
    log2e: float = 1.44269504
    if sm_scale <= 0.0:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * log2e
    else:
        sm_scale = sm_scale * log2e
    assert dim == 448 and tail_dim == 64
    assert topk1 % block_I == 0
    assert (
        topk1 // block_I
    ) % inner_iter == 0, (
        f"NI_1={topk1 // block_I} must be divisible by inner_iter={inner_iter}"
    )
    assert block_size_kv_1 > 0 and (block_size_kv_1 & (block_size_kv_1 - 1)) == 0

    is_dual = topk2 > 0
    if is_dual:
        assert inner_iter_2 > 0, "dual-cache call requires inner_iter_2 > 0"
        assert topk2 % block_I == 0
        assert (
            topk2 // block_I
        ) % inner_iter_2 == 0, (
            f"NI_2={topk2 // block_I} must be divisible by inner_iter_2={inner_iter_2}"
        )
        assert block_size_kv_2 > 0 and (block_size_kv_2 & (block_size_kv_2 - 1)) == 0

    PACKED_W = dim + 2 * tail_dim
    NOPE_TILE = 64
    NUM_TILES = dim // NOPE_TILE
    SCALE_W = 8
    PACKED_W4 = PACKED_W // 4
    SCALE_W4 = SCALE_W // 4

    kv_group = 1
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    num_blocks_kv_1 = T.symbolic("num_blocks_kv_1")
    block_pad_u32_1 = T.symbolic("block_pad_u32_1")
    if is_dual:
        num_blocks_kv_2 = T.symbolic("num_blocks_kv_2")
        block_pad_u32_2 = T.symbolic("block_pad_u32_2")

    head_kv = num_heads // kv_group
    D = dim
    D_tail = tail_dim
    BI = block_I
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if head_kv > 64:
        assert head_kv % 64 == 0
    REPLICATE_H = (head_kv + 63) // 64 if head_kv > 64 else 1
    H_per_block = 64 if REPLICATE_H > 1 else padded_H

    NI_1 = topk1 // BI
    n_groups_1 = NI_1 // inner_iter
    NI_2 = (topk2 // BI) if is_dual else 0
    n_groups_2 = (NI_2 // inner_iter_2) if is_dual else 0
    n_groups = n_groups_1 + n_groups_2

    BS_KV_1 = block_size_kv_1
    NOPE_ROPE_U32_PER_BLOCK_1 = BS_KV_1 * PACKED_W4
    if is_dual:
        BS_KV_2 = block_size_kv_2
        NOPE_ROPE_U32_PER_BLOCK_2 = BS_KV_2 * PACKED_W4

    q_shape = [batch, seq_len, num_heads, D + D_tail]
    k1_shape = [num_blocks_kv_1, block_pad_u32_1]
    indices1_shape = [batch, seq_len, topk1]
    topk_length_shape = [batch]
    partial_o_shape = [batch, seq_len, n_groups, num_heads, D + D_tail]
    partial_lse_shape = [batch, seq_len, n_groups, num_heads]
    if is_dual:
        k2_shape = [num_blocks_kv_2, block_pad_u32_2]
        indices2_shape = [batch, seq_len, topk2]

    accum_dtype = "float"
    indices_dtype = INT32

    if is_dual:

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, BF16),  # type: ignore
            K_combined_1: T.Tensor(k1_shape, "uint32"),  # type: ignore
            Indices_1: T.Tensor(indices1_shape, indices_dtype),  # type: ignore
            Topk_length_1: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
            K_combined_2: T.Tensor(k2_shape, "uint32"),  # type: ignore
            Indices_2: T.Tensor(indices2_shape, indices_dtype),  # type: ignore
            Topk_length_2: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
            Partial_O: T.Tensor(partial_o_shape, BF16),  # type: ignore
            Partial_LSE: T.Tensor(partial_lse_shape, accum_dtype),  # type: ignore
        ) -> None:
            """
            grid: (seq_len * REPLICATE_H * n_groups, batch, 1)
            Each block processes `inner_iter` (or `inner_iter_2`) consecutive
            KV tiles of one phase and writes one (partial_o, partial_lse) entry.
            """
            with T.Kernel(
                seq_len * REPLICATE_H * n_groups, batch, kv_group, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_fragment([H_per_block, D], BF16)
                Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], BF16)
                K_packed_shared = T.alloc_shared([BI, PACKED_W4], "uint32")
                K_scale_shared = T.alloc_shared([BI, SCALE_W4], "uint32")
                KV_shared = T.alloc_shared([BI, D], BF16)
                K_tail_shared = T.alloc_shared([BI, D_tail], BF16)
                S_shared = T.alloc_shared([H_per_block, BI], BF16)
                page_idx_shared = T.alloc_shared([BI], INT32)

                mask = T.alloc_fragment([BI], "bool")
                scale_byte_local = T.alloc_fragment([BI, NUM_TILES], "uint32")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_o_tail = T.alloc_fragment([H_per_block, D_tail], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(acc_o_tail, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_i, g_i = by, bz
                # bx encodes (s_i, h_replicate, group_i).
                spans_per_seq = REPLICATE_H * n_groups
                s_i = bx // spans_per_seq
                rest = bx % spans_per_seq
                group_i = rest // REPLICATE_H
                h_rep = rest % REPLICATE_H
                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else h_rep * 64)
                H1 = H0 + H_per_block

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                T.copy(Q[b_i, s_i, H0:H1, D : D + D_tail], Q_tail_shared)

                tk_len_1 = Topk_length_1[b_i]
                tk_len_2 = Topk_length_2[b_i]

                # Phase 1 (groups [0, n_groups_1)).
                if group_i < n_groups_1:
                    for k_i in T.Pipelined(inner_iter, num_stages=num_stages):
                        iter_i = group_i * inner_iter + k_i
                        for bi_i in T.Parallel(BI):
                            pos = iter_i * BI + bi_i
                            idx = Indices_1[b_i, s_i, pos]
                            valid = (idx >= 0) & (pos < tk_len_1)
                            page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                            mask[bi_i] = valid

                        for bi_i, w_i in T.Parallel(BI, PACKED_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_1
                            t_in_block = page % BS_KV_1
                            K_packed_shared[bi_i, w_i] = K_combined_1[
                                block_id, t_in_block * PACKED_W4 + w_i
                            ]

                        for bi_i, w_i in T.Parallel(BI, SCALE_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_1
                            t_in_block = page % BS_KV_1
                            K_scale_shared[bi_i, w_i] = K_combined_1[
                                block_id,
                                NOPE_ROPE_U32_PER_BLOCK_1 + t_in_block * SCALE_W4 + w_i,
                            ]

                        for bi_i, ti in T.Parallel(BI, NUM_TILES):
                            word_idx = ti // 4
                            byte_in_word = ti % 4
                            word = K_scale_shared[bi_i, word_idx]
                            scale_byte_local[bi_i, ti] = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)

                        for bi_i, d_i in T.Parallel(BI, D):
                            word_idx = d_i // 4
                            byte_in_word = d_i % 4
                            word = K_packed_shared[bi_i, word_idx]
                            b_u32 = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)
                            sign_bf = (b_u32 & T.uint32(0x80)) * T.uint32(0x100)
                            exp_e4 = (b_u32 & T.uint32(0x78)) >> T.uint32(3)
                            mant_bf = (b_u32 & T.uint32(0x7)) * T.uint32(0x10)
                            scale_byte = scale_byte_local[bi_i, d_i // NOPE_TILE]
                            exp_combined = exp_e4 + scale_byte - T.uint32(7)
                            bf16_bits = (
                                sign_bf | (exp_combined << T.uint32(7)) | mant_bf
                            )
                            KV_shared[bi_i, d_i] = T.reinterpret(
                                BF16, T.Cast("uint16", bf16_bits)
                            )

                        for bi_i, j in T.Parallel(BI, D_tail):
                            abs_off = D + 2 * j
                            word_idx = abs_off // 4
                            word_off = abs_off % 4
                            word = K_packed_shared[bi_i, word_idx]
                            half_u32 = T.if_then_else(
                                word_off == 0,
                                word & T.uint32(0xFFFF),
                                (word >> T.uint32(16)) & T.uint32(0xFFFF),
                            )
                            K_tail_shared[bi_i, j] = T.reinterpret(
                                BF16, T.Cast("uint16", half_u32)
                            )

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
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
                        for h_i, d_i in T.Parallel(H_per_block, D_tail):
                            acc_o_tail[h_i, d_i] *= alpha[h_i]
                        T.copy(acc_s, S_shared)
                        T.gemm(
                            S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                        )
                        T.gemm(
                            S_shared,
                            K_tail_shared,
                            acc_o_tail,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                else:
                    # Phase 2 (groups [n_groups_1, n_groups)).
                    for k_i in T.Pipelined(inner_iter_2, num_stages=num_stages):
                        iter_i = (group_i - n_groups_1) * inner_iter_2 + k_i
                        for bi_i in T.Parallel(BI):
                            pos = iter_i * BI + bi_i
                            idx = Indices_2[b_i, s_i, pos]
                            valid = (idx >= 0) & (pos < tk_len_2)
                            page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                            mask[bi_i] = valid

                        for bi_i, w_i in T.Parallel(BI, PACKED_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_2
                            t_in_block = page % BS_KV_2
                            K_packed_shared[bi_i, w_i] = K_combined_2[
                                block_id, t_in_block * PACKED_W4 + w_i
                            ]

                        for bi_i, w_i in T.Parallel(BI, SCALE_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_2
                            t_in_block = page % BS_KV_2
                            K_scale_shared[bi_i, w_i] = K_combined_2[
                                block_id,
                                NOPE_ROPE_U32_PER_BLOCK_2 + t_in_block * SCALE_W4 + w_i,
                            ]

                        for bi_i, ti in T.Parallel(BI, NUM_TILES):
                            word_idx = ti // 4
                            byte_in_word = ti % 4
                            word = K_scale_shared[bi_i, word_idx]
                            scale_byte_local[bi_i, ti] = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)

                        for bi_i, d_i in T.Parallel(BI, D):
                            word_idx = d_i // 4
                            byte_in_word = d_i % 4
                            word = K_packed_shared[bi_i, word_idx]
                            b_u32 = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)
                            sign_bf = (b_u32 & T.uint32(0x80)) * T.uint32(0x100)
                            exp_e4 = (b_u32 & T.uint32(0x78)) >> T.uint32(3)
                            mant_bf = (b_u32 & T.uint32(0x7)) * T.uint32(0x10)
                            scale_byte = scale_byte_local[bi_i, d_i // NOPE_TILE]
                            exp_combined = exp_e4 + scale_byte - T.uint32(7)
                            bf16_bits = (
                                sign_bf | (exp_combined << T.uint32(7)) | mant_bf
                            )
                            KV_shared[bi_i, d_i] = T.reinterpret(
                                BF16, T.Cast("uint16", bf16_bits)
                            )

                        for bi_i, j in T.Parallel(BI, D_tail):
                            abs_off = D + 2 * j
                            word_idx = abs_off // 4
                            word_off = abs_off % 4
                            word = K_packed_shared[bi_i, word_idx]
                            half_u32 = T.if_then_else(
                                word_off == 0,
                                word & T.uint32(0xFFFF),
                                (word >> T.uint32(16)) & T.uint32(0xFFFF),
                            )
                            K_tail_shared[bi_i, j] = T.reinterpret(
                                BF16, T.Cast("uint16", half_u32)
                            )

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
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
                        for h_i, d_i in T.Parallel(H_per_block, D_tail):
                            acc_o_tail[h_i, d_i] *= alpha[h_i]
                        T.copy(acc_s, S_shared)
                        T.gemm(
                            S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                        )
                        T.gemm(
                            S_shared,
                            K_tail_shared,
                            acc_o_tail,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                # Pre-normalise + write log2-LSE (combine handles attn_sink).
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                        sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                    )
                for h_i, d_i in T.Parallel(H_per_block, D_tail):
                    acc_o_tail[h_i, d_i] = acc_o_tail[h_i, d_i] / T.if_then_else(
                        sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                    )
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.if_then_else(
                        sumexp[h_i] == 0.0,
                        -(2.0**30),
                        T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                    )
                T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :D])
                T.copy(acc_o_tail, Partial_O[b_i, s_i, group_i, H0:H1, D : D + D_tail])
                T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])

        return main

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, BF16),  # type: ignore
        K_combined_1: T.Tensor(k1_shape, "uint32"),  # type: ignore
        Indices_1: T.Tensor(indices1_shape, indices_dtype),  # type: ignore
        Topk_length_1: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
        Partial_O: T.Tensor(partial_o_shape, BF16),  # type: ignore
        Partial_LSE: T.Tensor(partial_lse_shape, accum_dtype),  # type: ignore
    ) -> None:
        """
        grid: (seq_len * REPLICATE_H * n_groups, batch, 1)
        Each block processes `inner_iter` consecutive KV tiles and writes
        one (partial_o, partial_lse) entry.
        """
        with T.Kernel(
            seq_len * REPLICATE_H * n_groups, batch, kv_group, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_fragment([H_per_block, D], BF16)
            Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], BF16)
            K_packed_shared = T.alloc_shared([BI, PACKED_W4], "uint32")
            K_scale_shared = T.alloc_shared([BI, SCALE_W4], "uint32")
            KV_shared = T.alloc_shared([BI, D], BF16)
            K_tail_shared = T.alloc_shared([BI, D_tail], BF16)
            S_shared = T.alloc_shared([H_per_block, BI], BF16)
            page_idx_shared = T.alloc_shared([BI], INT32)

            mask = T.alloc_fragment([BI], "bool")
            scale_byte_local = T.alloc_fragment([BI, NUM_TILES], "uint32")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_o_tail = T.alloc_fragment([H_per_block, D_tail], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(acc_o_tail, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            spans_per_seq = REPLICATE_H * n_groups
            s_i = bx // spans_per_seq
            rest = bx % spans_per_seq
            group_i = rest // REPLICATE_H
            h_rep = rest % REPLICATE_H
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else h_rep * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D : D + D_tail], Q_tail_shared)

            tk_len_1 = Topk_length_1[b_i]

            for k_i in T.Pipelined(inner_iter, num_stages=num_stages):
                iter_i = group_i * inner_iter + k_i
                for bi_i in T.Parallel(BI):
                    pos = iter_i * BI + bi_i
                    idx = Indices_1[b_i, s_i, pos]
                    valid = (idx >= 0) & (pos < tk_len_1)
                    page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                    mask[bi_i] = valid

                for bi_i, w_i in T.Parallel(BI, PACKED_W4):
                    page = page_idx_shared[bi_i]
                    block_id = page // BS_KV_1
                    t_in_block = page % BS_KV_1
                    K_packed_shared[bi_i, w_i] = K_combined_1[
                        block_id, t_in_block * PACKED_W4 + w_i
                    ]

                for bi_i, w_i in T.Parallel(BI, SCALE_W4):
                    page = page_idx_shared[bi_i]
                    block_id = page // BS_KV_1
                    t_in_block = page % BS_KV_1
                    K_scale_shared[bi_i, w_i] = K_combined_1[
                        block_id,
                        NOPE_ROPE_U32_PER_BLOCK_1 + t_in_block * SCALE_W4 + w_i,
                    ]

                for bi_i, ti in T.Parallel(BI, NUM_TILES):
                    word_idx = ti // 4
                    byte_in_word = ti % 4
                    word = K_scale_shared[bi_i, word_idx]
                    scale_byte_local[bi_i, ti] = (
                        word >> T.Cast("uint32", byte_in_word * 8)
                    ) & T.uint32(0xFF)

                for bi_i, d_i in T.Parallel(BI, D):
                    word_idx = d_i // 4
                    byte_in_word = d_i % 4
                    word = K_packed_shared[bi_i, word_idx]
                    b_u32 = (word >> T.Cast("uint32", byte_in_word * 8)) & T.uint32(
                        0xFF
                    )
                    sign_bf = (b_u32 & T.uint32(0x80)) * T.uint32(0x100)
                    exp_e4 = (b_u32 & T.uint32(0x78)) >> T.uint32(3)
                    mant_bf = (b_u32 & T.uint32(0x7)) * T.uint32(0x10)
                    scale_byte = scale_byte_local[bi_i, d_i // NOPE_TILE]
                    exp_combined = exp_e4 + scale_byte - T.uint32(7)
                    bf16_bits = sign_bf | (exp_combined << T.uint32(7)) | mant_bf
                    KV_shared[bi_i, d_i] = T.reinterpret(
                        BF16, T.Cast("uint16", bf16_bits)
                    )

                for bi_i, j in T.Parallel(BI, D_tail):
                    abs_off = D + 2 * j
                    word_idx = abs_off // 4
                    word_off = abs_off % 4
                    word = K_packed_shared[bi_i, word_idx]
                    half_u32 = T.if_then_else(
                        word_off == 0,
                        word & T.uint32(0xFFFF),
                        (word >> T.uint32(16)) & T.uint32(0xFFFF),
                    )
                    K_tail_shared[bi_i, j] = T.reinterpret(
                        BF16, T.Cast("uint16", half_u32)
                    )

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
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
                for h_i, d_i in T.Parallel(H_per_block, D_tail):
                    acc_o_tail[h_i, d_i] *= alpha[h_i]
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(
                    S_shared, K_tail_shared, acc_o_tail, policy=T.GemmWarpPolicy.FullRow
                )

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            for h_i, d_i in T.Parallel(H_per_block, D_tail):
                acc_o_tail[h_i, d_i] = acc_o_tail[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2.0**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )
            T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :D])
            T.copy(acc_o_tail, Partial_O[b_i, s_i, group_i, H0:H1, D : D + D_tail])
            T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])

    return main


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def dpsk_v4_combine_kernel(
    num_heads: int,
    n_groups: int,
    *,
    dim: int = 448,
    tail_dim: int = 64,
    head_per_block: int = 16,
    threads: int = 256,
    use_attn_sink: bool = False,
) -> Any:
    """
    Combine `n_groups` flash-attention partials into the final output.

    Inputs:
      Partial_O   : (batch, seq_len, n_groups, num_heads, dim+tail_dim) bf16
                    already pre-normalised (acc_o / partial_sumexp)
      Partial_LSE : (batch, seq_len, n_groups, num_heads) fp32, log2 form
      Attn_sink   : (num_heads,) fp32

    Outputs:
      Output : (batch, seq_len, num_heads, dim+tail_dim) bf16
      LSE    : (batch, seq_len, num_heads) fp32, natural log

    Each grid block handles `head_per_block` heads of one (batch, seq) row.
    """
    log2e: float = 1.44269504
    ln2: float = 0.69314718
    assert num_heads % head_per_block == 0

    Hpb = head_per_block
    HEAD_BLOCKS = num_heads // Hpb
    DT = dim + tail_dim

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    accum_dtype = "float"

    @T.prim_func
    def main(
        Partial_O: T.Tensor(
            [batch, seq_len, n_groups, num_heads, DT], BF16
        ),  # type: ignore
        Partial_LSE: T.Tensor(
            [batch, seq_len, n_groups, num_heads], accum_dtype
        ),  # type: ignore
        Attn_sink: T.Tensor([num_heads], FP32),  # type: ignore
        Output: T.Tensor([batch, seq_len, num_heads, DT], BF16),  # type: ignore
        LSE: T.Tensor([batch, seq_len, num_heads], accum_dtype),  # type: ignore
    ) -> None:
        with T.Kernel(seq_len * HEAD_BLOCKS, batch, threads=threads) as (bx, by):
            shared_lse = T.alloc_shared([n_groups, Hpb], accum_dtype)

            lse_max = T.alloc_fragment([Hpb], accum_dtype)
            lse_sum = T.alloc_fragment([Hpb], accum_dtype)
            scale = T.alloc_fragment([Hpb, n_groups], accum_dtype)
            acc_o = T.alloc_fragment([Hpb, DT], accum_dtype)
            attn_sink_frag = T.alloc_fragment([Hpb], accum_dtype)
            o_scale_frag = T.alloc_fragment([Hpb], accum_dtype)
            final_lse = T.alloc_fragment([Hpb], accum_dtype)

            b_i = by
            s_i = bx // HEAD_BLOCKS
            head_block = bx % HEAD_BLOCKS
            H0 = head_block * Hpb
            H1 = H0 + Hpb

            for k in T.serial(n_groups):
                T.copy(Partial_LSE[b_i, s_i, k, H0:H1], shared_lse[k, :])

            T.fill(lse_max, -(2**30))
            for k in T.serial(n_groups):
                for h_i in T.Parallel(Hpb):
                    lse_max[h_i] = T.max(lse_max[h_i], shared_lse[k, h_i])
            T.fill(lse_sum, 0)
            for k in T.serial(n_groups):
                for h_i in T.Parallel(Hpb):
                    lse_sum[h_i] = lse_sum[h_i] + T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i]
                    )
            for k in T.serial(n_groups):
                for h_i in T.Parallel(Hpb):
                    scale[h_i, k] = T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i] - T.log2(lse_sum[h_i])
                    )

            T.fill(acc_o, 0)
            for k in T.serial(n_groups):
                for h_i, d_i in T.Parallel(Hpb, DT):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] + scale[h_i, k] * Partial_O[
                        b_i, s_i, k, H0 + h_i, d_i
                    ].astype(accum_dtype)

            # final_lse = log(sum_k exp(partial_lse_k)) (natural log)
            for h_i in T.Parallel(Hpb):
                # all partials zero -> +inf so callers can detect "no valid kv"
                empty = lse_max[h_i] <= -(2**29)
                final_lse[h_i] = T.if_then_else(
                    empty,
                    T.infinity(accum_dtype),
                    (lse_max[h_i] + T.log2(lse_sum[h_i])) * ln2,
                )

            if use_attn_sink:
                for h_i in T.Parallel(Hpb):
                    attn_sink_frag[h_i] = Attn_sink[H0 + h_i]
                for h_i in T.Parallel(Hpb):
                    empty = lse_max[h_i] <= -(2**29)
                    o_scale_frag[h_i] = T.if_then_else(
                        empty,
                        0.0,
                        1.0
                        / (
                            1.0 + T.exp2((attn_sink_frag[h_i] - final_lse[h_i]) * log2e)
                        ),
                    )
                for h_i, d_i in T.Parallel(Hpb, DT):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * o_scale_frag[h_i]

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(final_lse, LSE[b_i, s_i, H0:H1])

    return main


"""
2-stage attention kernel (partial + combine) over an FP8 KV cache,
with optional second cache (`extra_k_cache`).
"""


def dpsk_v4_fp8_attention_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Follows the original `flash_mla.flash_mla_with_kvcache` signature.
    """
    if _is_gfx95_supported:
        block_I, threads, num_stages, block_per_cu, cu = 64, 512, 0, 2, 256
    else:
        block_I, threads, num_stages, block_per_cu, cu = 32, 128, 1, 1, 304

    batch, seq_len, num_heads, _ = q.shape
    # Partial grid is (seq_len * REPLICATE_H * n_groups, batch, kv_group); the
    # heuristic in _pick_inner_iter assumes `total_blocks = seq * ni / inner_iter`,
    # so `seq` must include REPLICATE_H or n_groups doubles for medium batches.
    replicate_h = max((num_heads + 63) // 64, 1)
    seq = batch * seq_len * replicate_h

    k1, _, bs_kv_1 = _build_fp8_combined_view(k_cache)
    topk1 = indices.shape[-1]
    ni_1 = topk1 // block_I
    tk_len_1 = (
        topk_length
        if topk_length is not None
        else _topk_length_sentinel(q.device, batch)
    )
    if attn_sink is None:
        attn_sink = torch.full(
            (num_heads,), float("-inf"), dtype=torch.float32, device=q.device
        )

    has_extra = extra_k_cache is not None
    if not has_extra:
        inner_iter_1 = _pick_inner_iter(seq, ni_1, cu, block_per_cu)
        partial = dpsk_v4_fp8_partial_kernel(
            num_heads,
            topk1,
            bs_kv_1,
            sm_scale=softmax_scale,
            block_I=block_I,
            inner_iter=inner_iter_1,
            num_stages=num_stages,
            threads=threads,
        )
        partial_o, partial_lse = partial(q, k1, indices, tk_len_1)
        n_groups = ni_1 // inner_iter_1
    else:
        k2, _, bs_kv_2 = _build_fp8_combined_view(extra_k_cache)
        topk2 = extra_indices_in_kvcache.shape[-1]
        ni_2 = topk2 // block_I
        # Each phase picks its own optimal split-K independently — kernel
        # body uses two T.Pipelined loops with separate compile-time iter
        # counts, no shared-divisor constraint.
        inner_iter_1 = _pick_inner_iter(seq, ni_1, cu, block_per_cu)
        inner_iter_2 = _pick_inner_iter(seq, ni_2, cu, block_per_cu)
        tk_len_2 = (
            extra_topk_length
            if extra_topk_length is not None
            else _topk_length_sentinel(q.device, batch)
        )
        partial = dpsk_v4_fp8_partial_kernel(
            num_heads,
            topk1,
            bs_kv_1,
            topk2,
            bs_kv_2,
            sm_scale=softmax_scale,
            block_I=block_I,
            inner_iter=inner_iter_1,
            inner_iter_2=inner_iter_2,
            num_stages=num_stages,
            threads=threads,
        )
        partial_o, partial_lse = partial(
            q,
            k1,
            indices,
            tk_len_1,
            k2,
            extra_indices_in_kvcache,
            tk_len_2,
        )
        n_groups = ni_1 // inner_iter_1 + ni_2 // inner_iter_2

    combine = dpsk_v4_combine_kernel(
        num_heads,
        n_groups,
        head_per_block=4,
        threads=256,
        use_attn_sink=True,
    )
    return combine(partial_o, partial_lse, attn_sink)
