import functools
import logging
import math
from typing import Tuple

import torch

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_round_robin_split
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.utils.common import strict_contiguous

logger = logging.getLogger(__name__)

# Tilelang isn't packaged on every platform (notably Ascend NPU images) but
# this module is imported transitively from deepseek_v4.py — module-load
# must succeed even when tilelang is missing. The kernels themselves still
# require tilelang at runtime; we replace the package with a stub that lets
# `@tilelang.jit` decorations and `tilelang.PassConfigKey.*` references parse
# without ImportError, and any actual call into the kernels raises a clear
# message at execution time instead of crashing on import.
try:
    import tilelang
    import tilelang.language as T

    tilelang.set_log_level("WARNING")

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
except ImportError:

    class _TilelangMissing:
        """Stub so module-level @tilelang.jit and PassConfigKey accesses parse."""

        def __getattr__(self, name):
            if name == "jit":

                def _jit(*_args, **_kwargs):
                    def _wrap(fn):
                        def _raise(*a, **k):
                            raise RuntimeError(
                                "tilelang is not installed; this kernel cannot run "
                                "on the current platform"
                            )

                        return _raise

                    return _wrap

                return _jit
            return _TilelangMissing()

        def __call__(self, *_args, **_kwargs):
            return _TilelangMissing()

    tilelang = _TilelangMissing()
    T = _TilelangMissing()
    pass_configs = None

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    ENABLE_PDL = is_arch_support_pdl()

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(mix_hc,), T.float32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            if ENABLE_PDL:
                T.pdl_sync()

            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(
                    mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                )
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = (
                    mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])
            if ENABLE_PDL:
                T.pdl_trigger()

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes.view(-1, (2 + hc_mult) * hc_mult),
        hc_scale,
        hc_base,
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit
def mhc_pre_gemm_sqrsum_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        if ENABLE_PDL:
            T.pdl_sync()
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout(
                {x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)}
            )

            T.copy(x[px * token_block, pz * hidden_block], x_smem_16)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem_16, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                clear_accum=False,
            )
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]
        if ENABLE_PDL:
            T.pdl_trigger()


@functools.cache
def mhc_pre_gemm_sqrsum_splitk_kernel(
    hc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 32,
    hidden_block: int = 256,
    threads: int = 128,
) -> Tuple[tilelang.JITKernel, tilelang.JITKernel]:
    assert hc_mult3 <= 32
    assert hc_hidden_size % hidden_block == 0
    assert hc_hidden_size % split_k == 0
    split_size = hc_hidden_size // split_k
    assert split_size % hidden_block == 0

    num_tokens = T.dynamic("num_tokens")

    ENABLE_PDL = is_arch_support_pdl()

    @tilelang.jit
    def mhc_pre_gemm_sqrsum_splitk_stage_0(
        x: T.Tensor[(num_tokens, hc_hidden_size), T.bfloat16],
        fn: T.Tensor[(hc_mult3, hc_hidden_size), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block), split_k, threads=threads) as (
            px,
            bz,
        ):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sq_part4)

            k_base = bz * split_size

            if ENABLE_PDL:
                T.pdl_sync()

            for pz in T.Pipelined(split_size // hidden_block, num_stages=2):
                x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout(
                    {x_smem: tilelang.layout.make_swizzled_layout(x_smem)}
                )

                T.copy(x[px * token_block, k_base + pz * hidden_block], x_smem)
                T.copy(fn[0, k_base + pz * hidden_block], fn_smem)

                x_f16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem, x_f16)
                x_f = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_f16, x_f)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        v = x_f[i, jj * 4 + j]
                        sq_part4[i, j] += v * v

                T.gemm(
                    x_f,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                )

            sq_l = T.alloc_fragment((token_block,), T.float32)
            T.reduce_sum(sq_part4, sq_l)

            for i in T.Parallel(token_block):
                t = px * token_block + i
                if t < num_tokens:
                    sqrsum_partial[bz, t] = sq_l[i]

            for i, j in T.Parallel(token_block, 32):
                t = px * token_block + i
                if t < num_tokens:
                    out_partial[bz, t, j] = out_frag[i, j]

            if ENABLE_PDL:
                T.pdl_trigger()

    @tilelang.jit
    def mhc_pre_gemm_sqrsum_splitk_stage_1(
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
        out: T.Tensor[(num_tokens, hc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens,), T.float32],
    ):
        warps_per_cta = threads // 32
        num_reduce = T.ceildiv(split_k, 32)
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=threads) as (px,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            t = px * warps_per_cta + warp
            s = T.alloc_local((1,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            s[0] = 0
            acc[0] = 0
            if ENABLE_PDL:
                T.pdl_sync()

            if t < num_tokens:
                for r in T.serial(num_reduce):
                    bz = r * 32 + lane
                    s[0] += T.if_then_else(bz < split_k, sqrsum_partial[bz, t], 0.0)
                sqrsum[t] = T.warp_reduce_sum(s[0])
                if lane < hc_mult3:
                    for bz in T.serial(split_k):
                        acc[0] += out_partial[bz, t, lane]
                    out[t, lane] = acc[0]

            if ENABLE_PDL:
                T.pdl_trigger()

    return (
        mhc_pre_gemm_sqrsum_splitk_stage_0,
        mhc_pre_gemm_sqrsum_splitk_stage_1,
    )


def _compute_num_split_for_mhc_pre(num_tokens: int, hc_hidden_size: int) -> int:
    block_m, block_k = 64, 64
    grid_size = (num_tokens + block_m - 1) // block_m
    num_block_k = (hc_hidden_size + block_k - 1) // block_k
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(n_sms // max(grid_size, 1), num_block_k // 4))


def get_mhc_pre_token_count_representatives(
    max_num_tokens: int, hc_hidden_size: int
) -> Tuple[int, ...]:
    """One representative token count per distinct mhc_pre n_splits bucket over
    [1, max_num_tokens] (the kernel is specialized only by n_splits)."""
    reps = {}
    for grid in range(1, (max(1, max_num_tokens) + 63) // 64 + 1):
        num_tokens = min(grid * 64, max_num_tokens)
        reps[_compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)] = num_tokens
    return tuple(sorted(reps.values()))


def prewarm_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    n_splits_pre: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
):
    """Compile the prenorm kernel for every n_splits bucket by replaying the
    prenorm with the call's real weights. The compiled kernels are written to
    the TileLang/DeepGEMM on-disk JIT cache, so this cost is paid only on a cold
    cache; later server runs hit the cache. Driven once per process from load_weights.
    """
    from sglang.srt.server_args import get_global_server_args

    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    max_num_tokens = get_global_server_args().chunked_prefill_size
    buckets = get_mhc_pre_token_count_representatives(
        max_num_tokens, hc_mult * hidden_size
    )

    logger.info("DeepSeek V4 MHC prenorm prewarm: %d n_splits buckets", len(buckets))
    with torch.inference_mode():
        for num_tokens in buckets:
            mhc_pre(
                residual.new_zeros(num_tokens, hc_mult, hidden_size),
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                n_splits_pre,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    """Fused mhc_pre big_fuse + RMSNorm of layer_input.

    Identical to mhc_pre_big_fuse_tilelang for the (post_mix, comb_mix) path.
    For the layer_input path, the weighted-sum result is stashed in shared
    memory while accumulating sum_sq, then a second pipelined sweep applies
    rsqrt(sum_sq/D + norm_eps) * norm_weight before writing to HBM.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            # Stash unnormalized weighted-sum output in shared memory as bf16
            # (matches the rounding the reference path does when RMSNorm reads bf16).
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=3):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    # layer_input is the post-norm activation fed into the MoE. Allocate it in
    # the symmetric memory pool so the downstream all-reduce uses the low-latency
    # NCCL symmetric path: the Triton inplace MoE runner writes the expert
    # output back into this buffer, so a symmetric input yields a symmetric
    # all-reduce input.
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        layer_input = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        )

    if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)

        gemm_out_mul = torch.empty(
            n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual_flat.view(num_tokens, hc_hidden_size),
            fn_flat,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
        gemm_last_dim = hc_mult3
        big_fuse_n_splits = n_splits
    else:
        if num_tokens <= 2048:
            assert n_splits == 1
            if hc_hidden_size == 16384:
                hidden_block = 256
            elif hc_hidden_size == 28672:
                hidden_block = 128
            else:
                raise NotImplementedError(
                    f"mhc_pre splitk kernel only supports hc_hidden_size in {{16384, 28672}}, "
                    f"got {hc_hidden_size}"
                )
            kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
                hc_mult3,
                hc_hidden_size,
                split_k=n_splits_pre,
                token_block=32,
                hidden_block=hidden_block,
            )
            partial_out = torch.empty(
                n_splits_pre,
                num_tokens,
                32,
                dtype=torch.float32,
                device=residual.device,
            )
            partial_sqrsum = torch.empty(
                n_splits_pre, num_tokens, dtype=torch.float32, device=residual.device
            )
            kernel_0(
                residual_flat.view(num_tokens, hc_hidden_size),
                fn_flat,
                partial_out,
                partial_sqrsum,
            )
            # Stage_1 reduction is folded into big_fuse below; skip launching it.
            gemm_out_mul = partial_out
            gemm_out_sqrsum = partial_sqrsum
            gemm_last_dim = 32
            big_fuse_n_splits = n_splits_pre
        else:
            gemm_out_mul = torch.empty(
                n_splits,
                num_tokens,
                hc_mult3,
                dtype=torch.float32,
                device=residual.device,
            )
            gemm_out_sqrsum = torch.empty(
                n_splits, num_tokens, dtype=torch.float32, device=residual.device
            )
            assert (
                n_splits == 1
            ), "The simple TileLang version gemm_sqrsum doesn't support split-k"
            mhc_pre_gemm_sqrsum_tilelang(
                residual_flat.view(num_tokens, hc_mult * hidden_size),
                fn_flat,
                gemm_out_mul.squeeze(0),
                gemm_out_sqrsum.squeeze(0),
                hc_mult3,
                hc_mult * hidden_size,
            )
            gemm_last_dim = hc_mult3
            big_fuse_n_splits = n_splits

    if norm_weight is not None:
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"
        assert norm_weight.shape == (
            hidden_size,
        ), f"norm_weight shape {tuple(norm_weight.shape)} != (hidden_size={hidden_size},)"
        norm_weight_bf = (
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():
            norm_weight_bf = norm_weight_bf.contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )
    else:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(n, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if is_dsa_prefill_cp_round_robin_split():
        x = strict_contiguous(x)
        residual = strict_contiguous(residual)
        post_layer_mix = strict_contiguous(post_layer_mix)
        comb_res_mix = strict_contiguous(comb_res_mix)
    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_fused_post_pre_fma_tilelang(
    prev_comb_mix,
    prev_residual,
    prev_post_mix,
    hidden_in,
    pre_fn,
    mixes_partial_out,
    sqrsum_partial_out,
    cur_residual_out,
    hc: int,
    hidden_size: int,
    num_mix_outputs: int,
    n_thr: int = 256,
    tile_mix_outputs: int = 1,
    split_k: int = 1,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")

    hidden_per_split = (hidden_size + split_k - 1) // split_k
    num_mix_output_tiles = (num_mix_outputs + tile_mix_outputs - 1) // tile_mix_outputs

    prev_comb_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    prev_residual: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)
    prev_post_mix: T.Tensor((num_tokens, hc), T.float32)
    hidden_in: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    pre_fn: T.Tensor((num_mix_outputs, hc, hidden_size), T.float32)

    mixes_partial_out: T.Tensor((split_k, num_tokens, num_mix_outputs), T.float32)
    sqrsum_partial_out: T.Tensor((split_k, num_tokens), T.float32)
    cur_residual_out: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)

    hidden_iters_per_thread = (hidden_per_split + n_thr - 1) // n_thr
    num_warps = n_thr // 32

    ENABLE_PDL = is_arch_support_pdl()

    # CTA assignment:
    #   token_idx           : this CTA handles one token.
    #   mix_output_tile_idx : this CTA handles a small tile of mix output columns.
    #                          For HC=4, num_mix_outputs = 24:
    #                            [0:4]   -> pre logits
    #                            [4:8]   -> post logits
    #                            [8:24]  -> comb logits
    #   hidden_split_idx    : this CTA handles one split of the hidden dimension.
    #
    # Thread assignment inside one CTA:
    #   Each thread owns several hidden positions in this hidden split:
    #     hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx
    #
    # For each owned hidden_idx, the thread computes:
    #   1. post result: cur_residual[token, :, hidden_idx]
    #   2. sqrsum partial for pre RMS
    #   3. GEMM partial for several mix output columns
    with T.Kernel(
        num_tokens,
        num_mix_output_tiles,
        split_k,
        threads=n_thr,
    ) as (token_idx, mix_output_tile_idx, hidden_split_idx):
        thread_idx = T.get_thread_binding()
        warp_idx = T.get_warp_idx()
        lane_idx = T.get_lane_idx()

        warp_partials = T.alloc_shared((num_warps, tile_mix_outputs + 1), T.float32)
        post_mix_smem = T.alloc_shared((hc,), T.float32)
        comb_mix_smem = T.alloc_shared((hc, hc), T.float32)

        post_mix_for_token = T.alloc_local((hc,), T.float32)
        comb_mix_for_token = T.alloc_local((hc, hc), T.float32)

        mix_acc = T.alloc_local((tile_mix_outputs,), T.float32)
        sqrsum_acc = T.alloc_local((1,), T.float32)
        cur_residual_values = T.alloc_local((hc,), T.float32)

        T.clear(mix_acc)
        T.clear(sqrsum_acc)

        hidden_split_start = hidden_split_idx * hidden_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        # Load post/comb coefficients for this token.
        #
        # PyTorch equivalent:
        #   post = prev_post_mix[token_idx]      # [HC]
        #   comb = prev_comb_mix[token_idx]      # [HC, HC]
        T.copy(prev_post_mix[token_idx, 0], post_mix_smem)
        T.copy(prev_comb_mix[token_idx, 0, 0], comb_mix_smem)

        for route_idx in T.unroll(hc):
            post_mix_for_token[route_idx] = post_mix_smem[route_idx]

        for old_route_idx in T.unroll(hc):
            for new_route_idx in T.unroll(hc):
                comb_mix_for_token[old_route_idx, new_route_idx] = comb_mix_smem[
                    old_route_idx, new_route_idx
                ]

        for hidden_iter in T.serial(hidden_iters_per_thread):
            hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx

            if hidden_idx < hidden_size:
                # Step A: fused post.
                #
                # PyTorch equivalent:
                #   cur_residual =
                #       post.unsqueeze(-1) * hidden_in.unsqueeze(1)
                #       + (
                #           comb.unsqueeze(-1)
                #           * prev_residual.unsqueeze(2)
                #         ).sum(dim=1)
                #
                # Scalar form for this token and this hidden position:
                #   cur_residual[j, h]
                #     = post[j] * hidden_in[h]
                #     + sum_k comb[k, j] * prev_residual[k, h]
                for new_route_idx in T.unroll(hc):
                    cur_residual_values[new_route_idx] = (
                        post_mix_for_token[new_route_idx]
                        * hidden_in[token_idx, hidden_idx]
                    )

                    for old_route_idx in T.unroll(hc):
                        cur_residual_values[new_route_idx] += (
                            comb_mix_for_token[old_route_idx, new_route_idx]
                            * prev_residual[token_idx, old_route_idx, hidden_idx]
                        )

                # Match the unfused path:
                #   mhc_post writes bf16 residual,
                #   then mhc_pre reads bf16 residual.
                for route_idx in T.unroll(hc):
                    cur_residual_values[route_idx] = T.bfloat16(
                        cur_residual_values[route_idx]
                    )

                # Step B1: pre sqrsum partial.
                #
                # PyTorch equivalent:
                #   x_flat = cur_residual.reshape(T, HC * H).float()
                #   sqrsum = (x_flat * x_flat).sum(dim=-1)
                #
                # Only mix_output_tile_idx == 0 writes cur_residual and sqrsum,
                # otherwise different output-column CTAs would duplicate this work.
                if mix_output_tile_idx == 0:
                    for route_idx in T.unroll(hc):
                        cur_residual_out[token_idx, route_idx, hidden_idx] = (
                            cur_residual_values[route_idx]
                        )
                        sqrsum_acc[0] += (
                            cur_residual_values[route_idx]
                            * cur_residual_values[route_idx]
                        )

                # Step B2: pre GEMM partial.
                #
                # PyTorch equivalent:
                #   mixes = F.linear(x_flat, fn)
                #
                # Scalar form:
                #   mixes[token, o] +=
                #       pre_fn[o, route, hidden] * cur_residual[route, hidden]
                #
                # This CTA computes only tile_mix_outputs columns of mixes.
                for tile_col_idx in T.unroll(tile_mix_outputs):
                    mix_output_idx = (
                        mix_output_tile_idx * tile_mix_outputs + tile_col_idx
                    )

                    if mix_output_idx < num_mix_outputs:
                        for route_idx in T.unroll(hc):
                            mix_acc[tile_col_idx] += (
                                pre_fn[mix_output_idx, route_idx, hidden_idx]
                                * cur_residual_values[route_idx]
                            )

        # Reduce thread partials inside each warp.
        for tile_col_idx in T.unroll(tile_mix_outputs):
            mix_acc[tile_col_idx] = T.warp_reduce_sum(mix_acc[tile_col_idx])

        if mix_output_tile_idx == 0:
            sqrsum_acc[0] = T.warp_reduce_sum(sqrsum_acc[0])

        # One lane per warp writes warp-level partials to shared memory.
        if lane_idx == 0:
            for tile_col_idx in T.unroll(tile_mix_outputs):
                warp_partials[warp_idx, tile_col_idx] = mix_acc[tile_col_idx]

            if mix_output_tile_idx == 0:
                warp_partials[warp_idx, tile_mix_outputs] = sqrsum_acc[0]

        T.sync_threads()

        # Reduce across warps and write split partials.
        #
        # The full PyTorch result would be:
        #   mixes = F.linear(cur_residual.reshape(T, HC * H), fn)
        #   sqrsum = (cur_residual.float() ** 2).sum(dim=(1, 2))
        #
        # This kernel is split along hidden, so each CTA writes only:
        #   mixes_partial_out[hidden_split_idx, token, o]
        #   sqrsum_partial_out[hidden_split_idx, token]
        #
        # Later mhc_pre_big_fuse does:
        #   mixes = mixes_partial_out.sum(dim=0)
        #   sqrsum = sqrsum_partial_out.sum(dim=0)
        #   rms = rsqrt(sqrsum / (HC * H) + eps)
        #   mixes *= rms
        #   mixes -> pre/post/comb
        #   layer_input = sum_j pre[j] * cur_residual[j]
        if warp_idx == 0:
            for tile_col_idx in T.unroll(tile_mix_outputs):
                mix_output_idx = mix_output_tile_idx * tile_mix_outputs + tile_col_idx

                if mix_output_idx < num_mix_outputs and lane_idx == tile_col_idx:
                    mix_output_partial = T.alloc_var(T.float32, init=0.0)

                    for reduce_warp_idx in T.unroll(num_warps):
                        mix_output_partial += warp_partials[
                            reduce_warp_idx, tile_col_idx
                        ]

                    mixes_partial_out[hidden_split_idx, token_idx, mix_output_idx] = (
                        mix_output_partial
                    )

            if mix_output_tile_idx == 0 and lane_idx == 0:
                sqrsum_partial = T.alloc_var(T.float32, init=0.0)

                for reduce_warp_idx in T.unroll(num_warps):
                    sqrsum_partial += warp_partials[reduce_warp_idx, tile_mix_outputs]

                sqrsum_partial_out[hidden_split_idx, token_idx] = sqrsum_partial

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the boundary between one mHC post step and the next mHC pre step.

    The unfused sequence is ``mhc_post -> pre-norm GEMM -> mhc_pre big_fuse``.
    This wrapper keeps the numerically sensitive ``mhc_pre_big_fuse`` stage,
    including optional RMSNorm, but removes the separate post/pre boundary.
    Small token batches use the FMA kernel above to combine ``mhc_post`` and the
    pre-norm GEMM in one launch; larger batches keep DeepGEMM for throughput and
    only fuse the Python/model-level scheduling boundary.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        # Some DP/EP ranks can receive no tokens; return correctly typed empty
        # tensors so later fused layers keep the same contracts as mhc_pre/hc_post.
        return (
            torch.empty_like(residual),
            torch.empty(
                (*outer_shape, hc_mult, 1), dtype=torch.float32, device=residual.device
            ),
            torch.empty(
                (*outer_shape, hc_mult, hc_mult),
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                (*outer_shape, hidden_size),
                dtype=torch.bfloat16,
                device=residual.device,
            ),
        )
    x_flat = x.view(num_tokens, hidden_size)

    # The scalar-FMA kernel wins only for small batches where launch
    # overhead dominates; beyond the threshold DeepGEMM's tensor-core path wins.
    fma_token_threshold = 32
    if num_tokens <= fma_token_threshold:
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)

    if num_tokens <= fma_token_threshold:
        # Small-batch path: one TileLang launch computes hc_post, the bf16
        # residual write, GEMM partials, and the RMS square-sum partials.
        mhc_fused_post_pre_fma_tilelang(
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult),
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_mix_outputs=tile_n,
            split_k=n_splits,
        )
    else:
        # Large-batch path: keep the existing high-throughput TileLang hc_post +
        # DeepGEMM pre-norm GEMM decomposition instead of replacing tensor cores.
        mhc_post_tilelang(
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult),
            x_flat,
            residual_cur,
            hc_mult,
            hidden_size,
        )

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            import deep_gemm

            deep_gemm.tf32_hc_prenorm_gemm(
                residual_cur.view(num_tokens, hc_hidden_size),
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                num_splits=n_splits,
            )
        else:
            # Fallback mirrors mhc_pre when DeepGEMM prenorm is disabled.
            n_splits = 1
            gemm_out_mul_2d = torch.empty(
                num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
            )
            gemm_out_sqrsum_1d = torch.empty(
                num_tokens, dtype=torch.float32, device=residual.device
            )
            mhc_pre_gemm_sqrsum_tilelang(
                residual_cur.view(num_tokens, hc_hidden_size),
                fn,
                gemm_out_mul_2d,
                gemm_out_sqrsum_1d,
                hc_mult3,
                hc_hidden_size,
            )
            gemm_out_mul = gemm_out_mul_2d.unsqueeze(0)
            gemm_out_sqrsum = gemm_out_sqrsum_1d.unsqueeze(0)

    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    # layer_input_cur is the post-norm activation fed into the MoE; allocate it
    # in the symmetric memory pool so the Triton inplace MoE runner yields a
    # symmetric all-reduce input (see _mhc_pre_impl).
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        layer_input_cur = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        )

    if norm_weight is not None:
        # Final mhc_pre stage: convert GEMM partials into post/comb/layer_input
        # and fuse the following RMSNorm when the model passed a norm weight.
        assert norm_eps is not None
        assert norm_weight.shape == (hidden_size,)
        norm_weight_bf = (
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():
            norm_weight_bf = norm_weight_bf.contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            hc_mult3,
        )
    else:
        # Same mhc_pre finalization without the model-layer RMSNorm.
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            hc_mult3,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def npu_hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    rms_norm_eps: float,
    hc_eps: float,
    forward_batch=None,
) -> tuple:
    """NPU-accelerated hc_pre via the custom_ops kernel.

    Returns (y, post, comb, norm_fused).  norm_fused is always False
    because npu_hc_pre does not fold input_layernorm — the caller must
    apply it separately.
    """
    shape, dtype = x.size(), x.dtype

    # IDLE / empty short-circuit, mirroring the dsv4-flash source.
    # The kernel emits post/comb in fp32 (sinkhorn iterates in fp32),
    # so the dummies must too — otherwise downstream comb/post-aware
    # ops see a silent fp32 ↔ bf16 split between idle and non-idle
    # batches.
    is_idle = forward_batch is not None and forward_batch.forward_mode.is_idle()
    if is_idle or x.shape[0] == 0:
        bs = x.shape[0]
        y = torch.empty((bs, shape[-1]), dtype=dtype, device=x.device)
        post = torch.empty((bs, hc_mult), dtype=torch.float32, device=x.device)
        comb = torch.empty(
            (bs, hc_mult, hc_mult),
            dtype=torch.float32,
            device=x.device,
        )
        return y, post, comb, False

    # Note the return order: (y, post, comb) — y is the (T, hidden)
    # mixed activation, post / comb are the hc_post inputs. The
    # fused kernel emits y in fp32 (sinkhorn iterates in fp32), so
    # cast back to the input dtype before the downstream
    # aclnnRmsNorm (which has no x=fp32 / gamma=bf16 overload).
    y, post, comb = torch.ops.custom.npu_hc_pre(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=rms_norm_eps,
        hc_eps=hc_eps,
    )
    # npu_hc_pre uses norm_eps for sinkhorn's internal RMS only; it does
    # not fold input_layernorm. Return norm_fused=False so the caller
    # applies the layernorm itself, matching the deepgemm/torch paths.
    return y.to(dtype), post, comb, False
