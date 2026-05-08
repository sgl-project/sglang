import functools
import math
from typing import Tuple

import tilelang
import tilelang.language as T
import torch

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_round_robin_split
from sglang.srt.layers.utils.common import strict_contiguous

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

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
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
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
                wg_wait=0,
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
                    wg_wait=0,
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
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

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
        kernel_0, kernel_1 = mhc_pre_gemm_sqrsum_splitk_kernel(
            hc_mult3,
            hc_hidden_size,
            split_k=n_splits_pre,
            token_block=32,
            hidden_block=hidden_block,
        )
        partial_out = gemm_out_mul.new_empty(n_splits_pre, num_tokens, 32)
        partial_sqrsum = gemm_out_sqrsum.new_empty(n_splits_pre, num_tokens)
        kernel_0(
            residual_flat.view(num_tokens, hc_hidden_size),
            fn_flat,
            partial_out,
            partial_sqrsum,
        )
        kernel_1(
            partial_out,
            partial_sqrsum,
            gemm_out_mul.squeeze(0),
            gemm_out_sqrsum.squeeze(0),
        )
        del partial_out, partial_sqrsum
    else:
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
        n_splits,
        hc_mult,
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
    if is_nsa_prefill_cp_round_robin_split():
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
