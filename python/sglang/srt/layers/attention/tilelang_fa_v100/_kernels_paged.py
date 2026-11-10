"""Paged FlashAttention forward kernel for V100 (SM70).
   4D page-by-page loading handles scattered vLLM page blocks correctly.
   Dynamic tensor shapes via T.dynamic. Supports causal full attention, causal
   sliding-window attention, and bidirectional full attention.
   Supports split-KV for long-sequence parallelism (num_splits > 1).

   HD 512 (dim > 256) uses KV-union optimization where K and V share the same
   shared memory slot since they have non-overlapping lifetimes in the loop.
   This saves 32KB+ shared memory, enabling block_M=32, block_N=32, threads=128
   which doubles warp count from 2 to 4, improving V100 occupancy.
"""
import math
import torch
import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")

# Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
# `out_idx` list in place. Patch once on import (idempotent). Mirrors
# sglang/srt/layers/attention/dsa/tilelang_kernel.py.
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
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_USE_KV_UNION_FOR_DIM = 256  # use KV union when dim > this value


@tilelang.jit(out_idx=[9], pass_configs=pass_configs)
def _paged_kernel_func(batch, heads, heads_kv, dim, page_block_size,
                       max_blocks_per_seq, num_pages, is_causal,
                       sliding_window_size=-1,
                       block_M=32, block_N=128, num_stages=0, threads=256,
                       num_splits=1):
    scale = (1.0 / dim) ** 0.5
    nt = T.dynamic("nt")
    use_kv_union = dim > _USE_KV_UNION_FOR_DIM

    if num_splits > 1:
        # ── Split-KV variant ────────────────────────────────────────────────
        @T.prim_func
        def main(
            Q: T.Tensor([nt, heads, dim], T.float16),
            K_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
            V_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
            block_table: T.Tensor([batch, max_blocks_per_seq], T.int32),
            cache_seqlens: T.Tensor([batch], T.int32),
            query_start_loc: T.Tensor([batch + 1], T.int32),
            prefix_kv_lens: T.Tensor([batch], T.int32),
            max_tokens: T.int32,
            sm_scale: T.float32,
            Output: T.Tensor([nt, heads, dim], T.float16),
        ):
            num_tiles = T.ceildiv(max_tokens, block_M)
            glse = T.alloc_global([nt, num_splits], T.float32)
            op = T.alloc_global([nt, num_splits, dim], T.float16)

            # ── Split phase: each block handles ONE Q-tile, iterates over all splits ─
            with T.Kernel(num_tiles, heads, batch,
                          threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], T.float16)
                if use_kv_union:
                    KV_shared = T.alloc_shared([block_N, dim], T.float16)
                    K_ref = KV_shared
                    V_ref = KV_shared
                else:
                    K_ref = T.alloc_shared([block_N, dim], T.float16)
                    V_ref = T.alloc_shared([block_N, dim], T.float16)
                P_shared = T.alloc_shared([block_M, block_N], T.float16)

                acc_s = T.alloc_fragment([block_M, block_N], T.float32)
                acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
                acc_o = T.alloc_fragment([block_M, dim], T.float32)
                m_i = T.alloc_fragment([block_M], T.float32)
                m_prev = T.alloc_fragment([block_M], T.float32)
                l_i = T.alloc_fragment([block_M], T.float32)
                sf = T.alloc_fragment([block_M], T.float32)
                row_sum = T.alloc_fragment([block_M], T.float32)

                kv_head = by // (heads // heads_kv)
                q_len = query_start_loc[bz + 1] - query_start_loc[bz]
                start_q = query_start_loc[bz] + bx * block_M
                total_kv = cache_seqlens[bz]
                split_len = T.ceildiv(total_kv, num_splits)

                # Running state across splits (online softmax) — allocated unconditionally
                max_state = T.alloc_fragment([block_M], T.float32)
                exp_sum = T.alloc_fragment([block_M], T.float32)
                out_sum = T.alloc_fragment([block_M, dim], T.float32)
                old_max = T.alloc_fragment([block_M], T.float32)

                with T.If(bx * block_M < q_len), T.Then():
                    T.clear(Q_shared)
                    for i, j in T.Parallel(block_M, dim):
                        if start_q + i < query_start_loc[bz + 1]:
                            Q_shared[i, j] = Q[start_q + i, by, j]

                    T.fill(max_state, -T.infinity(T.float32))
                    T.fill(exp_sum, 0)
                    T.fill(out_sum, 0)

                    for sid in T.serial(num_splits):
                        split_start = sid * split_len
                        split_end = T.min(split_start + split_len, total_kv)
                        kv_blocks = T.ceildiv(split_end - split_start, block_N)

                        T.fill(acc_o, 0)
                        T.fill(m_i, -T.infinity(T.float32))
                        T.fill(l_i, 0)

                        for k in T.Pipelined(kv_blocks, num_stages=num_stages):
                            T.clear(K_ref)
                            for i, j in T.Parallel(block_N, dim):
                                gkv = split_start + k * block_N + i
                                lp = T.floordiv(gkv, page_block_size)
                                off = gkv - lp * page_block_size
                                if lp < max_blocks_per_seq:
                                    phys = block_table[bz, lp]
                                    K_ref[i, j] = K_cache[phys, off, kv_head, j]

                            if is_causal:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_pos = bx * block_M + i
                                    kv_pos = split_start + k * block_N + j
                                    window_ok = (
                                        kv_pos
                                        >= prefix_kv_lens[bz]
                                        + q_pos
                                        - sliding_window_size
                                        if sliding_window_size >= 0
                                        else True
                                    )
                                    acc_s[i, j] = T.if_then_else(
                                        (kv_pos <= prefix_kv_lens[bz] + q_pos)
                                        & window_ok,
                                        T.if_then_else(kv_pos < cache_seqlens[bz], T.cast(0, T.float32), -T.infinity(T.float32)),
                                        -T.infinity(T.float32)
                                    )
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    kv_pos = split_start + k * block_N + j
                                    acc_s[i, j] = T.if_then_else(
                                        kv_pos < cache_seqlens[bz], 0, -T.infinity(acc_s.dtype)
                                    )

                            T.gemm(Q_shared, K_ref, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(m_i, m_prev)
                            T.reduce_max(acc_s, m_i, dim=1, clear=False)
                            for i in T.Parallel(block_M):
                                m_i[i] = T.if_then_else(m_i[i] == -T.infinity(T.float32), T.cast(0, T.float32), m_i[i])
                            for i in T.Parallel(block_M):
                                m_i[i] = T.max(m_i[i], m_prev[i])
                            for i in T.Parallel(block_M):
                                sf[i] = T.exp(m_prev[i] * sm_scale - m_i[i] * sm_scale)
                                l_i[i] *= sf[i]
                            for i, j in T.Parallel(block_M, dim):
                                acc_o[i, j] *= sf[i]
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp(acc_s[i, j] * sm_scale - m_i[i] * sm_scale)
                            T.reduce_sum(acc_s, row_sum, dim=1)
                            for i in T.Parallel(block_M):
                                l_i[i] += row_sum[i]

                            T.clear(V_ref)
                            for i, j in T.Parallel(block_N, dim):
                                gkv2 = split_start + k * block_N + i
                                lp2 = T.floordiv(gkv2, page_block_size)
                                off2 = gkv2 - lp2 * page_block_size
                                if lp2 < max_blocks_per_seq:
                                    phys2 = block_table[bz, lp2]
                                    V_ref[i, j] = V_cache[phys2, off2, kv_head, j]

                            for i, j in T.Parallel(block_M, block_N):
                                P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                            T.copy(P_shared, acc_s_cast)
                            T.gemm(acc_s_cast, V_ref, acc_o, policy=T.GemmWarpPolicy.Square)

                        # Merge this split into running state (online softmax across splits)
                        for i in T.Parallel(block_M):
                            old_max[i] = max_state[i]
                            new_max = T.max(max_state[i], m_i[i])
                            rescale_old = T.exp((old_max[i] - new_max) * sm_scale)
                            rescale_new = T.exp((m_i[i] - new_max) * sm_scale)
                            exp_sum[i] = exp_sum[i] * rescale_old + l_i[i] * rescale_new
                            max_state[i] = new_max
                        for i, j in T.Parallel(block_M, dim):
                            new_max = T.max(old_max[i], m_i[i])
                            rescale_old = T.exp((old_max[i] - new_max) * sm_scale)
                            rescale_new = T.exp((m_i[i] - new_max) * sm_scale)
                            out_sum[i, j] = out_sum[i, j] * rescale_old + acc_o[i, j] * rescale_new

                    for i, j in T.Parallel(block_M, dim):
                        if start_q + i < query_start_loc[bz + 1]:
                            Output[start_q + i, by, j] = T.cast(out_sum[i, j] / exp_sum[i], T.float16)

    else:
        # ── No-split variant (original kernel) ──────────────────────────────
        @T.prim_func
        def main(
            Q: T.Tensor([nt, heads, dim], T.float16),
            K_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
            V_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
            block_table: T.Tensor([batch, max_blocks_per_seq], T.int32),
            cache_seqlens: T.Tensor([batch], T.int32),
            query_start_loc: T.Tensor([batch + 1], T.int32),
            prefix_kv_lens: T.Tensor([batch], T.int32),
            max_tokens: T.int32,
            sm_scale: T.float32,
            Output: T.Tensor([nt, heads, dim], T.float16),
        ):
                with T.Kernel(T.ceildiv(max_tokens, block_M), heads, batch, threads=threads) as (bx, by, bz):
                    Q_shared = T.alloc_shared([block_M, dim], T.float16)
                    if use_kv_union:
                        KV_shared = T.alloc_shared([block_N, dim], T.float16)
                        K_ref = KV_shared
                        V_ref = KV_shared
                    else:
                        K_ref = T.alloc_shared([block_N, dim], T.float16)
                        V_ref = T.alloc_shared([block_N, dim], T.float16)
                    P_shared = T.alloc_shared([block_M, block_N], T.float16)

                    acc_s = T.alloc_fragment([block_M, block_N], T.float32)
                    acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
                    acc_o = T.alloc_fragment([block_M, dim], T.float32)
                    m_i = T.alloc_fragment([block_M], T.float32)
                    m_prev = T.alloc_fragment([block_M], T.float32)
                    l_i = T.alloc_fragment([block_M], T.float32)
                    sf = T.alloc_fragment([block_M], T.float32)
                    row_sum = T.alloc_fragment([block_M], T.float32)

                    kv_head = by // (heads // heads_kv)
                    q_len = query_start_loc[bz + 1] - query_start_loc[bz]
                    start_q = query_start_loc[bz] + bx * block_M

                    with T.If(bx * block_M < q_len), T.Then():
                        T.clear(Q_shared)
                        for i, j in T.Parallel(block_M, dim):
                            if start_q + i < query_start_loc[bz + 1]:
                                Q_shared[i, j] = Q[start_q + i, by, j]
                        T.fill(acc_o, 0)
                        T.fill(m_i, -T.infinity(T.float32))
                        T.fill(l_i, 0)

                        loop_start = (
                            T.max(
                                0,
                                T.floordiv(
                                    prefix_kv_lens[bz]
                                    + bx * block_M
                                    - sliding_window_size,
                                    block_N,
                                ),
                            )
                            if is_causal and sliding_window_size >= 0
                            else 0
                        )
                        loop_end = (
                            T.min(T.ceildiv(cache_seqlens[bz], block_N),
                                  T.ceildiv(prefix_kv_lens[bz] + bx * block_M + block_M, block_N))
                            if is_causal
                            else T.ceildiv(cache_seqlens[bz], block_N)
                        )

                        num_loop_blocks = loop_end - loop_start
                        for k_offset in T.Pipelined(
                            num_loop_blocks, num_stages=num_stages
                        ):
                            k = loop_start + k_offset
                            T.clear(K_ref)
                            for i, j in T.Parallel(block_N, dim):
                                gkv = k * block_N + i
                                lp = T.floordiv(gkv, page_block_size)
                                off = gkv - lp * page_block_size
                                if lp < max_blocks_per_seq:
                                    phys = block_table[bz, lp]
                                    K_ref[i, j] = K_cache[phys, off, kv_head, j]

                            if is_causal:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_pos = bx * block_M + i
                                    kv_pos = k * block_N + j
                                    causal_ok = kv_pos <= prefix_kv_lens[bz] + q_pos
                                    window_ok = (
                                        kv_pos
                                        >= prefix_kv_lens[bz]
                                        + q_pos
                                        - sliding_window_size
                                        if sliding_window_size >= 0
                                        else True
                                    )
                                    seq_ok = kv_pos < cache_seqlens[bz]
                                    acc_s[i, j] = T.if_then_else(
                                        causal_ok & window_ok & seq_ok,
                                        0,
                                        -T.infinity(acc_s.dtype),
                                    )
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    acc_s[i, j] = T.if_then_else(
                                        k * block_N + j < cache_seqlens[bz], 0, -T.infinity(acc_s.dtype)
                                    )

                            T.gemm(Q_shared, K_ref, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(m_i, m_prev)
                            T.reduce_max(acc_s, m_i, dim=1, clear=False)
                            for i in T.Parallel(block_M):
                                m_i[i] = T.if_then_else(m_i[i] == -T.infinity(T.float32), T.cast(0, T.float32), m_i[i])
                            for i in T.Parallel(block_M):
                                m_i[i] = T.max(m_i[i], m_prev[i])
                            for i in T.Parallel(block_M):
                                sf[i] = T.exp(m_prev[i] * sm_scale - m_i[i] * sm_scale)
                                l_i[i] *= sf[i]
                            for i, j in T.Parallel(block_M, dim):
                                acc_o[i, j] *= sf[i]
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp(acc_s[i, j] * sm_scale - m_i[i] * sm_scale)
                            T.reduce_sum(acc_s, row_sum, dim=1)
                            for i in T.Parallel(block_M):
                                l_i[i] += row_sum[i]

                            T.clear(V_ref)
                            for i, j in T.Parallel(block_N, dim):
                                gkv2 = k * block_N + i
                                lp2 = T.floordiv(gkv2, page_block_size)
                                off2 = gkv2 - lp2 * page_block_size
                                if lp2 < max_blocks_per_seq:
                                    phys2 = block_table[bz, lp2]
                                    V_ref[i, j] = V_cache[phys2, off2, kv_head, j]

                            for i, j in T.Parallel(block_M, block_N):
                                P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                            T.copy(P_shared, acc_s_cast)
                            T.gemm(acc_s_cast, V_ref, acc_o, policy=T.GemmWarpPolicy.Square)

                        for i, j in T.Parallel(block_M, dim):
                            if start_q + i < query_start_loc[bz + 1]:
                                Output[start_q + i, by, j] = T.cast(acc_o[i, j] / l_i[i], T.float16)

    return main


_KERNEL_CACHE = {}

_BEST_CONFIGS = {
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    256: dict(block_M=64, block_N=32, threads=256, num_stages=0, num_splits=1),
    # HD 512: Uses KV-union (K+V share shared memory slot) to fit 96KB V100 limit.
    #   With union: Q(32KB) + KV(32KB) + P(2KB) = 66KB fits, enabling 4 warps.
    #   Without union: 99KB exceeds V100 limit.
    #   MMA constraint: block_N=32 → n_warp ≤ 2. With block_M=32, m_warp=2 → 4 warps.
    512: dict(block_M=32, block_N=32, threads=128, num_stages=0, num_splits=1),
}


def get_paged_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                     max_blocks, causal, sliding_window_size=-1):
    """Return compiled kernel."""
    cfg = _BEST_CONFIGS.get(dim, dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1))
    key = (heads, heads_kv, dim, block_size, causal, sliding_window_size,
           cfg["block_M"], cfg["block_N"], cfg["threads"], cfg["num_stages"], cfg["num_splits"],
           batch, max_blocks, num_pages)
    if key not in _KERNEL_CACHE:
        kt = _paged_kernel_func(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            **cfg,
        )
        _KERNEL_CACHE[key] = kt
    return _KERNEL_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════════
# Decode kernel (shared-memory softmax to avoid 1D fragment layout conflicts)
# ═══════════════════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def _decode_kernel_func(batch, heads, heads_kv, dim, page_block_size,
                        max_blocks_per_seq, num_pages,
                        block_N=128, num_stages=0, threads=128):
    scale = (1.0 / dim) ** 0.5
    pts = block_N // page_block_size
    block_M = 16  # SM70 MMA minimum; row 0 = real Q, rows 1-15 = zero padding
    use_kv_union = dim > _USE_KV_UNION_FOR_DIM

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch, heads, dim], T.float16),
        Kc: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        Vc: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        bt: T.Tensor([batch, max_blocks_per_seq], T.int32),
        sl: T.Tensor([batch], T.int32),
        Out: T.Tensor([batch, heads, dim], T.float16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (bx, bz):
            Qs = T.alloc_shared([block_M, dim], T.float16)
            if use_kv_union:
                KVs = T.alloc_shared([block_N, dim], T.float16)
                K_ref = KVs
                V_ref = KVs
            else:
                K_ref = T.alloc_shared([block_N, dim], T.float16)
                V_ref = T.alloc_shared([block_N, dim], T.float16)
            Ps = T.alloc_shared([block_M, block_N], T.float16)

            As = T.alloc_fragment([block_M, block_N], T.float32)
            Ac = T.alloc_fragment([block_M, block_N], T.float16)
            Ao = T.alloc_fragment([block_M, dim], T.float32)

            mi = T.alloc_fragment([block_M], T.float32)
            mp = T.alloc_fragment([block_M], T.float32)
            sf = T.alloc_fragment([block_M], T.float32)
            rs = T.alloc_fragment([block_M], T.float32)

            # Shared memory for l_i — bypasses 1D fragment layout conflict with mi
            li = T.alloc_shared([block_M], T.float32)

            kvh = bx // (heads // heads_kv)

            T.copy(Q[bz, bx, :], Qs[0, :])
            T.fill(Ao, 0)
            T.fill(mi, -T.infinity(T.float32))
            T.fill(li, 0)

            for k in T.Pipelined(T.ceildiv(sl[bz], block_N), num_stages=num_stages):
                T.clear(K_ref)
                for p in T.serial(pts):
                    lp = T.floordiv(k * block_N, page_block_size) + p
                    if lp < max_blocks_per_seq:
                        ph = bt[bz, lp]
                        po = p * page_block_size
                        for i, j in T.Parallel(page_block_size, dim):
                            K_ref[po + i, j] = Kc[ph, i, kvh, j]

                for i, j in T.Parallel(block_M, block_N):
                    As[i, j] = T.if_then_else(k * block_N + j < sl[bz], T.cast(0, T.float32), -T.infinity(T.float32))
                T.gemm(Qs, K_ref, As, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(mi, mp)
                T.reduce_max(As, mi, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    mi[i] = T.if_then_else(mi[i] == -T.infinity(T.float32), T.cast(0, T.float32), mi[i])
                for i in T.Parallel(block_M):
                    mi[i] = T.max(mi[i], mp[i])
                for i in T.Parallel(block_M):
                    sf[i] = T.exp(mp[i] * scale - mi[i] * scale)
                    li[i] *= sf[i]
                for i, j in T.Parallel(block_M, dim):
                    Ao[i, j] *= sf[i]
                for i, j in T.Parallel(block_M, block_N):
                    As[i, j] = T.exp(As[i, j] * scale - mi[i] * scale)
                T.reduce_sum(As, rs, dim=1)
                for i in T.Parallel(block_M):
                    li[i] += rs[i]

                T.clear(V_ref)
                for p in T.serial(pts):
                    lp2 = T.floordiv(k * block_N, page_block_size) + p
                    if lp2 < max_blocks_per_seq:
                        ph2 = bt[bz, lp2]
                        po2 = p * page_block_size
                        for i, j in T.Parallel(page_block_size, dim):
                            V_ref[po2 + i, j] = Vc[ph2, i, kvh, j]

                for i, j in T.Parallel(block_M, block_N):
                    Ps[i, j] = T.cast(As[i, j], T.float16)
                T.copy(Ps, Ac)
                T.gemm(Ac, V_ref, Ao, policy=T.GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                if i == 0:
                    Out[bz, bx, j] = T.cast(Ao[i, j] / li[i], T.float16)

    return kernel


_DECODE_CACHE = {}

_DECODE_BEST_CONFIGS = {
    64:  dict(block_N=128, threads=128, num_stages=0),
    128: dict(block_N=128, threads=128, num_stages=0),
    256: dict(block_N=64,  threads=128, num_stages=0),
}


def get_decode_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                      max_blocks):
    cfg = _DECODE_BEST_CONFIGS.get(dim, dict(block_N=128, threads=128, num_stages=0))
    key = (heads, heads_kv, dim, block_size, cfg["block_N"], cfg["threads"], cfg["num_stages"],
           batch, max_blocks, num_pages)
    if key not in _DECODE_CACHE:
        kt = _decode_kernel_func(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            **cfg,
        )
        _DECODE_CACHE[key] = kt
    return _DECODE_CACHE[key]
