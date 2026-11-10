# SPDX-License-Identifier: Apache-2.0
# Adapted from GooseLLM (vLLM fork) tilelang chunk_scaled_dot_kkt_fwd for V100/SM70.
# Original: vllm/model_executor/layers/fla/ops/tilelang/_kkt.py (ai-bond / GooseLLM)
#
# Tilelang-accelerated K·Kᵀ inner-product stage of the chunked GatedDeltaNet
# (linear attention) prefill. This is the hottest sub-op of the GDN chunk
# algorithm and the one GooseLLM tunes for SM70. Produces the raw (strictly
# lower-triangular, gated, beta-scaled) KKT matrix A; call solve_tril afterwards
# to obtain (I+A)^-1.

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


# ---------------------------------------------------------------------------
# Fixed-length factory: k is [B, S, Hg, Kdim], S % BT == 0.
# out_idx=[3] => the 4th param (A) is the kernel output.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def _kkt_factory(B, S, H, Hg, Kdim, BT):
    BK = 16
    threads = 128
    NT = S // BT

    @T.prim_func
    def main(
        k: T.Tensor([B, S, Hg, Kdim], T.float16),
        beta: T.Tensor([B, S, H], T.float16),
        g: T.Tensor([B, S, H], T.float16),
        A: T.Tensor([B, S, H, BT], T.float32),
    ):
        with T.Kernel(NT, H, B, threads=threads) as (bx, by, bz):
            sh = T.alloc_shared([BT, BK], T.float16)
            sc = T.alloc_shared([BT, BK], T.float16)
            bs = T.alloc_shared([BT], T.float16)
            gs = T.alloc_shared([BT], T.float16)
            ac = T.alloc_fragment([BT, BT], T.float32)
            T.fill(ac, 0)

            cs = bx * BT
            for i in T.Parallel(BT):
                bs[i] = T.cast(beta[bz, cs + i, by], T.float16)
                gs[i] = T.cast(g[bz, cs + i, by], T.float16)

            hk = by // (H // Hg) if Hg > 0 else 0

            for ik in T.Pipelined(T.ceildiv(Kdim, BK)):
                ko = ik * BK
                for i, j in T.Parallel(BT, BK):
                    kv = k[bz, cs + i, hk, ko + j]
                    sh[i, j] = kv * bs[i]
                    sc[i, j] = kv
                T.gemm(
                    sh, sc, ac, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )

            for i, j in T.Parallel(BT, BT):
                v = ac[i, j] * T.exp(gs[i] - gs[j])
                A[bz, cs + i, by, j] = (
                    T.cast(v, T.float32) if j < i else T.cast(0, T.float32)
                )
    return main


# ---------------------------------------------------------------------------
# Variable-length factory: k is [1, TT, Hg, Kdim] with cu_seqlens + chunk_indices.
# out_idx=[5] => the 6th param (A) is the kernel output.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def _kkt_factory_varlen(H, Hg, Kdim, BT):
    BK = 16
    threads = 128
    TT = T.dynamic("TT")
    NC = T.dynamic("NC")
    NT = T.dynamic("NT")

    @T.prim_func
    def main(
        k: T.Tensor([1, TT, Hg, Kdim], T.float16),
        beta: T.Tensor([1, TT, H], T.float16),
        g: T.Tensor([1, TT, H], T.float16),
        cu_seqlens: T.Tensor([NC], T.int32),
        chunk_indices: T.Tensor([NT, 2], T.int32),
        A: T.Tensor([1, TT, H, BT], T.float32),
    ):
        with T.Kernel(NT, H, 1, threads=threads) as (bx, by, bz):
            sh = T.alloc_shared([BT, BK], T.float16)
            sc = T.alloc_shared([BT, BK], T.float16)
            bs = T.alloc_shared([BT], T.float16)
            gs = T.alloc_shared([BT], T.float16)
            ac = T.alloc_fragment([BT, BT], T.float32)
            T.fill(ac, 0)

            i_n = chunk_indices[bx, 0]
            i_t = chunk_indices[bx, 1]
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            cs = bos + i_t * BT

            hk = by // (H // Hg) if Hg > 0 else 0

            for i in T.Parallel(BT):
                valid = cs + i < eos
                bs[i] = T.if_then_else(
                    valid, T.cast(beta[0, cs + i, by], T.float16), T.cast(0, T.float16)
                )
                gs[i] = T.if_then_else(
                    valid, T.cast(g[0, cs + i, by], T.float16), T.cast(-100, T.float16)
                )

            for ik in T.Pipelined(T.ceildiv(Kdim, BK)):
                ko = ik * BK
                for i, j in T.Parallel(BT, BK):
                    kv = T.if_then_else(
                        cs + i < eos, k[0, cs + i, hk, ko + j], T.cast(0, T.float16)
                    )
                    sh[i, j] = kv * bs[i]
                    sc[i, j] = kv
                T.gemm(
                    sh, sc, ac, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )

            for i, j in T.Parallel(BT, BT):
                v = ac[i, j] * T.exp(gs[i] - gs[j])
                if j < i and cs + i < eos:
                    A[0, cs + i, by, j] = T.cast(v, T.float32)
                elif cs + i < eos:
                    A[0, cs + i, by, j] = T.cast(0, T.float32)
    return main


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Tilelang KKT for V100/SM70. Drop-in for sglang's triton
    `chunk_scaled_dot_kkt_fwd` (same signature, same output A: [B,T,H,BT] fp32).

    Returns the raw strictly-lower-triangular KKT matrix; the caller must run
    `solve_tril` to obtain (I+A)^-1.
    """
    B, TT, Hg, Kd = k.shape
    H = beta.shape[-1]
    BT = chunk_size

    # Tilelang kernel is fp16 (V100 tensor cores); cast inputs.
    k = k.to(torch.float16)
    beta = beta.to(torch.float16)
    if g_cumsum is None:
        g_cumsum = torch.zeros(B, TT, H, dtype=torch.float16, device=k.device)
    else:
        g_cumsum = g_cumsum.to(torch.float16)

    if cu_seqlens is not None:
        from sglang.srt.layers.attention.fla.index import prepare_chunk_indices

        cu_seqlens_i32 = cu_seqlens.to(torch.int32)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT).to(torch.int32)
        kernel = _kkt_factory_varlen(H, Hg, Kd, BT)
        A = kernel(k, beta, g_cumsum, cu_seqlens_i32, chunk_indices)
        if output_dtype != torch.float32:
            A = A.to(output_dtype)
        return A

    assert TT % BT == 0, f"T={TT} must be divisible by BT={BT} for tilelang KKT"
    kernel = _kkt_factory(B, TT, H, Hg, Kd, BT)
    A = kernel(k, beta, g_cumsum)
    if output_dtype != torch.float32:
        A = A.to(output_dtype)
    return A
