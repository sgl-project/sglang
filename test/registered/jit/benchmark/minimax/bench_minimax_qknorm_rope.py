"""Benchmark: fused MiniMax-M3 Gemma-RMSNorm + partial NeoX RoPE (1 in-place
launch) vs the unfused path (GemmaRMSNorm(q) + GemmaRMSNorm(k) + rotary_emb,
3 launches + intermediates). Main attention branch, per-rank TP8 shape (nq=8, nk=1).
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.minimax_qknorm_rope import (
    minimax_qknorm_rope,
    minimax_qknorm_rope_grouped,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=6, suite="jit-kernel-unit-test-amd")

HEAD_DIM, ROTARY_DIM, BASE, EPS, MAXPOS = 128, 64, 5_000_000, 1e-6, 131072
NQ, NK = 64, 4


def _cache():
    inv_freq = 1.0 / (
        BASE
        ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float, device="cuda")
            / ROTARY_DIM
        )
    )
    t = torch.arange(MAXPOS, dtype=torch.float, device="cuda")
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _unfused(qkv, wq, wk, cache, positions):
    # GemmaRMSNorm (1+w) on q,k head-wise + partial neox rope, in plain torch
    # (representative of the separate norm + rope launches).
    T = qkv.shape[0]
    q, k, v = qkv.split([NQ * HEAD_DIM, NK * HEAD_DIM, NK * HEAD_DIM], dim=-1)

    def norm(x, w, nh):
        x = x.reshape(T, nh, HEAD_DIM).float()
        var = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + EPS) * (1.0 + w.float())).to(torch.bfloat16)

    qn, kn = norm(q, wq, NQ), norm(k, wk, NK)
    cs = cache.index_select(0, positions).float()
    cos, sin = cs[:, None, :32], cs[:, None, 32:]

    def rope(x):
        x1, x2 = x[..., :32].float(), x[..., 32:64].float()
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2, x[..., 64:].float()], dim=-1).to(torch.bfloat16)

    return rope(qn), rope(kn)


def _fused(qkv, wq, wk, cache, positions):
    minimax_qknorm_rope(qkv, wq, wk, cache, positions, NQ, NK, NK, EPS)
    return qkv


FN_MAP = {"fused": _fused, "unfused_torch": _unfused}


@marker.parametrize("T", [1, 16, 64, 256, 1024, 8192], [64, 1024])
@marker.benchmark("impl", ["fused", "unfused_torch"])
def benchmark(T: int, impl: str):
    cache = _cache()
    wq = (torch.randn(HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16)
    wk = (torch.randn(HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16)
    qkv = torch.randn(T, (NQ + 2 * NK) * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    positions = torch.randint(0, MAXPOS, (T,), device="cuda", dtype=torch.int64)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(qkv, wq, wk, cache, positions),
        graph_clone_args=(0,),
        memory_args=None,
    )


# --- Combined main + index single launch (the fused qkv+index_qkv GEMM path) ---
# Per-rank TP8 sparse shape: main q=8/k=1/v=1 + index idx_q=1/idx_k=1 (value
# disabled). One grouped launch (q, k, idx_q, idx_k) vs two separate launches.
C_NQ, C_NKV, C_NIQ = 8, 1, 1
C_OFF_Q = 0
C_OFF_K = C_NQ
C_OFF_V = C_NQ + C_NKV
C_OFF_IQ = C_NQ + 2 * C_NKV
C_OFF_IK = C_OFF_IQ + C_NIQ
C_TOTAL_HEADS = C_OFF_IK + 1


def _combined_one(args):
    qkv, wq, wk, wiq, wik, cache, positions = args
    minimax_qknorm_rope_grouped(
        qkv,
        [
            (wq, C_OFF_Q, C_NQ),
            (wk, C_OFF_K, C_NKV),
            (wiq, C_OFF_IQ, C_NIQ),
            (wik, C_OFF_IK, 1),
        ],
        cache,
        positions,
        EPS,
    )
    return qkv


def _combined_two(args):
    # Two launches over the same buffer: main (q,k) then index (idx_q, idx_k).
    qkv, wq, wk, wiq, wik, cache, positions = args
    minimax_qknorm_rope_grouped(
        qkv, [(wq, C_OFF_Q, C_NQ), (wk, C_OFF_K, C_NKV)], cache, positions, EPS
    )
    minimax_qknorm_rope_grouped(
        qkv, [(wiq, C_OFF_IQ, C_NIQ), (wik, C_OFF_IK, 1)], cache, positions, EPS
    )
    return qkv


C_FN_MAP = {"combined_one_launch": _combined_one, "two_launches": _combined_two}


@marker.parametrize("T", [1, 16, 64, 256, 1024, 8192], [64, 1024])
@marker.benchmark("impl", ["combined_one_launch", "two_launches"])
def benchmark_combined(T: int, impl: str):
    cache = _cache()
    ws = [
        (torch.randn(HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16)
        for _ in range(4)
    ]
    qkv = torch.randn(T, C_TOTAL_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    positions = torch.randint(0, MAXPOS, (T,), device="cuda", dtype=torch.int64)
    return marker.do_bench(
        C_FN_MAP[impl],
        input_args=((qkv, *ws, cache, positions),),
        graph_clone_args=(0,),
        memory_args=None,
    )


if __name__ == "__main__":
    benchmark.run()
    benchmark_combined.run()
