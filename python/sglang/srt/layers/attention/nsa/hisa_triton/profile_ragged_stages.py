"""Per-stage profiler for the ragged prefill orchestrator (K=128).

Decomposes both the triton and tilelang versions into 4 sub-stages and
times each independently to attribute the e2e gap to specific kernels.

Stages (identical inputs/outputs in both paths):
  1. mean-pool ragged K → [num_pool, D] fp8 + [num_pool] f32 scale
  2. block-MQA on pooled K → [seq_q, num_pool] f32
  3. torch.topk on bf16 cast → [seq_q, block_topk] i64
  4. ragged sparse-MQA → [seq_q, block_topk * k_block_size] f32

Output: per-(sq, skv) breakdown with absolute μs per stage and the
triton/tilelang ratio. Plus a "winner" attribution per stage.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m \\
        sglang.srt.layers.attention.nsa.hisa_triton.profile_ragged_stages
"""
from __future__ import annotations

import statistics
import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_mean_pooling_interface,
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
    fp8_native_hierarchy_mqa_logits,
    pool_mqa_attn_return_logits_fp8_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    block_mean_pooling_triton,
    block_sparse_mqa_triton,
    ragged_pool_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.orchestrator import (
    fp8_native_hierarchy_mqa_logits_triton,
)


DEVICE = torch.device("cuda")
H, D = 64, 128
K = 128
BLOCK_TOPK = 64                 # K=128 production formula: 8192 // K = 64


def _flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int, device=DEVICE).zero_()


def cuda_bench(fn, warmups: int = 5, iters: int = 30) -> float:
    torch.cuda.synchronize()
    for _ in range(warmups): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        _flush_l2()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


# --------------------------------------------------------------------------
def make_inputs(seq_q: int, seq_kv: int, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(seq_q, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(seq_kv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(seq_kv // 4, seq_kv, seq_q, device=DEVICE).to(torch.int32)
    return q, k_fp8, k_scale, weights, cu_ks, cu_ke


def profile_one_config(seq_q: int, seq_kv: int) -> list[tuple[str, float, float]]:
    """Run all 4 stages of triton and tilelang ragged orchestrator at given
    (sq, skv); return [(stage_name, t_triton_ms, t_tilelang_ms)]."""
    q, k_fp8, k_scale, weights, cu_ks, cu_ke = make_inputs(seq_q, seq_kv)
    cu_blocked_ks = (cu_ks // K).to(torch.int32)
    cu_blocked_ke = ((cu_ke + K - 1) // K).to(torch.int32)

    # --- Stage 1: mean-pool. Both paths produce blocked_k_fp8 + scale. ---
    # Sanity: pre-run both to allocate / warmup
    bk_t, bks_t = block_mean_pooling_triton(k_fp8=k_fp8, k_scale=k_scale, k_block_size=K)
    bk_l, bks_l = fp8_native_block_mean_pooling_interface(k_fp8, k_scale, K)
    torch.cuda.synchronize()

    t1_triton = cuda_bench(lambda: block_mean_pooling_triton(
        k_fp8=k_fp8, k_scale=k_scale, k_block_size=K,
    ))
    t1_tilelang = cuda_bench(lambda: fp8_native_block_mean_pooling_interface(
        k_fp8, k_scale, K,
    ))

    # --- Stage 2: block-MQA on pooled K. Uses bk_t / bk_l from stage 1. ---
    score_t = ragged_pool_mqa_triton(
        q_fp8=q, blocked_k_fp8=bk_t, blocked_k_scale=bks_t,
        weights=weights,
        cu_seqlen_ks=cu_blocked_ks, cu_seqlen_ke=cu_blocked_ke,
    )
    score_l = pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q, blocked_kv_fp8=bk_l, blocked_kv_scale=bks_l,
        kv_block_size=K, weights_f32=weights,
        cu_seqlen_blocked_ks=cu_blocked_ks, cu_seqlen_blocked_ke=cu_blocked_ke,
    )
    torch.cuda.synchronize()

    t2_triton = cuda_bench(lambda: ragged_pool_mqa_triton(
        q_fp8=q, blocked_k_fp8=bk_t, blocked_k_scale=bks_t, weights=weights,
        cu_seqlen_ks=cu_blocked_ks, cu_seqlen_ke=cu_blocked_ke,
    ))
    t2_tilelang = cuda_bench(lambda: pool_mqa_attn_return_logits_fp8_interface(
        q_fp8=q, blocked_kv_fp8=bk_l, blocked_kv_scale=bks_l,
        kv_block_size=K, weights_f32=weights,
        cu_seqlen_blocked_ks=cu_blocked_ks, cu_seqlen_blocked_ke=cu_blocked_ke,
    ))

    # --- Stage 3: torch.topk. Both paths use the same call (identical) ---
    topk_actual = min(BLOCK_TOPK, score_t.shape[-1])
    t3_triton = cuda_bench(lambda: torch.topk(
        score_t.bfloat16(), k=topk_actual, dim=-1, sorted=False,
    ).indices)
    t3_tilelang = t3_triton  # same kernel; report once

    topk_idx_t = torch.topk(score_t.bfloat16(), k=topk_actual, dim=-1, sorted=False).indices
    topk_idx_l = torch.topk(score_l.bfloat16(), k=topk_actual, dim=-1, sorted=False).indices

    # --- Stage 4: ragged sparse-MQA. ---
    # block_sparse_mqa_triton's grouped path requires topk % (256/K) == 0;
    # K=128 → group=1, no padding needed.
    out_t = block_sparse_mqa_triton(
        q_fp8=q, k_fp8=k_fp8, k_scale=k_scale, topk_block_index=topk_idx_t,
        kv_block_size=K, weights=weights, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    out_l = fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q=q, k=k_fp8, k_scale=k_scale, topk_block_index=topk_idx_l,
        kv_block_size=K, weights=weights, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    torch.cuda.synchronize()

    t4_triton = cuda_bench(lambda: block_sparse_mqa_triton(
        q_fp8=q, k_fp8=k_fp8, k_scale=k_scale, topk_block_index=topk_idx_t,
        kv_block_size=K, weights=weights, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    ))
    t4_tilelang = cuda_bench(lambda: fp8_native_block_sparse_mqa_attn_return_logits_interface(
        q=q, k=k_fp8, k_scale=k_scale, topk_block_index=topk_idx_l,
        kv_block_size=K, weights=weights, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    ))

    # --- E2E: run the full orchestrators ---
    # Match the production input shape: kv = (k_fp8, k_scale_uint8 [N,4]).
    k_scale_u8 = k_scale.view(torch.uint8).reshape(seq_kv, 4)
    t_e2e_triton = cuda_bench(lambda: fp8_native_hierarchy_mqa_logits_triton(
        q, (k_fp8, k_scale_u8), weights, cu_ks, cu_ke,
        k_block_size=K, block_topk=BLOCK_TOPK,
    ))
    t_e2e_tilelang = cuda_bench(lambda: fp8_native_hierarchy_mqa_logits(
        q, (k_fp8, k_scale_u8), weights, cu_ks, cu_ke,
        k_block_size=K, block_topk=BLOCK_TOPK,
    ))

    return [
        ("1.mean-pool",   t1_triton, t1_tilelang),
        ("2.block-MQA",   t2_triton, t2_tilelang),
        ("3.torch.topk",  t3_triton, t3_tilelang),
        ("4.sparse-MQA",  t4_triton, t4_tilelang),
        ("E2E orch.",     t_e2e_triton, t_e2e_tilelang),
    ]


def main():
    CONFIGS = [
        (32, 4096), (128, 8192), (256, 8192), (512, 32768),
        (1024, 65536), (2048, 65536), (4096, 65536), (8192, 65536),
    ]

    all_data = []   # list of (cfg_label, [(stage, t_t, t_l), ...])
    for sq, skv in CONFIGS:
        try:
            data = profile_one_config(sq, skv)
            all_data.append((f"sq={sq:>4} skv={skv:>6}", data))
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at sq={sq} skv={skv}")
            break

    # Per-stage breakdown table (4 stages + e2e)
    print("\n" + "=" * 100)
    print(f"{'config':<22} {'stage':<14} {'triton(μs)':>12} {'tilelang(μs)':>14} {'ratio':>8} {'Δ μs':>10}")
    print("-" * 100)
    for cfg_label, data in all_data:
        # data[0:4] = 4 stages, data[4] = e2e
        sum_t = sum(t for _, t, _ in data[:4]) * 1000
        sum_l = sum(l for _, _, l in data[:4]) * 1000
        e2e_t = data[4][1] * 1000
        e2e_l = data[4][2] * 1000

        for stage, t, l in data[:4]:
            t_us = t * 1000; l_us = l * 1000
            delta = t_us - l_us
            ratio = t_us / l_us if l_us > 0 else float("nan")
            marker = "★" if delta < -1.0 else ("⚠" if delta > 5.0 else "")
            print(f"{cfg_label:<22} {stage:<14} {t_us:>12.2f} {l_us:>14.2f} {ratio:>7.2f}x {delta:>+9.2f}  {marker}")
        print(f"{cfg_label:<22} {'SUM-of-stages':<14} {sum_t:>12.2f} {sum_l:>14.2f} {sum_t/sum_l:>7.2f}x {sum_t-sum_l:>+9.2f}")
        print(f"{cfg_label:<22} {'E2E orch.':<14} {e2e_t:>12.2f} {e2e_l:>14.2f} {e2e_t/e2e_l:>7.2f}x {e2e_t-e2e_l:>+9.2f}")
        # Orchestrator overhead = e2e − sum
        oh_t = e2e_t - sum_t
        oh_l = e2e_l - sum_l
        print(f"{cfg_label:<22} {'orch.overhead':<14} {oh_t:>12.2f} {oh_l:>14.2f}    {'':>7}  {oh_t-oh_l:>+9.2f}")
        print("-" * 100)

    # Summary
    print("\n" + "=" * 95)
    print("Summary: per-stage Δ + orchestrator overhead Δ")
    print("=" * 95)
    print(f"{'config':<22} {'Δs1':>8} {'Δs2':>8} {'Δs3':>8} {'Δs4':>8} {'Δsum':>9} {'Δe2e':>9} {'Δoh':>9}")
    for cfg_label, data in all_data:
        deltas = [(t - l) * 1000 for _, t, l in data[:4]]
        sum_d = sum(deltas)
        e2e_d = (data[4][1] - data[4][2]) * 1000
        oh_d = e2e_d - sum_d
        print(f"{cfg_label:<22} {deltas[0]:>+8.1f} {deltas[1]:>+8.1f} {deltas[2]:>+8.1f} {deltas[3]:>+8.1f} {sum_d:>+9.2f} {e2e_d:>+9.2f} {oh_d:>+9.2f}")


if __name__ == "__main__":
    main()
