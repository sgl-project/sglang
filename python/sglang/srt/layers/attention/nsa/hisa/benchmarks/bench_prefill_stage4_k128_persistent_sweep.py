"""K=128 prefill stage 4: try the persistent kernel (GEMM_TILE=128
GROUP_SIZE=1) at various (K_CHUNKS, num_warps, num_stages) to see if any
config beats tilelang at production shapes.

Reference: tilelang (current production for K=128) hit ~2440-2453 μs at
sq=8192 across skv∈{16K..128K} (per bench_prefill_stage4_k128_ab.py).
Triton legacy (BLOCK_N=128, no grouping, no persistent) was 2700+ μs
(11-14% slower). The persistent kernel reuses Q[H,D]+W[H] across K_CHUNKS
iters — at K=128 each iter fills GEMM_TILE=128 with one topk index.
"""
from __future__ import annotations

import torch
import triton

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    _block_sparse_mqa_persistent_kernel,
)
from sglang.srt.layers.attention.nsa.hisa.tests.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, D, H, PREFILL_CHUNK, make_prefill_inputs,
)


def bench_eager(fn, w=20, n=50):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / n


def random_topk(K, sq, cu_ke, block_topk):
    torch.manual_seed(2)
    ke_blocks = ((cu_ke + K - 1) // K).long()
    n_blocks_max = int(ke_blocks.max().item())
    idx = torch.randint(
        0, n_blocks_max, (sq, block_topk),
        device=cu_ke.device, dtype=torch.int64,
    )
    idx = torch.minimum(idx, (ke_blocks - 1).clamp_min(0).unsqueeze(1))
    return idx


def call_persistent(q_fp8, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke,
                    K, K_CHUNKS, num_warps, num_stages):
    seq_len, H_, D_ = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_idx.shape[-1])
    GROUP_SIZE = 1  # K=128 fills GEMM_TILE entirely
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    outer = (num_chunks + K_CHUNKS - 1) // K_CHUNKS
    logits = torch.empty(
        (seq_len, topk * K), device=q_fp8.device, dtype=torch.float32,
    )
    grid = (seq_len, outer)
    # The persistent kernel hardcodes num_stages=2 in its tl.range. We can
    # still vary num_warps via the launch config. To experiment with
    # num_stages we'd need to clone the kernel — skip for v0.
    _block_sparse_mqa_persistent_kernel[grid](
        q_fp8, k_fp8, k_scale, topk_idx, logits, w,
        cu_ks, cu_ke,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        topk_idx.stride(0), topk_idx.stride(1),
        logits.stride(0), logits.stride(1),
        w.stride(0), w.stride(1),
        seq_kv,
        topk,
        HEADS=H_, DIM=D_,
        KV_BLOCK_SIZE=K,
        GROUP_SIZE=GROUP_SIZE,
        K_CHUNKS=K_CHUNKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits


def main():
    K = 128
    block_topk = BLOCK_TOPK_FORMULA // K  # 64
    sq = PREFILL_CHUNK
    print("=" * 110)
    print(f"K=128 prefill stage 4 — persistent kernel sweep "
          f"(sq={sq}, block_topk={block_topk})")
    print("=" * 110)

    skv_list = [(16, "16K"), (32, "32K"), (65, "65K"), (128, "128K")]
    # K_CHUNKS sweep: 64 / K_CHUNKS = num outer steps. Try 4, 8, 16, 32, 64.
    chunks_list = [4, 8, 16, 32, 64]
    warps_list = [2, 4, 8]
    stages_list = [2, 3, 4]

    # Cache inputs per skv to avoid re-allocation in the inner sweep.
    inputs_cache = {}
    for skv_k, label in skv_list:
        skv = skv_k * 1024
        q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
        topk_idx = random_topk(K, sq, cu_ke, block_topk)
        inputs_cache[label] = (q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke)

    # Tilelang baseline.
    print(f"{'skv':>5} | {'tilelang (μs)':>14}")
    print("-" * 30)
    tile_baseline = {}
    for skv_k, label in skv_list:
        q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke = inputs_cache[label]
        t = bench_eager(lambda: fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q, k=k_fp8, k_scale=k_scale, topk_block_index=topk_idx,
            kv_block_size=K, weights=w,
            cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        ))
        tile_baseline[label] = t
        print(f"{label:>5} | {t:>14.1f}")

    # Persistent sweep.
    print()
    print(f"{'K_CHUNKS':>9} {'warps':>6} {'stages':>7} | "
          + " ".join([f"{lbl:>10}" for _, lbl in skv_list])
          + " | best vs tile")
    print("-" * 110)

    best = {label: (float("inf"), None) for _, label in skv_list}
    for kc in chunks_list:
        for nw in warps_list:
            for ns in stages_list:
                row = []
                row_ratios = []
                ok = True
                for skv_k, label in skv_list:
                    inp = inputs_cache[label]
                    try:
                        # Warmup once to compile.
                        call_persistent(*inp[:7] if False else
                                        [inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6]],
                                        K, kc, nw, ns)
                        t = bench_eager(lambda: call_persistent(
                            inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6],
                            K, kc, nw, ns,
                        ))
                    except Exception as exc:
                        ok = False
                        row.append("X")
                        continue
                    row.append(f"{t:.1f}")
                    row_ratios.append((label, t))
                    if t < best[label][0]:
                        best[label] = (t, (kc, nw, ns))
                if not ok:
                    continue
                cells = " ".join(f"{c:>10}" for c in row)
                ratio_str = " ".join(
                    f"{label}={tile_baseline[label]/t:.2f}x"
                    for (label, t) in row_ratios
                )
                print(f"{kc:>9} {nw:>6} {ns:>7} | {cells} | {ratio_str}")

    print()
    print("Best config per skv:")
    for skv_k, label in skv_list:
        t, cfg = best[label]
        if cfg is None:
            print(f"  {label}: no valid config")
            continue
        ratio = tile_baseline[label] / t
        print(f"  {label}: {t:.1f} μs  cfg=(K_CHUNKS={cfg[0]}, "
              f"warps={cfg[1]}, stages={cfg[2]})  vs tile {tile_baseline[label]:.1f} → "
              f"{ratio:.2f}x")


if __name__ == "__main__":
    main()
