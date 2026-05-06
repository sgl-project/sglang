"""K=64 prefill stage 4: persistent kernel sweep.

Reference: K=64 currently goes through ``_block_sparse_mqa_grouped_kernel``
(GEMM_TILE=256, GROUP_SIZE=4, no persistent). The orchestrator comment
hints "K=64: K_CHUNKS=32 num_warps=4 wins 1.31× (different config; TODO)"
— this script verifies and finds the production config.

Persistent kernel uses GEMM_TILE = K * GROUP_SIZE. For K=64:
  - GROUP_SIZE=4 → GEMM_TILE=256 (same tile shape as current grouped path)
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    _block_sparse_mqa_persistent_kernel,
    block_sparse_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa.tests.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, PREFILL_CHUNK, make_prefill_inputs,
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


def random_topk(K, sq, cu_ke, block_topk, seed=2):
    torch.manual_seed(seed)
    ke_blocks = ((cu_ke + K - 1) // K).long()
    n_blocks_max = int(ke_blocks.max().item())
    idx = torch.randint(
        0, n_blocks_max, (sq, block_topk),
        device=cu_ke.device, dtype=torch.int64,
    )
    idx = torch.minimum(idx, (ke_blocks - 1).clamp_min(0).unsqueeze(1))
    return idx


def call_persistent_k64(q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke,
                        K_CHUNKS, num_warps, num_stages):
    seq_len, H, D = q.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_idx.shape[-1])
    GROUP_SIZE = 4  # K=64 → GEMM_TILE=256
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    outer = (num_chunks + K_CHUNKS - 1) // K_CHUNKS
    logits = torch.empty(
        (seq_len, topk * 64), device=q.device, dtype=torch.float32,
    )
    grid = (seq_len, outer)
    _block_sparse_mqa_persistent_kernel[grid](
        q, k_fp8, k_scale, topk_idx, logits, w,
        cu_ks, cu_ke,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        topk_idx.stride(0), topk_idx.stride(1),
        logits.stride(0), logits.stride(1),
        w.stride(0), w.stride(1),
        seq_kv,
        topk,
        HEADS=H, DIM=D,
        KV_BLOCK_SIZE=64,
        GROUP_SIZE=GROUP_SIZE,
        K_CHUNKS=K_CHUNKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits


def main():
    K = 64
    block_topk = BLOCK_TOPK_FORMULA // K  # 128
    sq = PREFILL_CHUNK
    print("=" * 110)
    print(f"K={K} prefill stage 4 — persistent kernel sweep "
          f"(sq={sq}, block_topk={block_topk}, GROUP_SIZE=4 → GEMM_TILE=256)")
    print("=" * 110)

    skv_list = [(16, "16K"), (32, "32K"), (65, "65K"), (128, "128K")]

    # block_topk=128, GROUP_SIZE=4 → num_chunks=32 grouped chunks/seq.
    # K_CHUNKS sweep: 4, 8, 16, 32 (full persistent at 32).
    chunks_list = [4, 8, 16, 32]
    warps_list = [2, 4, 8]
    stages_list = [2, 3, 4]

    inputs_cache = {}
    for skv_k, label in skv_list:
        skv = skv_k * 1024
        q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
        topk_idx = random_topk(K, sq, cu_ke, block_topk)
        inputs_cache[label] = (q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke)

    # Current production baseline: block_sparse_mqa_triton (grouped, no persistent).
    print(f"{'skv':>5} | {'baseline grouped (μs)':>22}")
    print("-" * 40)
    base = {}
    for skv_k, label in skv_list:
        q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke = inputs_cache[label]
        t = bench_eager(lambda: block_sparse_mqa_triton(
            q_fp8=q, k_fp8=k_fp8, k_scale=k_scale,
            topk_block_index=topk_idx, kv_block_size=K, weights=w,
            cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        ))
        base[label] = t
        print(f"{label:>5} | {t:>22.1f}")

    print()
    print(f"{'K_CHUNKS':>9} {'warps':>6} {'stages':>7} | "
          + " ".join([f"{lbl:>10}" for _, lbl in skv_list])
          + " | speedup vs grouped")
    print("-" * 110)

    best = {label: (float("inf"), None) for _, label in skv_list}
    for kc in chunks_list:
        for nw in warps_list:
            for ns in stages_list:
                row = []
                ok = True
                for skv_k, label in skv_list:
                    inp = inputs_cache[label]
                    try:
                        # Warmup compile.
                        call_persistent_k64(*inp, kc, nw, ns)
                        t = bench_eager(lambda: call_persistent_k64(*inp, kc, nw, ns))
                    except Exception as exc:
                        ok = False
                        row.append("X")
                        continue
                    row.append(f"{t:.1f}")
                    if t < best[label][0]:
                        best[label] = (t, (kc, nw, ns))
                if not ok:
                    continue
                cells = " ".join(f"{c:>10}" for c in row)
                ratios = " ".join(
                    f"{label}={base[label]/float(c):.2f}x"
                    for (_, label), c in zip(skv_list, row)
                    if c != "X"
                )
                print(f"{kc:>9} {nw:>6} {ns:>7} | {cells} | {ratios}")

    print()
    print("Best persistent config per skv (vs current grouped):")
    for skv_k, label in skv_list:
        t, cfg = best[label]
        if cfg is None:
            print(f"  {label}: no valid config")
            continue
        ratio = base[label] / t
        print(f"  {label}: {t:.1f} μs  cfg=(K_CHUNKS={cfg[0]}, warps={cfg[1]}, "
              f"stages={cfg[2]})  vs grouped {base[label]:.1f} → {ratio:.2f}x")


if __name__ == "__main__":
    main()
