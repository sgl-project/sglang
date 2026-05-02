"""K=16, K=32 persistent resweep at current production block_topk
(8192//K = 512, 256). Existing in-tree config (K_CHUNKS=16, w=8) was
chosen for the old block_topk=2048 era; need to verify under prod.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    _block_sparse_mqa_persistent_kernel,
    block_sparse_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
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


def call_persistent(K, q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke,
                    K_CHUNKS, num_warps, num_stages):
    seq_len, H, D = q.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_idx.shape[-1])
    GROUP_SIZE = 256 // K  # GEMM_TILE=256 for K∈{16,32,64}
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    outer = (num_chunks + K_CHUNKS - 1) // K_CHUNKS
    logits = torch.empty(
        (seq_len, topk * K), device=q.device, dtype=torch.float32,
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
        KV_BLOCK_SIZE=K,
        GROUP_SIZE=GROUP_SIZE,
        K_CHUNKS=K_CHUNKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits


def main():
    sq = PREFILL_CHUNK
    skv_list = [(16, "16K"), (32, "32K"), (65, "65K"), (128, "128K")]
    chunks_list = [4, 8, 16, 32]
    warps_list = [4, 8]
    stages_list = [2, 3, 4]

    for K in (16, 32):
        block_topk = BLOCK_TOPK_FORMULA // K  # 512 / 256
        GROUP_SIZE = 256 // K                  # 16 / 8
        num_chunks_per_seq = block_topk // GROUP_SIZE  # 32 for both
        print()
        print("=" * 110)
        print(f"K={K} prefill stage 4 — persistent resweep "
              f"(sq={sq}, block_topk={block_topk}, GROUP_SIZE={GROUP_SIZE}, "
              f"num_chunks/seq={num_chunks_per_seq})")
        print("=" * 110)

        inputs_cache = {}
        for skv_k, label in skv_list:
            skv = skv_k * 1024
            q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
            topk_idx = random_topk(K, sq, cu_ke, block_topk)
            inputs_cache[label] = (q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke)

        # Current in-tree (uses block_sparse_mqa_triton, which already routes
        # K∈{16,32} to persistent K_CHUNKS=16 num_warps=8).
        print(f"{'skv':>5} | {'in-tree (μs)':>14}")
        print("-" * 30)
        base = {}
        for skv_k, label in skv_list:
            inp = inputs_cache[label]
            t = bench_eager(lambda: block_sparse_mqa_triton(
                q_fp8=inp[0], k_fp8=inp[1], k_scale=inp[2],
                topk_block_index=inp[3], kv_block_size=K, weights=inp[4],
                cu_seqlen_ks=inp[5], cu_seqlen_ke=inp[6],
            ))
            base[label] = t
            print(f"{label:>5} | {t:>14.1f}")

        print()
        print(f"{'K_CHUNKS':>9} {'warps':>6} {'stages':>7} | "
              + " ".join([f"{lbl:>10}" for _, lbl in skv_list])
              + " | speedup vs in-tree")
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
                            call_persistent(K, *inp, kc, nw, ns)
                            t = bench_eager(lambda: call_persistent(K, *inp, kc, nw, ns))
                        except Exception:
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
                        for (_, label), c in zip(skv_list, row) if c != "X"
                    )
                    print(f"{kc:>9} {nw:>6} {ns:>7} | {cells} | {ratios}")

        print()
        print(f"K={K} best:")
        for skv_k, label in skv_list:
            t, cfg = best[label]
            ratio = base[label] / t
            print(f"  {label}: {t:.1f} μs  cfg=(K_CHUNKS={cfg[0]}, "
                  f"warps={cfg[1]}, stages={cfg[2]})  vs in-tree {base[label]:.1f} → "
                  f"{ratio:.2f}x")

        inputs_cache.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
