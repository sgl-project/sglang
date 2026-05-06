"""K=8 prefill stage 4 persistent sweep at production block_topk=1024.

Old comment said "K=8 best persistent only +5%; K_CHUNKS=16/w=8 regresses
18%" — but that was at block_topk=2048. Production now uses block_topk =
8192//K = 1024 for K=8.
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


def call_persistent_k8(q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke,
                       K_CHUNKS, num_warps, num_stages, GROUP_SIZE=32):
    seq_len, H, D = q.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_idx.shape[-1])
    K = 8
    GEMM_TILE = K * GROUP_SIZE  # 256 default
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
    K = 8
    block_topk = BLOCK_TOPK_FORMULA // K  # 1024
    sq = PREFILL_CHUNK
    print("=" * 110)
    print(f"K={K} prefill stage 4 — persistent sweep "
          f"(sq={sq}, block_topk={block_topk})")
    print("=" * 110)

    skv_list = [(16, "16K"), (32, "32K"), (65, "65K"), (128, "128K")]

    # GROUP_SIZE=32 → GEMM_TILE=256, num_chunks per seq = 1024/32 = 32.
    # Also try GROUP_SIZE=16 (GEMM_TILE=128) and GROUP_SIZE=64 (GEMM_TILE=512).
    sweeps = []
    for G in (16, 32, 64):
        # K_CHUNKS values aligned with num_chunks = block_topk // G.
        ncc = block_topk // G
        for kc in (max(1, ncc // 4), max(1, ncc // 2), ncc):
            for nw in (4, 8):
                for ns in (2, 3):
                    sweeps.append((G, kc, nw, ns))

    inputs_cache = {}
    for skv_k, label in skv_list:
        skv = skv_k * 1024
        q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
        topk_idx = random_topk(K, sq, cu_ke, block_topk)
        inputs_cache[label] = (q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke)

    # Current in-tree (K=8 falls into the K=8 grouped fallback after my edit;
    # before my edit it went through grouped at GEMM_TILE=256 G=32).
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
    print(f"{'GROUP':>5} {'K_CHUNKS':>9} {'warps':>6} {'stages':>7} | "
          + " ".join([f"{lbl:>10}" for _, lbl in skv_list])
          + " | speedup vs in-tree")
    print("-" * 110)

    best = {label: (float("inf"), None) for _, label in skv_list}
    for G, kc, nw, ns in sweeps:
        row = []
        ok = True
        for skv_k, label in skv_list:
            inp = inputs_cache[label]
            try:
                call_persistent_k8(*inp, kc, nw, ns, GROUP_SIZE=G)
                t = bench_eager(lambda: call_persistent_k8(*inp, kc, nw, ns, GROUP_SIZE=G))
            except Exception:
                ok = False
                row.append("X")
                continue
            row.append(f"{t:.1f}")
            if t < best[label][0]:
                best[label] = (t, (G, kc, nw, ns))
        if not ok:
            continue
        cells = " ".join(f"{c:>10}" for c in row)
        ratios = " ".join(
            f"{label}={base[label]/float(c):.2f}x"
            for (_, label), c in zip(skv_list, row) if c != "X"
        )
        print(f"{G:>5} {kc:>9} {nw:>6} {ns:>7} | {cells} | {ratios}")

    print()
    print(f"K={K} best:")
    for skv_k, label in skv_list:
        t, cfg = best[label]
        if cfg is None:
            print(f"  {label}: no valid config")
            continue
        ratio = base[label] / t
        print(f"  {label}: {t:.1f} μs  cfg=(GROUP_SIZE={cfg[0]}, "
              f"K_CHUNKS={cfg[1]}, warps={cfg[2]}, stages={cfg[3]})  "
              f"vs in-tree {base[label]:.1f} → {ratio:.2f}x")


if __name__ == "__main__":
    main()
