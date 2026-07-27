"""Cold-cache benchmark / tuning sweep for router_gate_matvec.

Rotates NLAYER distinct gate weights inside one CUDA graph so every call
misses L2, mimicking the real decode step where ~94MB/layer of expert
traffic flushes the cache between gate calls. A hot same-weight loop
overestimates this kernel badly (7us hot vs 11us cold at the same shape) —
always tune with this harness.

Usage (single GPU):
    python3 benchmark/kernels/bench_router_gate_matvec.py

Adjust ROUTER_GATE_MATVEC_BLOCK_E / _NUM_WARPS / _MAX_M in
sglang/srt/layers/moe/router.py to the winners printed for your arch.
"""

import time

import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.moe.router import router_gate_matvec_kernel

DEV = "cuda"
E, K = 513, 2560  # ling-v3-flash router shape
NLAYER = 41
TOPK = 8


def matvec(hidden, weight, block_e, warps):
    M = hidden.shape[0]
    out = torch.empty((M, E), dtype=torch.float32, device=DEV)
    block_k = min(4096, triton.next_power_of_2(K))
    router_gate_matvec_kernel[(M, triton.cdiv(E, block_e))](
        hidden, weight, out, K, E, hidden.stride(0), weight.stride(0),
        BLOCK_E=block_e, BLOCK_K=block_k, num_warps=warps,
    )
    return out


def bench_cold(fn_per_layer, tag):
    def run_all():
        for i in range(NLAYER):
            fn_per_layer(i)

    for _ in range(3):
        run_all()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            run_all()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 30 / NLAYER * 1e6
    print(f"  {tag}: {us:.2f} us/call")
    return us


def main():
    torch.manual_seed(0)
    wf = [torch.randn(E, K, device=DEV, dtype=torch.float32) * 0.02 for _ in range(NLAYER)]
    wb = [w.to(torch.bfloat16).contiguous() for w in wf]

    # correctness: fp32 path must match the fp32 reference bit-tightly and
    # produce zero top-k routing flips
    flips = tot = 0
    max_abs = 0.0
    for t in range(100):
        torch.manual_seed(t)
        m = [1, 2, 4, 8][t % 4]
        h = torch.randn(m, K, device=DEV, dtype=torch.bfloat16) * 0.7
        w = torch.randn(E, K, device=DEV, dtype=torch.float32) * 0.02
        bias = torch.randn(E, device=DEV) * 0.01
        ref = F.linear(h.to(torch.float32), w)
        new = matvec(h, w, 4, 8)
        max_abs = max(max_abs, (ref - new).abs().max().item())
        r = (torch.sigmoid(ref) + bias).topk(TOPK, -1).indices
        n = (torch.sigmoid(new) + bias).topk(TOPK, -1).indices
        flips += torch.ne(r, n).sum().item()
        tot += m * TOPK
    print(f"fp32 correctness: max abs {max_abs:.3e}, top-{TOPK} flips {flips}/{tot}")

    for m in (1, 2, 4, 8, 16):
        h = torch.randn(m, K, device=DEV, dtype=torch.bfloat16) * 0.7
        print(f"---- M={m} ----")
        bench_cold(lambda i: F.linear(h, wb[i]), "lib bf16 F.linear")
        bench_cold(lambda i: F.linear(h.to(torch.float32), wf[i]), "lib fp32 upcast+linear")
        for be in (2, 4, 8):
            for warps in (4, 8):
                bench_cold(
                    lambda i, a=be, b=warps: matvec(h, wb[i], a, b),
                    f"matvec bf16 BLOCK_E={be} warps={warps}",
                )
        bench_cold(lambda i: matvec(h, wf[i], 4, 8), "matvec fp32 BLOCK_E=4 warps=8")


if __name__ == "__main__":
    main()
