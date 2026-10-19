"""Why the MoE router costs 6.7 us at M=1, and whether more warps fix it.

`_router_triton_kernel` is launched at grid (1,) for single-token decode, and the
live profile puts it at 6.7 us per call, 40 calls per token (0.27 ms/token). The
body loads the 384 logits as one (BLOCK_M, BLOCK_N) tile, so unlike the mHC
reduce kernel it is not load-count bound. The cost is the sequence of
row-wise reductions over the 512-wide padded expert axis, all of them executed by
a single warp because the launch picks `num_warps = 1 if BLOCK_N <= 512 else 4`:

  * grouped routing: N_GROUP iterations x 2 tl.max, then TOPK_GROUP x 2 more;
  * the top-k loop: K_ROUTED iterations x 3 reductions (max of cur, min of the
    tie-break lane id, and a masked sum to extract the winner's activation).

At 32 lanes a 512-wide reduction is 16 sequential elements per lane plus 5
shuffle steps, and each iteration depends on the previous one. This measures the
kernel at the live shape (M=1, N=384, K=6, sqrtsoftplus, renormalize, bias) for
warps 1/2/4/8 and for both routing geometries, and checks that the weights and
indices are bitwise identical across warp counts - they should be, since every
reduction involved is exact regardless of association (tl.max, tl.min over
integers, and a sum of one non-zero plus zeros).

Run: PYTHONPATH=python python3 test/manual/layers/moe/bench_router_warps.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton


def graph_ms(fn, inner=20, warm=3, rounds=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warm):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / inner)
    del graph
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--calls", type=int, default=40)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    print(json.dumps({"kind": "env", "name": torch.cuda.get_device_name(),
                      "sms": torch.cuda.get_device_properties(0).multi_processor_count}),
          flush=True)

    from sglang.kernels.ops.moe.moe_fused_gate import _router_triton_kernel

    torch.manual_seed(99)
    N, K, M = args.experts, args.topk, 1
    scores = (torch.randn(M, N, device="cuda") * 1.5).float()
    bias = (torch.randn(N, device="cuda") * 0.1).float()
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = triton.next_power_of_2(K)
    unused = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run(warps, n_group, topk_group):
        w = torch.empty((M, K), dtype=torch.float32, device="cuda")
        i = torch.empty((M, K), dtype=torch.int32, device="cuda")
        _router_triton_kernel[(triton.cdiv(M, 1),)](
            scores, bias, scores, scores, scores,
            w, i, unused, M, 1.5, 0.0,
            N=N, K=K, K_ROUTED=K,
            BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            N_GROUP=n_group, TOPK_GROUP=topk_group,
            EXPERTS_PER_GROUP=N // n_group,
            BLOCK_G=max(1, triton.next_power_of_2(n_group)),
            SCORING_FUNC=1, SQRTSOFTPLUS_LOG1P=True, HAS_SOFTCAP=False,
            RENORMALIZE=True, APPLY_SCALE=False,
            HAS_BIAS=True, HAS_TOKEN_BIAS=False, BIAS_ALT_TOKEN_ID=-1,
            HAS_PADDING=False, HAS_PACKED=False, RENORMALIZE_EPSILON=0.0,
            USE_PDL=False,
            stride_bias=bias.stride(0), stride_bias_alt=0, stride_input_ids=0,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_wm=w.stride(0), stride_wk=w.stride(1),
            stride_im=i.stride(0), stride_ik=i.stride(1),
            stride_pm=0, stride_pk=0,
            num_warps=warps,
        )
        return w, i

    for n_group, topk_group, label in ((1, 1, "ungrouped"), (8, 4, "n_group=8 topk_group=4")):
        ref = run(1, n_group, topk_group)
        torch.cuda.synchronize()
        print(json.dumps({"kind": "geometry", "label": label, "n_group": n_group,
                          "topk_group": topk_group,
                          "indices": ref[1].flatten().tolist()}), flush=True)
        for warps in (1, 2, 4, 8):
            out = run(warps, n_group, topk_group)
            torch.cuda.synchronize()
            same_w = bool(torch.equal(out[0], ref[0]))
            same_i = bool(torch.equal(out[1], ref[1]))
            ms = graph_ms(lambda warps=warps, n_group=n_group, topk_group=topk_group:
                          run(warps, n_group, topk_group))
            print(json.dumps({"kind": "router", "label": label, "warps": warps,
                              "median_us": ms * 1000,
                              "projected_ms_per_token": ms * args.calls,
                              "weights_bitwise_equal": same_w,
                              "indices_bitwise_equal": same_i}), flush=True)
    print("ROUTER_WARPS_DONE", flush=True)


if __name__ == "__main__":
    main()
