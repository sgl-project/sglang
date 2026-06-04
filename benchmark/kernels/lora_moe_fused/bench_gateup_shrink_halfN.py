"""Probe: gate_up LoRA shrink can produce N=16 (gate-shrink only) instead of N=32.

The gate_up shrink weight is [E, 2*rank=32, K]; the downstream expand
(_moe_lora_expand_add, GATED_A_HALF=0) reads ONLY intermediate columns [0:rank]=[0:16]
for both the gate and up output halves. The shrink's [16:32] output (the "up shrink")
is therefore never consumed in the current (gated-split-removed) path. Because shrink
column r depends only on weight row r, computing N=16 (weight rows [0:16]) yields a
bit-identical [0:16] while halving the weight read.

This probe times the gate_up shrink at N=32 vs N=16 and asserts inter[:,0:16] is
bitwise identical, on the exact e2e decode shape.

  python3 bench_gateup_shrink_halfN.py --mode correctness
  python3 bench_gateup_shrink_halfN.py --mode bench
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    _invoke_moe_lora_shrink_splitk,
)

QWEN35_EP4 = {
    "num_experts": 256,
    "local_num_experts": 64,
    "local_expert_offset": 0,
    "top_k": 8,
}
K = 2048


def make_inputs(bs, ep, dtype, device, seed=0, routing="skewed", skew_a=0.9):
    gen = torch.Generator(device=device).manual_seed(seed)
    if routing == "skewed":
        pop = torch.arange(1, ep["num_experts"] + 1, dtype=torch.float32, device=device).pow(-skew_a)
        pop = pop[torch.randperm(ep["num_experts"], generator=gen, device=device)]
        topk_ids = torch.multinomial(pop.expand(bs, -1), ep["top_k"], replacement=False, generator=gen).to(torch.int32)
    else:
        scores = torch.rand(bs, ep["num_experts"], generator=gen, device=device)
        topk_ids = torch.topk(scores, k=ep["top_k"], dim=1).indices.to(torch.int32)
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    hidden = torch.randn(bs, K, generator=gen, device=device, dtype=dtype) * 0.1
    weight = torch.randn(ep["num_experts"], 32, K, generator=gen, device=device, dtype=dtype) * 0.1
    return topk_ids, tlm, hidden, weight


def build_routing(topk_ids, tlm, ep, block_m):
    vt, _, vne = _fused_virtual_topk_ids(
        topk_ids, tlm, ep["num_experts"], shared_outer=False, max_loras=1,
        local_expert_offset=ep["local_expert_offset"], local_num_experts=ep["local_num_experts"],
    )
    sti, eid, ntp = moe_align_block_size(vt, block_m, vne)
    num_tokens = topk_ids.numel()
    populated = ep["local_num_experts"] + 1
    max_nonempty = min(num_tokens, populated)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    return sti[:tight], eid[: tight // block_m], ntp


def shrink(hidden, weight, out, topk_ids, routing, cfg):
    sti, eid, ntp = routing
    _invoke_moe_lora_shrink_splitk(hidden, weight, out, topk_ids, sti, eid, ntp, topk_ids.shape[1], cfg)


def auto_groups(gbytes, l2_mult, mn, mx):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    return max(mn, min(math.ceil(l2 * l2_mult / max(gbytes, 1)), mx))


def touched(topk_ids, ep, n):
    lo = ep["local_expert_offset"]
    owned = (topk_ids >= lo) & (topk_ids < lo + ep["local_num_experts"])
    pairs = int(owned.sum().item())
    uniq = int(topk_ids[owned.bool()].unique().numel())
    return 2 * (topk_ids.shape[0] * K + uniq * n * K + pairs * n)


def bench(call, rep):
    call(); torch.cuda.synchronize()
    return float(triton.testing.do_bench_cudagraph(call, rep=rep)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "bench", "sweep"], default="bench")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--routing", choices=["uniform", "skewed"], default="skewed")
    ap.add_argument("--rep-ms", type=int, default=400)
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--max-groups", type=int, default=2000)
    args = ap.parse_args()
    device, dtype, ep = "cuda", torch.bfloat16, QWEN35_EP4
    cfg = {"BLOCK_SIZE_M": 16, "num_warps": 4, "num_stages": 3}

    if args.mode == "correctness":
        for seed in range(4):
            for routing in ("uniform", "skewed"):
                topk_ids, tlm, hidden, weight = make_inputs(args.bs, ep, dtype, device, seed=seed, routing=routing)
                rt = build_routing(topk_ids, tlm, ep, cfg["BLOCK_SIZE_M"])
                out32 = torch.zeros(args.bs * ep["top_k"], 32, device=device, dtype=dtype)
                out16 = torch.zeros(args.bs * ep["top_k"], 16, device=device, dtype=dtype)
                shrink(hidden, weight, out32, topk_ids, rt, cfg)
                shrink(hidden, weight[:, :16, :].contiguous(), out16, topk_ids, rt, cfg)
                diff = (out32[:, :16].float() - out16.float()).abs().max().item()
                print(f"seed={seed} routing={routing}: max|inter32[:16]-inter16| = {diff:.3e} {'PASS' if diff==0 else 'FAIL'}")
                assert diff == 0, "N=16 not bit-identical to N=32[:16]"
        print("ALL PASS (bitwise identical)")
        return

    if args.mode == "sweep":
        topk_ids0, _, _, _ = make_inputs(args.bs, ep, dtype, device, seed=0, routing=args.routing)
        n = 16
        gbytes = touched(topk_ids0, ep, n)
        ng = auto_groups(gbytes, args.l2_mult, 4, args.max_groups)
        groups = [make_inputs(args.bs, ep, dtype, device, seed=g, routing=args.routing) for g in range(ng)]
        results = []
        for bm in (16, 32):
            for nw in (2, 4, 8):
                for ns in (2, 3, 4):
                    scfg = {"BLOCK_SIZE_M": bm, "num_warps": nw, "num_stages": ns}
                    routings = [build_routing(gg[0], gg[1], ep, bm) for gg in groups]
                    outs = [torch.zeros(args.bs * ep["top_k"], n, device=device, dtype=dtype) for _ in range(ng)]
                    calls = []
                    for (topk_ids, tlm, hidden, weight), rt, out in zip(groups, routings, outs):
                        w = weight[:, :16, :].contiguous()
                        calls.append((lambda h, w, o, t, r, c: (lambda: shrink(h, w, o, t, r, c)))(hidden, w, out, topk_ids, rt, scfg))
                    try:
                        us = bench(lambda: [c() for c in calls], args.rep_ms) / ng
                    except Exception:
                        continue
                    results.append((us, f"block_m={bm} warps={nw} stages={ns}"))
        results.sort()
        for us, tag in results[:10]:
            print(f"  N=16 {us:.2f} us  {tag}")
        return

    topk_ids0, _, _, _ = make_inputs(args.bs, ep, dtype, device, seed=0, routing=args.routing)
    for n in (32, 16):
        g32 = touched(topk_ids0, ep, n)
        ng = auto_groups(g32, args.l2_mult, 4, args.max_groups)
        groups = [make_inputs(args.bs, ep, dtype, device, seed=g, routing=args.routing) for g in range(ng)]
        routings = [build_routing(gg[0], gg[1], ep, cfg["BLOCK_SIZE_M"]) for gg in groups]
        outs = [torch.zeros(args.bs * ep["top_k"], n, device=device, dtype=dtype) for _ in range(ng)]
        calls = []
        for (topk_ids, tlm, hidden, weight), rt, out in zip(groups, routings, outs):
            w = weight if n == 32 else weight[:, :16, :].contiguous()
            calls.append((lambda h, w, o, t, r: (lambda: shrink(h, w, o, t, r, cfg)))(hidden, w, out, topk_ids, rt))
        us = bench(lambda: [c() for c in calls], args.rep_ms) / ng
        print(f"  gate_up shrink N={n}: {us:.2f} us  (groups={ng})")


if __name__ == "__main__":
    main()
