"""Shrink+expand PAIR bench: measures the real PDL overlap between the two MoE LoRA
kernels that the separate per-kernel benches cannot see.

Runs the production shrink (writes the rank intermediate) immediately followed by the
production expand (reads that intermediate, writes the LoRA delta) in ONE CUDA-graph
rotation, so PDL's shrink->expand hand-off (shrink gdc_launch_dependents after its
store; expand prefetches B then gdc_wait before reading the intermediate) is exercised.
Reports PDL vs no-PDL pair time, and checks the pair output against a fp32 PEFT
reference.

Routing is computed once (same block_size for shrink and expand), matching the
production cached-routing decode where the expand's only fresh dependency on its
immediate predecessor (the shrink) is the intermediate.

  python3 bench_pair_pdl.py --mode correctness --proj down|gate_up
  python3 bench_pair_pdl.py --mode bench --proj down|gate_up        # PDL vs no-PDL
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.testing

import sglang.srt.lora.trtllm_moe.specialized_expand as _se
import sglang.srt.lora.triton_ops.virtual_experts as _ve
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    _invoke_moe_lora_shrink_splitk,
)
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add

QWEN35_EP4 = {"num_experts": 256, "local_num_experts": 64, "local_expert_offset": 0, "top_k": 8}
# proj -> (K, rank, out_per_half, gated). down: single rank, N=2048. gate_up: 2*rank shrink, N=2*512.
PROJ = {
    "down": {"K": 512, "rank": 16, "N": 2048, "gated": False, "input_top_k": 1, "mul_routed": True, "fuse_sum": True},
    "gate_up": {"K": 2048, "rank": 16, "N": 1024, "gated": True, "input_top_k": 8, "mul_routed": False, "fuse_sum": False},
}


def disable_pdl():
    def no_pdl():
        return False, {}
    _ve.get_pdl_launch_metadata = no_pdl
    _se.get_pdl_launch_metadata = no_pdl


def make_inputs(proj, bs, ep, dtype, device, seed=0, routing="skewed", skew_a=0.9):
    spec = PROJ[proj]
    K, r, N = spec["K"], spec["rank"], spec["N"]
    gen = torch.Generator(device=device).manual_seed(seed)
    if routing == "skewed":
        pop = torch.arange(1, ep["num_experts"] + 1, dtype=torch.float32, device=device).pow(-skew_a)
        pop = pop[torch.randperm(ep["num_experts"], generator=gen, device=device)]
        topk_ids = torch.multinomial(pop.expand(bs, -1), ep["top_k"], replacement=False, generator=gen).to(torch.int32)
    else:
        scores = torch.rand(bs, ep["num_experts"], generator=gen, device=device)
        topk_ids = torch.topk(scores, k=ep["top_k"], dim=1).indices.to(torch.int32)
    topk_weights = torch.rand(bs, ep["top_k"], generator=gen, device=device, dtype=torch.float32) * 0.9 + 0.1
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    a_rows = bs if spec["input_top_k"] > 1 else bs * ep["top_k"]
    x = torch.randn(a_rows, K, generator=gen, device=device, dtype=dtype) * 0.1
    shrink_n = 2 * r if spec["gated"] else r
    a_w = torch.randn(ep["num_experts"], shrink_n, K, generator=gen, device=device, dtype=dtype) * 0.1
    b_w = torch.randn(ep["num_experts"], N, r, generator=gen, device=device, dtype=dtype) * 0.1
    return topk_ids, topk_weights, tlm, x, a_w, b_w


def build_routing(topk_ids, tlm, ep, block_m):
    vt, _, vne = _fused_virtual_topk_ids(
        topk_ids, tlm, ep["num_experts"], shared_outer=False, max_loras=1,
        local_expert_offset=ep["local_expert_offset"], local_num_experts=ep["local_num_experts"],
    )
    sti, eid, ntp = moe_align_block_size(vt, block_m, vne)
    num_tokens = topk_ids.numel()
    populated = ep["local_num_experts"] + 1
    tight = triton.cdiv(num_tokens + min(num_tokens, populated) * (block_m - 1), block_m) * block_m
    return sti[:tight], eid[: tight // block_m], ntp


def run_pair(proj, x, a_w, b_w, inter, output, topk_ids, topk_weights, routing, ep):
    spec = PROJ[proj]
    r = spec["rank"]
    sti, eid, ntp = routing
    a_cfg = {"BLOCK_SIZE_M": 16, "num_warps": 4, "num_stages": 3}
    b_cfg = {"BLOCK_SIZE_M": 16, "num_warps": 4, "num_stages": 1, "GROUP_SIZE_M": 1, "BLOCK_SIZE_N": 64}
    shrink_n = a_w.shape[1]
    inter.zero_()
    _invoke_moe_lora_shrink_splitk(
        x, a_w, inter.view(-1, shrink_n), topk_ids, sti, eid, ntp, spec["input_top_k"], a_cfg
    )
    _invoke_moe_lora_expand_add(
        inter.view(-1, shrink_n), b_w, output, topk_weights, topk_ids, sti, eid, ntp,
        b_cfg, spec["mul_routed"], spec["fuse_sum"],
    )


def ref_pair(proj, x, a_w, b_w, topk_ids, topk_weights, ep, device):
    spec = PROJ[proj]
    bs, top_k = topk_ids.shape
    r, N = spec["rank"], spec["N"]
    lo, hi = ep["local_expert_offset"], ep["local_expert_offset"] + ep["local_num_experts"]
    A, B, xf = a_w.float(), b_w.float(), x.float()
    if spec["fuse_sum"]:
        out = torch.zeros(bs, N, device=device, dtype=torch.float32)
    else:
        out = torch.zeros(bs, top_k, N, device=device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            if not (lo <= e < hi):
                continue
            vt = m * top_k + k
            row = xf[m] if spec["input_top_k"] > 1 else xf[vt]
            if spec["gated"]:
                gate = (row @ A[e][0:r].t()) @ B[e][0 : N // 2].t()
                up = (row @ A[e][r : 2 * r].t()) @ B[e][N // 2 : N].t()
                delta = torch.cat([gate, up], dim=-1)
            else:
                delta = (row @ A[e][0:r].t()) @ B[e].t()
            if spec["mul_routed"]:
                delta = delta * float(topk_weights[m, k].item())
            if spec["fuse_sum"]:
                out[m] += delta
            else:
                out[m, k] = delta
    return out


def alloc_io(proj, bs, ep, a_w, device, dtype):
    spec = PROJ[proj]
    shrink_n = a_w.shape[1]
    inter = torch.zeros(bs * ep["top_k"], shrink_n, device=device, dtype=dtype)
    if spec["fuse_sum"]:
        output = torch.zeros(bs, spec["N"], device=device, dtype=dtype)
    else:
        output = torch.zeros(bs, ep["top_k"], spec["N"], device=device, dtype=dtype)
    return inter, output


def auto_groups(gbytes, l2_mult, mn, mx):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    return max(mn, min(math.ceil(l2 * l2_mult / max(gbytes, 1)), mx))


def touched(proj, topk_ids, ep, a_w):
    spec = PROJ[proj]
    lo = ep["local_expert_offset"]
    owned = (topk_ids >= lo) & (topk_ids < lo + ep["local_num_experts"])
    pairs = int(owned.sum().item())
    uniq = int(topk_ids[owned.bool()].unique().numel())
    x_b = (topk_ids.shape[0] if spec["input_top_k"] > 1 else pairs) * spec["K"]
    a_b = uniq * a_w.shape[1] * spec["K"]
    b_b = uniq * spec["N"] * spec["rank"]
    out_b = topk_ids.shape[0] * spec["N"]
    return 2 * (x_b + a_b + b_b + out_b)


def bench(call, rep):
    call(); torch.cuda.synchronize()
    return float(triton.testing.do_bench_cudagraph(call, rep=rep)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "bench"], default="bench")
    ap.add_argument("--proj", choices=["down", "gate_up"], default="down")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--routing", choices=["uniform", "skewed"], default="skewed")
    ap.add_argument("--rep-ms", type=int, default=400)
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--max-groups", type=int, default=2000)
    args = ap.parse_args()
    device, dtype, ep, proj = "cuda", torch.bfloat16, QWEN35_EP4, args.proj

    if args.mode == "correctness":
        import sglang.srt.environ as _env
        _env.envs.SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.set(True)
        fails = 0
        for seed in range(4):
            for rt in ("uniform", "skewed"):
                topk_ids, topk_weights, tlm, x, a_w, b_w = make_inputs(proj, args.bs, ep, dtype, device, seed=seed, routing=rt)
                routing = build_routing(topk_ids, tlm, ep, 16)
                inter, output = alloc_io(proj, args.bs, ep, a_w, device, dtype)
                run_pair(proj, x, a_w, b_w, inter, output, topk_ids, topk_weights, routing, ep)
                ref = ref_pair(proj, x, a_w, b_w, topk_ids, topk_weights, ep, device)
                err = (output.float() - ref).abs().max().item()
                rel = err / (ref.abs().max().item() + 1e-9)
                ok = err < 2e-2 or rel < 2e-2
                fails += int(not ok)
                print(f"proj={proj} seed={seed} routing={rt}: max_abs={err:.3e} rel={rel:.3e} {'PASS' if ok else 'FAIL'}")
        if fails:
            raise SystemExit(f"{fails} FAIL")
        print("ALL PASS")
        return

    import sglang.srt.environ as _env
    _env.envs.SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.set(True)
    topk_ids0, _, _, _, a_w0, _ = make_inputs(proj, args.bs, ep, dtype, device, seed=0, routing=args.routing)
    ng = auto_groups(touched(proj, topk_ids0, ep, a_w0), args.l2_mult, 4, args.max_groups)
    groups = [make_inputs(proj, args.bs, ep, dtype, device, seed=g, routing=args.routing) for g in range(ng)]
    routings = [build_routing(g[0], g[2], ep, 16) for g in groups]
    ios = [alloc_io(proj, args.bs, ep, g[4], device, dtype) for g in groups]

    def make_calls():
        calls = []
        for (topk_ids, topk_weights, tlm, x, a_w, b_w), routing, (inter, output) in zip(groups, routings, ios):
            calls.append((lambda *a: (lambda: run_pair(proj, *a)))(x, a_w, b_w, inter, output, topk_ids, topk_weights, routing, ep))
        return calls

    def run_all(calls):
        for c in calls:
            c()

    pdl_us = bench(lambda: run_all(make_calls()), args.rep_ms) / ng
    disable_pdl()
    nopdl_us = bench(lambda: run_all(make_calls()), args.rep_ms) / ng
    pad = int(routings[0][2].item())
    print(f"PAIR proj={proj} bs={args.bs} routing={args.routing} padded={pad} groups={ng}")
    print(f"  shrink+expand pair  no-PDL: {nopdl_us:.2f} us")
    print(f"  shrink+expand pair  PDL:    {pdl_us:.2f} us   speedup {nopdl_us/pdl_us:.2f}x  (saved {nopdl_us-pdl_us:.2f} us)")


if __name__ == "__main__":
    main()
