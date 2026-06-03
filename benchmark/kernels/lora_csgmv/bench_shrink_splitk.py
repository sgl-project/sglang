"""Self-contained micro-benchmark for _moe_lora_shrink_splitk_kernel (LoRA-A shrink).

Scoped to the qwen3.5-35b local-EP gate_up shrink: tp=4/ep=4 -> 64 local experts,
hidden=2048, rank=16, top_k=8. Builds v1 routing and times the shrink kernel in
isolation (CUDA-graph replay), with a correctness check and an eager mode for ncu.

  python3 bench_shrink_splitk.py --mode bench
  python3 bench_shrink_splitk.py --mode correctness   # sweeps block_m {16,32,64} x bs {16,64}
  python3 bench_shrink_splitk.py --mode profile --iters 2   # eager, for ncu

correctness mode also guards the routing/tiling block-size contract: the launcher must tile
with the same block size the routing buffers were aligned with, else expert_ids overruns -> IMA.
"""
from __future__ import annotations

import argparse

import torch
import triton
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    _invoke_moe_lora_shrink_splitk,
    fused_sanitize_expert_ids,
)


def make_inputs(bs, num_experts, top_k, hidden, rank, dtype, device):
    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    hidden_states = torch.randn(bs, hidden, device=device, dtype=dtype) * 0.1
    lora_a = torch.randn(1, num_experts, rank, hidden, device=device, dtype=dtype) * 0.1
    return topk_ids, token_lora_mapping, hidden_states, lora_a


def build_v1_routing(topk_ids, token_lora_mapping, num_experts, block_m):
    virtual_topk_ids, _, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids, token_lora_mapping, num_experts, shared_outer=False, max_loras=1
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        virtual_topk_ids, block_m, virtual_num_experts
    )
    num_tokens = topk_ids.numel()
    max_nonempty = min(num_tokens, virtual_num_experts)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    return (
        sorted_token_ids[:tight],
        fused_sanitize_expert_ids(expert_ids[: tight // block_m], virtual_num_experts),
        num_tokens_post_padded,
    )


def shrink(hidden_states, lora_a, topk_ids, routing, block_m, block_k,
           num_warps=4, num_stages=4, force_split_k=None, swap_mn=False):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    lora_a_virtual = lora_a.reshape(lora_a.shape[0] * lora_a.shape[1], *lora_a.shape[2:])
    rank = lora_a.shape[2]
    top_k = topk_ids.shape[1]
    # SPLIT_K > 1 atomic-adds, so the intermediate must be zeroed each call.
    intermediate = torch.zeros(
        topk_ids.numel(), rank, dtype=hidden_states.dtype, device=hidden_states.device
    )
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SWAP_MN": swap_mn,
    }
    if force_split_k is not None:
        config["FORCE_SPLIT_K"] = force_split_k
    _invoke_moe_lora_shrink_splitk(
        hidden_states,
        lora_a_virtual,
        intermediate,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        config,
    )
    return intermediate


def ref_shrink(hidden_states, lora_a, topk_ids):
    bs, top_k = topk_ids.shape
    rank = lora_a.shape[2]
    a = lora_a[0].float()
    out = torch.zeros(bs * top_k, rank, device=hidden_states.device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            out[m * top_k + k] = hidden_states[m].float() @ a[e].t()
    return out


def bench_ms(fn, warmup=25, rep=100, cudagraph=True, inner=200):
    """Per-call milliseconds.

    With ``cudagraph``, capture ``inner`` back-to-back ``fn()`` calls in ONE graph
    and divide the measured replay time by ``inner``. A single fn()-per-graph
    ``do_bench(g.replay)`` floors at ~8-10us for ANY tiny op -- it measures the
    fixed per-replay launch/dispatch overhead, not the kernel. Amortizing over
    ``inner`` back-to-back calls drives that overhead to ~0 and exposes the true
    device time. (Same technique as control_shrink_floor.py.)
    """
    torch.cuda.synchronize()
    if cudagraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for _ in range(inner):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(g.replay, warmup=warmup, rep=rep) / inner
    else:
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    torch.cuda.synchronize()
    return float(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness", "profile", "sweep", "sweepk"], default="bench")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-k", type=int, default=256)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--force-split-k", type=int, default=None)
    ap.add_argument("--transpose", action="store_true",
                    help="SWAP_MN: rank in MMA M-slot, tokens in N-slot")
    ap.add_argument("--tol", type=float, default=5e-1)
    args = ap.parse_args()

    dev = "cuda"
    if args.mode == "sweep":
        for bs in [1, 2, 4, 8, 16, 32, 64]:
            for rank in [16, 32, 64]:
                for hidden in [512, 768, 2048, 7168]:
                    topk_ids, tlm, hidden_states, lora_a = make_inputs(
                        bs, args.num_experts, args.top_k, hidden, rank,
                        torch.bfloat16, dev,
                    )
                    # print(f"Testing with bs={bs}, rank={rank}, hidden={hidden}")
                    best = None
                    for block_m in [8, 16, 32, 64]:
                        routing_m = build_v1_routing(topk_ids, tlm, args.num_experts, block_m)
                        for block_k in [128, 256, 512]:
                            for nw in [2, 4]:
                                for ns in [2, 3, 4]:
                                    for sk in [1, 2, 3, 4, 5, 6, 7, 8]:
                                        for swap_mn in [False, True]:
                                            f = lambda bm=block_m, bk=block_k, w=nw, s=ns, r=routing_m: shrink(
                                                hidden_states, lora_a, topk_ids, r, bm, bk, w, s,
                                                swap_mn=swap_mn, force_split_k=sk)
                                            try:
                                                ms = bench_ms(f, warmup=15, rep=60)
                                            except Exception as e:
                                                # print(f"  ERROR: {e}")
                                                continue
                                            us = ms * 1000
                                            tag = f"bm={block_m} bk={block_k} warps={nw} stages={ns} swap_mn={swap_mn}, force_split_k={sk}"
                                            if best is None or us < best[0]:
                                                best = (us, tag)
                                            #     print(f"  * {us:7.2f} us  {tag}")
                                            # else:
                                            #     print(f"    {us:7.2f} us  {tag}")
                    print(f"\nBEST bs={bs}, rank={rank}, hidden={hidden}: {best[0]:.2f} us  {best[1]}")
        return

    # Shared inputs for the non-sweep modes (sweep builds its own per iteration).
    topk_ids, tlm, hidden_states, lora_a = make_inputs(
        args.bs, args.num_experts, args.top_k, args.hidden, args.rank,
        torch.bfloat16, dev,
    )
    routing = build_v1_routing(topk_ids, tlm, args.num_experts, args.block_m)

    if args.mode == "sweepk":
        for sk in [1, 2, 3, 4, 5, 6, 7, 8]:
            f = lambda s=sk: shrink(hidden_states, lora_a, topk_ids, routing,
                                    args.block_m, args.block_k, force_split_k=s,
                                    swap_mn=args.transpose)
            ms = bench_ms(f, warmup=20, rep=80)
            print(f"  SPLIT_K={sk}: {ms * 1000:7.2f} us")
        return

    fn = lambda: shrink(hidden_states, lora_a, topk_ids, routing, args.block_m,
                        args.block_k, force_split_k=args.force_split_k,
                        swap_mn=args.transpose)

    if args.mode == "correctness":
        # Guard the routing/tiling block-size contract: _invoke_moe_lora_shrink_splitk must
        # tile with the SAME block the routing buffers were aligned with (it reads
        # config["BLOCK_SIZE_M"]; expert_ids has one entry per M-block). A launcher that
        # hardcodes one block size while the routing uses another overruns expert_ids ->
        # CUDA illegal memory access (the f2adddd "shrink" regression, which only bit the
        # server because its tuned config used block_m=64 while the launcher hardcoded 16).
        # Build routing AND config with the same block_m per iteration, but SWEEP block_m so
        # any single hardcoded value diverges from the routing at the other block sizes. Sweep
        # bs too: a larger grid makes the overrun deterministic enough to fault.
        block_ms = sorted({args.block_m, 16, 32, 64})
        batch_sizes = sorted({args.bs, 16, 64})
        failures = 0
        for bs in batch_sizes:
            tk, tlm_b, hs, la = make_inputs(
                bs, args.num_experts, args.top_k, args.hidden, args.rank,
                torch.bfloat16, dev,
            )
            ref = ref_shrink(hs, la, tk)
            for bm in block_ms:
                routing_bm = build_v1_routing(tk, tlm_b, args.num_experts, bm)
                out = shrink(
                    hs, la, tk, routing_bm, bm, args.block_k,
                    force_split_k=args.force_split_k, swap_mn=args.transpose,
                ).float()
                err = float((out - ref).abs().max().item())
                ok = err <= args.tol
                failures += int(not ok)
                print(f"{'PASS' if ok else 'FAIL'} bs={bs:<3d} block_m={bm:<2d} "
                      f"max_abs_err={err:.4e}")
        if failures:
            raise SystemExit(1)
    elif args.mode == "profile":
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
    else:
        ms = bench_ms(fn)
        print(f"BENCH shrink_splitk gate_up bs={args.bs} r={args.rank} "
              f"K={args.hidden} N={args.rank}: {ms * 1000:.2f} us "
              f"(amortized true device time)")


if __name__ == "__main__":
    main()
