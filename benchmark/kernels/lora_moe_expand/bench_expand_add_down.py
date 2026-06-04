"""Self-contained micro-benchmark + correctness check for _moe_lora_expand_add_kernel
(virtual-experts LoRA-B expand-add), covering BOTH per-layer callsites.

Shapes reproduce the measured e2e decode of Qwen3.5-35B-A3B-FP8 tp4/ep4 bs64 with a
single rank-16 adapter (SHAPECAP capture 2026-06-04):

  * ``--proj gate_up``: intermediate [512, 32] (the 2-slice shrink output; the kernel
    reads columns [0:R]), weight [256, 1024, 16] (N = gate+up = 2*512), output
    [64, 8, 1024] per-(token, expert), mul_routed_weight=False,
    fuse_sum_all_reduce=False. e2e: BLOCK 16/128/16, grid (744,), warps=4.
  * ``--proj down``: intermediate [512, 16], weight [256, 2048, 16] (N = hidden),
    output [64, 2048], mul_routed_weight=True, fuse_sum_all_reduce=True (each token's
    top-k deltas are weight-scaled and atomic-added into one row). e2e: BLOCK
    16/128/16, grid (1488,), warps=4.

EP matters: LoRA expert weights stay GLOBAL (256 experts) while this rank owns 64;
non-owned (token, expert) slots are dropped to the -1 sentinel in routing, so per-rank
work is ~1/4 of the routed pairs and the routing buffers are sorted_token_ids [1488] /
expert_ids [93] at bs64 block_m=16 (production ``_get_routing`` non-fused path replica).

Benchmark methodology: rotate N auto-sized buffer groups (footprint = ``--l2-mult`` x
L2, default 4x) inside one CUDA graph timed by ``triton.testing.do_bench_cudagraph``;
reported time = graph time / N. Host launch overhead amortizes to ~0 and no weight /
intermediate is served out of L2 on its next use.

  python3 bench_expand_add_down.py --mode bench --proj down
  python3 bench_expand_add_down.py --mode bench --proj gate_up
  python3 bench_expand_add_down.py --mode correctness
  python3 bench_expand_add_down.py --mode sweep --proj down       # block_n/group/warps scan
  python3 bench_expand_add_down.py --mode profile --proj down --iters 4   # for ncu
"""

from __future__ import annotations

import argparse
import functools
import math

import torch
import triton
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import _fused_virtual_topk_ids
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add

# qwen3.5-35b tp4/ep4: 256 global routed experts, 64 owned per rank, router top-8.
QWEN35_EP4 = {
    "num_experts": 256,
    "local_num_experts": 64,
    "local_expert_offset": 0,
    "top_k": 8,
}
# Per-projection expand shapes. intermediate_cols: the shrink output the expand reads
# from (gate_up's 2-slice shrink is 2*rank wide; the kernel reads columns [0:rank]).
PROJ = {
    "gate_up": {
        "n": 1024,
        "intermediate_cols": 32,
        "mul_routed_weight": False,
        "fuse_sum_all_reduce": False,
    },
    "down": {
        "n": 2048,
        "intermediate_cols": 16,
        "mul_routed_weight": True,
        "fuse_sum_all_reduce": True,
    },
}


def production_config(num_experts, n, rank, bs, dtype):
    """The config production's ``_get_stage_config`` launches the expand with
    (try_get_optimal_moe_config on the GLOBAL virtual weight shape, stage_top_k=1,
    M=bs). BLOCK_SIZE_N is left to the launcher (forces 128 when N % 128 == 0).
    Fallback = the e2e-observed decode config (16, 1, 4)."""
    try:
        from sglang.srt.server_args import (
            ServerArgs,
            get_global_server_args,
            set_global_server_args_for_scheduler,
        )

        try:
            get_global_server_args()
        except Exception:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        weight_shape = (num_experts, n, rank)
        config_dtype = get_config_dtype_str(dtype=dtype)
        cfg = functools.partial(
            try_get_optimal_moe_config, weight_shape, weight_shape, 1, config_dtype
        )(bs)
        return cfg["BLOCK_SIZE_M"], cfg.get("GROUP_SIZE_M", 1), cfg.get("num_warps", 4)
    except Exception:
        return 16, 1, 4


def make_routing_inputs(bs, ep, device, seed=0, routing="uniform", skew_a=0.9):
    """Top-k routing over the GLOBAL expert ids + per-token routing weights + the
    all-zero (single adapter, slot 0) token->lora mapping. ``routing="skewed"`` draws
    experts from a Zipf(skew_a) popularity (hot experts), which reproduces the e2e
    num_tokens_post_padded range (measured decode bs64: 832..1312, median 1056)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    if routing == "skewed":
        pop = torch.arange(
            1, ep["num_experts"] + 1, dtype=torch.float32, device=device
        ).pow(-skew_a)
        pop = pop[torch.randperm(ep["num_experts"], generator=gen, device=device)]
        topk_ids = (
            torch.multinomial(
                pop.expand(bs, -1), ep["top_k"], replacement=False, generator=gen
            )
        ).to(torch.int32)
    else:
        scores = torch.rand(bs, ep["num_experts"], generator=gen, device=device)
        topk_ids = torch.topk(scores, k=ep["top_k"], dim=1).indices.to(torch.int32)
    topk_weights = (
        torch.rand(bs, ep["top_k"], generator=gen, device=device, dtype=torch.float32)
        * 0.9
        + 0.1
    )
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    return topk_ids, topk_weights, token_lora_mapping


def build_ep_routing(topk_ids, token_lora_mapping, ep, block_m):
    """Production ``_get_routing`` replica (non-fused path, max_loras=1, EP local mask):
    virtual ids (non-owned -> -1) -> moe_align_block_size -> tight trim with
    ``local_num_experts * max_loras + 1`` populated buckets."""
    max_loras = 1
    virtual_topk_ids, _, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        ep["num_experts"],
        shared_outer=False,
        max_loras=max_loras,
        local_expert_offset=ep["local_expert_offset"],
        local_num_experts=ep["local_num_experts"],
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        virtual_topk_ids, block_m, virtual_num_experts
    )
    num_tokens = topk_ids.numel()
    populated_buckets = ep["local_num_experts"] * max_loras + 1
    max_nonempty = min(num_tokens, populated_buckets)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    # max_loras == 1 -> fused_sanitize_expert_ids is an identity, skipped (as production).
    return (
        sorted_token_ids[:tight],
        expert_ids[: tight // block_m],
        num_tokens_post_padded,
    )


def make_gemm_inputs(proj, bs, ep, rank, dtype, device, seed=0):
    """intermediate is the per-(token, expert) shrink output [bs*top_k, cols]; weight is
    the GLOBAL [num_experts, N, rank] LoRA-B (production's pre-merged free view).
    Output rows: per-(token, expert) 3D for gate_up, per-token 2D for down."""
    gen = torch.Generator(device=device).manual_seed(seed)
    spec = PROJ[proj]
    intermediate = (
        torch.randn(
            bs * ep["top_k"],
            spec["intermediate_cols"],
            generator=gen,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    weight = (
        torch.randn(
            ep["num_experts"],
            spec["n"],
            rank,
            generator=gen,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    if spec["fuse_sum_all_reduce"]:
        output = torch.zeros(bs, spec["n"], device=device, dtype=dtype)
    else:
        output = torch.zeros(bs, ep["top_k"], spec["n"], device=device, dtype=dtype)
    return intermediate, weight, output


def expand_call(
    proj,
    intermediate,
    weight,
    output,
    topk_weights,
    topk_ids,
    routing,
    config,
    force_block_n=None,
):
    spec = PROJ[proj]
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    return lambda: _invoke_moe_lora_expand_add(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        spec["mul_routed_weight"],
        spec["fuse_sum_all_reduce"],
        force_block_size_n=force_block_n,
    )


def ref_expand(proj, intermediate, weight, topk_ids, topk_weights, ep):
    """fp32 reference over OWNED (token, expert) pairs only; non-owned slots stay zero
    (the direct expand never writes them). The kernel reads intermediate[:, 0:R]."""
    spec = PROJ[proj]
    bs, top_k = topk_ids.shape
    rank = weight.shape[2]
    lo = ep["local_expert_offset"]
    hi = lo + ep["local_num_experts"]
    w = weight.float()
    inter = intermediate[:, :rank].float()
    if spec["fuse_sum_all_reduce"]:
        out = torch.zeros(bs, spec["n"], device=weight.device, dtype=torch.float32)
    else:
        out = torch.zeros(
            bs, top_k, spec["n"], device=weight.device, dtype=torch.float32
        )
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            if not (lo <= e < hi):
                continue
            delta = inter[m * top_k + k] @ w[e].t()
            if spec["mul_routed_weight"]:
                delta = delta * float(topk_weights[m, k].item())
            if spec["fuse_sum_all_reduce"]:
                out[m] += delta
            else:
                out[m, k] = delta
    return out


def auto_num_groups(
    group_bytes: int, l2_mult: float, min_groups: int, max_groups: int
) -> int:
    """Enough buffer groups that the rotation footprint is ``l2_mult`` x L2, so no
    group survives in L2 until its next use. Err on the high side: an optimized kernel
    that reads less memory needs MORE groups for the same eviction guarantee."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    need = math.ceil(l2_bytes * l2_mult / max(group_bytes, 1))
    n = max(min_groups, min(need, max_groups))
    if n * group_bytes < l2_mult * l2_bytes:
        print(
            f"WARNING: rotation footprint {n * group_bytes / 1e6:.0f} MB < "
            f"{l2_mult:.1f} x L2 ({l2_mult * l2_bytes / 1e6:.0f} MB); raise "
            f"--max-groups for full L2 eviction (small-shape kernel)"
        )
    return n


def bench_us_rotated(calls, rep_ms: int) -> float:
    """Capture all rotated calls in ONE CUDA graph via do_bench_cudagraph; per-call us =
    graph time / num_groups. Outputs accumulate garbage across replays -- harmless for
    timing; correctness mode re-zeroes."""

    def fn():
        for call in calls:
            call()

    fn()  # eager warmup: triton JIT compile outside graph capture
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(fn, rep=rep_ms)
    return float(ms) * 1e3 / len(calls)


def build_rotated_calls(
    args, proj, ep, config, routing_pack, device, dtype, force_block_n=None
):
    topk_ids, topk_weights, routing = routing_pack
    spec = PROJ[proj]
    group_bytes = 2 * (
        args.bs * ep["top_k"] * spec["intermediate_cols"]
        + ep["num_experts"] * spec["n"] * args.rank
        + (args.bs if spec["fuse_sum_all_reduce"] else args.bs * ep["top_k"])
        * spec["n"]
    )
    num_groups = args.num_groups or auto_num_groups(
        group_bytes, args.l2_mult, args.min_groups, args.max_groups
    )
    groups = [
        make_gemm_inputs(proj, args.bs, ep, args.rank, dtype, device, seed=g)
        for g in range(num_groups)
    ]
    calls = [
        expand_call(
            proj,
            inter,
            weight,
            out,
            topk_weights,
            topk_ids,
            routing,
            config,
            force_block_n,
        )
        for inter, weight, out in groups
    ]
    return calls, num_groups, group_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["bench", "correctness", "profile", "sweep"],
        default="bench",
    )
    ap.add_argument("--proj", choices=[*PROJ, "all"], default="all")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--routing",
        choices=["uniform", "skewed"],
        default="uniform",
        help="skewed = Zipf(--skew-a) hot experts, matches e2e padded-token range",
    )
    ap.add_argument("--skew-a", type=float, default=0.9)
    ap.add_argument(
        "--num-groups",
        type=int,
        default=0,
        help="rotated buffer groups; 0 = auto-size to --l2-mult x L2",
    )
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--min-groups", type=int, default=16)
    ap.add_argument("--max-groups", type=int, default=512)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--iters", type=int, default=4, help="profile-mode eager sweeps")
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    ep = QWEN35_EP4
    projs = list(PROJ) if args.proj == "all" else [args.proj]

    if args.mode == "correctness":
        # Also guard the routing/tiling block-size contract: the launcher must tile
        # expert_ids with the same BLOCK_SIZE_M the routing was aligned with.
        failures = 0
        for proj in projs:
            for bs in sorted({args.bs, 16, 64}):
                topk_ids, topk_weights, tlm = make_routing_inputs(
                    bs, ep, device, seed=args.seed
                )
                inter, weight, out = make_gemm_inputs(
                    proj, bs, ep, args.rank, dtype, device
                )
                ref = ref_expand(proj, inter, weight, topk_ids, topk_weights, ep)
                for block_m in [16, 32, 64]:
                    config = {
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": 128,
                        "GROUP_SIZE_M": 1,
                        "num_warps": 4,
                    }
                    routing = build_ep_routing(topk_ids, tlm, ep, block_m)
                    out.zero_()
                    expand_call(
                        proj,
                        inter,
                        weight,
                        out,
                        topk_weights,
                        topk_ids,
                        routing,
                        config,
                    )()
                    err = float((out.float() - ref).abs().max().item())
                    ok = err <= args.tol
                    failures += int(not ok)
                    print(
                        f"{'PASS' if ok else 'FAIL'} proj={proj:<7s} bs={bs:<3d} "
                        f"block_m={block_m:<2d} max_abs_err={err:.4e}"
                    )
        if failures:
            raise SystemExit(1)
        return

    for proj in projs:
        spec = PROJ[proj]
        block_m, group_m, num_warps = production_config(
            ep["num_experts"], spec["n"], args.rank, args.bs, dtype
        )
        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": 128,  # launcher forces 128 anyway (N % 128 == 0)
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
        }
        topk_ids, topk_weights, tlm = make_routing_inputs(
            args.bs, ep, device, seed=args.seed, routing=args.routing, skew_a=args.skew_a
        )

        if args.mode == "sweep":
            best = None
            block_ns = [bn for bn in [64, 128, 256, 512] if spec["n"] % bn == 0]
            for bm in [16, 32, 64]:
                routing = build_ep_routing(topk_ids, tlm, ep, bm)
                for bn in block_ns:
                    for gm in [1, 4, 8]:
                        for nw in [2, 4, 8]:
                            cfg = {
                                "BLOCK_SIZE_M": bm,
                                "BLOCK_SIZE_N": bn,
                                "GROUP_SIZE_M": gm,
                                "num_warps": nw,
                            }
                            calls, _, _ = build_rotated_calls(
                                args,
                                proj,
                                ep,
                                cfg,
                                (topk_ids, topk_weights, routing),
                                device,
                                dtype,
                                force_block_n=bn,
                            )
                            try:
                                us = bench_us_rotated(calls, args.rep_ms)
                            except Exception:
                                continue
                            tag = f"block_m={bm} block_n={bn} group_m={gm} warps={nw}"
                            if best is None or us < best[0]:
                                best = (us, tag)
                            print(f"  {us:7.2f} us  proj={proj} {tag}")
            print(f"\nBEST proj={proj} bs={args.bs}: {best[0]:.2f} us  {best[1]}")
            continue

        routing = build_ep_routing(topk_ids, tlm, ep, block_m)
        calls, num_groups, group_bytes = build_rotated_calls(
            args, proj, ep, config, (topk_ids, topk_weights, routing), device, dtype
        )

        if args.mode == "profile":
            for _ in range(2):
                calls[0]()
            torch.cuda.synchronize()
            for _ in range(args.iters):
                for call in calls:
                    call()
            torch.cuda.synchronize()
            print(f"PROFILE proj={proj}: {args.iters} x {num_groups} groups done")
            continue

        sorted_token_ids = routing[0]
        padded = int(routing[2].item())  # actual aligned tokens (e2e decode: 832..1312)
        grid = triton.cdiv(sorted_token_ids.shape[0], block_m) * triton.cdiv(
            spec["n"], 128
        )
        us = bench_us_rotated(calls, args.rep_ms)
        print(
            f"BENCH moe_expand proj={proj} bs={args.bs} r={args.rank} N={spec['n']} "
            f"E={ep['num_experts']}(local {ep['local_num_experts']}) "
            f"mul_routed={int(spec['mul_routed_weight'])} fuse_sum={int(spec['fuse_sum_all_reduce'])} "
            f"sorted={sorted_token_ids.shape[0]} padded={padded} block_m={block_m} grid=({grid},) "
            f"groups={num_groups} ({group_bytes * num_groups / 1e6:.0f} MB rotated): {us:.2f} us"
        )


if __name__ == "__main__":
    main()
