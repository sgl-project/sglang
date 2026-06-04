"""Self-contained micro-benchmark + correctness check for _moe_lora_shrink_splitk_kernel
(virtual-experts LoRA-A shrink).

Shapes reproduce the measured e2e decode of Qwen3.5-35B-A3B-FP8 tp4/ep4 bs64 with a
single rank-16 adapter (SHAPECAP capture 2026-06-04). Per layer the kernel runs twice:

  * ``--proj gate_up``: hidden [64, 2048], weight [256, 32, 2048] (gate+up LoRA-A
    stacked, N = 2*rank = 32), top_k=8, out [512, 32]. e2e: SPLIT_K=5, grid (465,),
    BLOCK 16/32/256, warps=4 stages=3.
  * ``--proj down``: hidden [512, 512] (per-(token,expert) activation output, top_k=1),
    weight [256, 16, 512], out [512, 16]. e2e: SPLIT_K=2, grid (186,),
    BLOCK 16/16/256, warps=2 stages=3.

EP matters: the LoRA expert weights stay GLOBAL (all 256 experts) while this rank owns
only 64 of them; non-owned (token, expert) slots are dropped to the -1 sentinel inside
the routing, which shrinks the per-rank work to ~1/4 of the routed pairs and bounds the
routing buffers to sorted_token_ids [1488] / expert_ids [93] (= 64 local experts + 1
sentinel bucket, tight-trimmed). The bench replicates production ``_get_routing``
(non-fused path: _fused_virtual_topk_ids -> moe_align_block_size -> tight trim).

Benchmark methodology: rotate N auto-sized buffer groups (footprint = ``--l2-mult`` x
L2, default 4x) inside one CUDA graph timed by ``triton.testing.do_bench_cudagraph``;
reported time = graph time / N. Host launch overhead amortizes to ~0 and no weight /
activation is served out of L2 on its next use. Production zeroes the split-K
intermediate with a separate memset each step; that memset is NOT in the timed kernel
(matching kernel-level profile numbers).

  python3 bench_shrink_splitk.py --mode bench --proj gate_up
  python3 bench_shrink_splitk.py --mode bench --proj down
  python3 bench_shrink_splitk.py --mode correctness
  python3 bench_shrink_splitk.py --mode sweepk --proj gate_up      # SPLIT_K scan
  python3 bench_shrink_splitk.py --mode sweep --proj gate_up       # block/warp/stage scan
  python3 bench_shrink_splitk.py --mode profile --proj gate_up --iters 4  # for ncu
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
from sglang.srt.lora.triton_ops.kernel_utils import get_pdl_launch_metadata
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    _get_moe_lora_shrink_split_k,
    _invoke_moe_lora_shrink_splitk,
    _moe_lora_shrink_splitk_kernel,
)

# qwen3.5-35b tp4/ep4: 256 global routed experts, 64 owned per rank, router top-8.
QWEN35_EP4 = {
    "num_experts": 256,
    "local_num_experts": 64,
    "local_expert_offset": 0,
    "top_k": 8,
}
# Per-projection shrink shapes (N = LoRA-A output rows, K = input dim, input_top_k =
# the kernel's top_k arg: 8 when input rows are per-token, 1 when per-(token,expert)).
PROJ = {
    "gate_up": {"n": 32, "k": 2048, "input_top_k": 8},
    "down": {"n": 16, "k": 512, "input_top_k": 1},
}


def disable_pdl(modules) -> None:
    """Launch kernels without PDL (launch_pdl/gdc_wait). The default PDL launch lets
    back-to-back identical kernels in the bench graph overlap launch tails, reporting
    a faster per-call time than an e2e nsys duration, which includes the gdc_wait
    stall on a DIFFERENT (often slower) producer kernel. --no-pdl gives the
    standalone-execution number for comparing against e2e profile durations."""
    import sglang.srt.lora.triton_ops.kernel_utils as _ku

    def no_pdl():
        return False, {}

    _ku.get_pdl_launch_metadata = no_pdl
    for mod in modules:
        mod.get_pdl_launch_metadata = no_pdl
    globals()["get_pdl_launch_metadata"] = no_pdl


def production_stage_config(proj: str, num_input_tokens: int) -> dict:
    """Mirror of ``_get_shrink_stage_config`` (virtual_experts.py) for the decode
    regime: BLOCK_SIZE_M=16; warps 4 unless the LoRA-A N < 32 (then 2); stages 3.
    SPLIT_K is intentionally not set -- the launcher derives it from the grid
    (``_get_moe_lora_shrink_split_k``), exactly as production does."""
    n = PROJ[proj]["n"]
    if num_input_tokens < 512:
        return {
            "BLOCK_SIZE_M": 16,
            "num_warps": 4 if (num_input_tokens <= 4 or n >= 32) else 2,
            "num_stages": 3,
        }
    return {"BLOCK_SIZE_M": 32, "num_warps": 2, "num_stages": 2}


def make_routing_inputs(bs, ep, device, seed=0, routing="uniform", skew_a=0.9):
    """Top-k routing over the GLOBAL expert ids + the all-zero (single adapter, slot 0)
    token->lora mapping. ``routing="skewed"`` draws experts from a Zipf(skew_a)
    popularity (hot experts), which reproduces the e2e num_tokens_post_padded range
    (measured decode bs64: 832..1312, median 1056) that uniform routing undershoots."""
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
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    return topk_ids, token_lora_mapping


def build_ep_routing(topk_ids, token_lora_mapping, ep, block_m):
    """Production ``_get_routing`` replica (non-fused path, max_loras=1, EP local mask):
    virtual ids (non-owned -> -1) -> moe_align_block_size -> tight trim with
    ``local_num_experts * max_loras + 1`` populated buckets. Reproduces the e2e decode
    buffers sorted_token_ids [1488] / expert_ids [93] at bs=64 block_m=16."""
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
    """hidden_states is per-token for gate_up, per-(token, expert) for down. Weight is
    the GLOBAL [num_experts, N, K] LoRA-A (the [max_loras=1, E, N, K] buffer pre-merged,
    as production's free-view reshape produces)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    spec = PROJ[proj]
    rows = bs if spec["input_top_k"] > 1 else bs * ep["top_k"]
    hidden = (
        torch.randn(rows, spec["k"], generator=gen, device=device, dtype=dtype) * 0.1
    )
    weight = (
        torch.randn(
            ep["num_experts"],
            spec["n"],
            spec["k"],
            generator=gen,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    out = torch.zeros(bs * ep["top_k"], spec["n"], device=device, dtype=dtype)
    return hidden, weight, out


def _invoke_shrink_forced(
    hidden_states,
    weight,
    output,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    top_k,
    config,
    split_k,
):
    """Mirror of ``_invoke_moe_lora_shrink_splitk`` with an explicit SPLIT_K (the
    production launcher derives it and exposes no override; the previous bench's
    FORCE_SPLIT_K config key is silently ignored by the current launcher)."""
    N = weight.shape[1]
    K = weight.shape[2]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_K = 256
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)
    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)
    grid = (split_k * num_m_blocks,)
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _moe_lora_shrink_splitk_kernel[grid](
        hidden_states,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        topk_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=split_k,
        ENABLE_PDL=enable_pdl,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 4),
        **pdl_kwargs,
    )


def shrink_call(proj, hidden, weight, out, topk_ids, routing, config):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    split_k = config.get("FORCE_SPLIT_K")
    if split_k is not None:
        return lambda: _invoke_shrink_forced(
            hidden,
            weight,
            out,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            PROJ[proj]["input_top_k"],
            config,
            split_k,
        )
    return lambda: _invoke_moe_lora_shrink_splitk(
        hidden,
        weight,
        out,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        PROJ[proj]["input_top_k"],
        config,
    )


def ref_shrink(proj, hidden, weight, topk_ids, ep):
    """fp32 reference: per owned (token, expert) pair only; non-owned rows stay zero
    (production relies on the pre-zeroed split-K intermediate for those)."""
    bs, top_k = topk_ids.shape
    n = weight.shape[1]
    lo = ep["local_expert_offset"]
    hi = lo + ep["local_num_experts"]
    w = weight.float()
    out = torch.zeros(bs * top_k, n, device=hidden.device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            if not (lo <= e < hi):
                continue
            vt = m * top_k + k
            row = hidden[m] if PROJ[proj]["input_top_k"] > 1 else hidden[vt]
            out[vt] = row.float() @ w[e].t()
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
    timing (atomic-add throughput is value-independent); correctness mode re-zeroes."""

    def fn():
        for call in calls:
            call()

    fn()  # eager warmup: triton JIT compile outside graph capture
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(fn, rep=rep_ms)
    return float(ms) * 1e3 / len(calls)


def touched_bytes_of(proj, topk_ids, ep) -> int:
    """Bytes the kernel actually reads/writes per call: only OWNED routed (token,
    expert) pairs touch weight rows / hidden rows / output rows (the EP -1 sentinel
    blocks early-return before any load). Sizing rotation by the full global weight
    (4x larger) silently under-rotates and lets L2 serve the hot expert rows."""
    spec = PROJ[proj]
    lo = ep["local_expert_offset"]
    owned_mask = (topk_ids >= lo) & (topk_ids < lo + ep["local_num_experts"])
    owned_pairs = int(owned_mask.sum().item())
    unique_owned = int(topk_ids[owned_mask.bool()].unique().numel())
    if spec["input_top_k"] > 1:
        hidden_touched = topk_ids.shape[0] * spec["k"]  # per-token rows, all read
    else:
        hidden_touched = owned_pairs * spec["k"]  # per-(token,expert) rows
    weight_touched = unique_owned * spec["n"] * spec["k"]
    out_touched = owned_pairs * spec["n"]
    return 2 * (hidden_touched + weight_touched + out_touched)


def build_rotated_calls(args, proj, ep, config, device, dtype):
    topk_ids, tlm = make_routing_inputs(
        args.bs, ep, device, seed=args.seed, routing=args.routing, skew_a=args.skew_a
    )
    routing = build_ep_routing(topk_ids, tlm, ep, config["BLOCK_SIZE_M"])
    spec = PROJ[proj]
    group_bytes = touched_bytes_of(proj, topk_ids, ep)
    num_groups = args.num_groups or auto_num_groups(
        group_bytes, args.l2_mult, args.min_groups, args.max_groups
    )
    groups = [
        make_gemm_inputs(proj, args.bs, ep, args.rank, dtype, device, seed=g)
        for g in range(num_groups)
    ]
    calls = [
        shrink_call(proj, hidden, weight, out, topk_ids, routing, config)
        for hidden, weight, out in groups
    ]
    return calls, routing, num_groups, group_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["bench", "correctness", "profile", "sweep", "sweepk"],
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
        "--no-pdl", action="store_true", help="disable PDL (see disable_pdl docstring)"
    )
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
    if args.no_pdl:
        import sglang.srt.lora.triton_ops.virtual_experts as _ve

        disable_pdl([_ve])

    device = "cuda"
    dtype = torch.bfloat16
    ep = QWEN35_EP4
    projs = list(PROJ) if args.proj == "all" else [args.proj]

    if args.mode == "correctness":
        # Also guard the routing/tiling block-size contract: the launcher must tile
        # expert_ids with the same BLOCK_SIZE_M the routing was aligned with (one entry
        # per M-block); sweeping block_m catches a hardcoded launcher block size.
        failures = 0
        for proj in projs:
            for bs in sorted({args.bs, 16, 64}):
                topk_ids, tlm = make_routing_inputs(bs, ep, device, seed=args.seed)
                hidden, weight, out = make_gemm_inputs(
                    proj, bs, ep, args.rank, dtype, device
                )
                ref = ref_shrink(proj, hidden, weight, topk_ids, ep)
                for block_m in [16, 32, 64]:
                    config = {
                        **production_stage_config(proj, bs),
                        "BLOCK_SIZE_M": block_m,
                    }
                    routing = build_ep_routing(topk_ids, tlm, ep, block_m)
                    out.zero_()
                    shrink_call(proj, hidden, weight, out, topk_ids, routing, config)()
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
        config = production_stage_config(proj, args.bs)

        if args.mode == "sweepk":
            for sk in [1, 2, 3, 4, 5, 6, 7, 8]:
                cfg = {**config, "FORCE_SPLIT_K": sk}
                calls, routing, num_groups, _ = build_rotated_calls(
                    args, proj, ep, cfg, device, dtype
                )
                try:
                    us = bench_us_rotated(calls, args.rep_ms)
                except Exception as e:
                    print(f"  proj={proj} SPLIT_K={sk}: ERROR {e}")
                    continue
                print(f"  proj={proj} SPLIT_K={sk}: {us:7.2f} us")
            continue

        if args.mode == "sweep":
            best = None
            for block_m in [16, 32, 64]:
                for nw in [2, 4, 8]:
                    for ns in [2, 3, 4]:
                        for sk in [1, 2, 4, 5, 8]:
                            cfg = {
                                "BLOCK_SIZE_M": block_m,
                                "num_warps": nw,
                                "num_stages": ns,
                                "FORCE_SPLIT_K": sk,
                            }
                            calls, _, _, _ = build_rotated_calls(
                                args, proj, ep, cfg, device, dtype
                            )
                            try:
                                us = bench_us_rotated(calls, args.rep_ms)
                            except Exception:
                                continue
                            tag = (
                                f"block_m={block_m} warps={nw} stages={ns} split_k={sk}"
                            )
                            if best is None or us < best[0]:
                                best = (us, tag)
                            print(f"  {us:7.2f} us  proj={proj} {tag}")
            print(f"\nBEST proj={proj} bs={args.bs}: {best[0]:.2f} us  {best[1]}")
            continue

        calls, routing, num_groups, group_bytes = build_rotated_calls(
            args, proj, ep, config, device, dtype
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
        # Echo the launch geometry the production launcher derives, for cross-checking
        # against the e2e capture (gate_up: SPLIT_K=5 grid 465; down: SPLIT_K=2 grid 186).
        weight_like = torch.empty(
            ep["num_experts"], spec["n"], spec["k"], device="meta"
        )
        split_k = _get_moe_lora_shrink_split_k(weight_like, sorted_token_ids, config)
        num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], config["BLOCK_SIZE_M"])
        us = bench_us_rotated(calls, args.rep_ms)
        print(
            f"BENCH moe_shrink proj={proj} bs={args.bs} r={args.rank} N={spec['n']} "
            f"K={spec['k']} E={ep['num_experts']}(local {ep['local_num_experts']}) "
            f"sorted={sorted_token_ids.shape[0]} padded={padded} SPLIT_K={split_k} "
            f"grid=({split_k * num_m_blocks},) groups={num_groups} "
            f"({group_bytes * num_groups / 1e6:.0f} MB touched rotated): {us:.2f} us"
        )


if __name__ == "__main__":
    main()
