"""Self-contained micro-benchmark + correctness check for _moe_lora_expand_add_kernel
(LoRA-B expand-add, down-proj GEMM).

Companion to ``bench_shrink_splitk.py`` (which covers the LoRA-A shrink). The down-proj
expand uses the fused sum-all-reduce + routed-weight variant: each of a token's top_k
per-expert deltas is scaled by its routing weight and atomic-added into the single
per-token output row.

Two model shapes via ``--model`` (per-flag overrides win):
  * ``qwen35`` (default): qwen3.5-35b local-EP, tp=4/ep=4 -> 64 local experts, N (down
    output hidden) = 2048, rank = 16, top_k = 8.
  * ``kimi-k25``: Kimi-K2.5-NVFP4, TP8/no-EP -> 384 routed experts, N = 7168, rank = 16,
    top_k = 8. (Shapes from the adapter + config: down-proj LoRA-B is [r=16 x out=7168],
    384 routed experts.)

P0 scope: bs=64, rank=16 first.

  python3 bench_expand_add_down.py --mode bench   --model kimi-k25   # default: reproduces the
                                                                     #   PRODUCTION config (e2e kernel)
  python3 bench_expand_add_down.py --mode bench   --model kimi-k25 --config manual --block-m 16 --block-n 512
  python3 bench_expand_add_down.py --mode correctness --model kimi-k25  # block_m {16,32,64} x bs {16,64}
  python3 bench_expand_add_down.py --mode profile --iters 2   # eager, for ncu
  python3 bench_expand_add_down.py --mode sweep   --model kimi-k25      # block_m x block_n x group_m x warps

By default ``--mode bench``/``profile`` use ``--config production``: the BLOCK_SIZE_M/GROUP_SIZE_M/
num_warps that ``_get_stage_config`` -> ``try_get_optimal_moe_config`` picks at runtime (BLOCK_SIZE_N
left to the launcher), so the headline number matches the e2e kernel. ``--config manual`` uses the
explicit ``--block-*`` / ``--group-m`` / ``--num-warps`` flags instead (for A/B and tuning).

``--mode sweep`` also tunes BLOCK_SIZE_N (the launcher's production default forces 128 for
N%128==0; the sweep passes ``force_block_size_n`` to explore other tiles — valid for the
non-gated down-proj).

correctness mode also guards the routing/tiling block-size contract: the launcher must
tile ``expert_ids`` with the same block size the routing buffers were aligned with, else
expert_ids overruns -> IMA (the same class of bug as the shrink f2adddd regression).
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
    fused_sanitize_expert_ids,
)
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add


def production_config(num_experts, n, rank, bs, dtype):
    """The config production actually launches the expand with, so the default bench run
    reproduces the e2e kernel (not a hand-picked one).

    Mirrors ``_get_stage_config`` (virtual_experts.py): call
    ``try_get_optimal_moe_config(lora_b_virtual.shape, .., stage_top_k=1, M=bs)`` and, if it
    raises (e.g. no tuned JSON + server args unset), use the same fallback (BLOCK_SIZE_M=64).
    BLOCK_SIZE_N is left to the launcher (forced to 128 when N % 128 == 0), so ``force_block_n``
    stays None. Returns ``(block_m, group_m, num_warps)``.
    """
    import functools

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
        # _get_stage_config's fallback when try_get_optimal_moe_config raises.
        return 64, 1, 4


def make_inputs(bs, num_experts, top_k, n, rank, dtype, device):
    """Down-proj expand inputs.

    ``intermediate`` is the LoRA-A shrink output [bs*top_k, rank]; ``lora_b`` is the
    per-(lora, expert) down LoRA-B weight [1, num_experts, N, rank].
    """
    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    # Positive routing weights (down-proj scales each per-expert delta by these).
    topk_weights = torch.rand(bs, top_k, device=device, dtype=torch.float32) * 0.9 + 0.1
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    intermediate = torch.randn(bs * top_k, rank, device=device, dtype=dtype) * 0.1
    lora_b = torch.randn(1, num_experts, n, rank, device=device, dtype=dtype) * 0.1
    return topk_ids, topk_weights, token_lora_mapping, intermediate, lora_b


def build_v1_routing(topk_ids, token_lora_mapping, num_experts, block_m):
    """Single-adapter (max_loras=1) virtual-expert routing, tiled at ``block_m``.

    Mirrors ``_get_routing`` in virtual_experts.py: virtual topk ids -> align ->
    tight trim -> sanitize. The trim+sanitize matter because the launcher reads one
    ``expert_ids`` entry per M-block.
    """
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


def expand(
    intermediate,
    lora_b,
    topk_ids,
    topk_weights,
    routing,
    block_m,
    block_n=64,
    group_m=1,
    num_warps=4,
    mul_routed_weight=True,
    fuse_sum_all_reduce=True,
    force_block_n=None,
):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    lora_b_virtual = lora_b.reshape(lora_b.shape[0] * lora_b.shape[1], *lora_b.shape[2:])
    n = lora_b.shape[2]
    bs, top_k = topk_ids.shape
    # FUSE_SUM_ALL_REDUCE atomic-adds the top_k deltas into one row -> zero each call.
    # (Without it, each (token,expert) slot is written once -> bs*top_k rows.)
    out_rows = bs if fuse_sum_all_reduce else bs * top_k
    output = torch.zeros(
        out_rows, n, dtype=intermediate.dtype, device=intermediate.device
    )
    config = {
        # BLOCK_SIZE_N here is only used when N % 128 != 0 AND force_block_n is None;
        # otherwise the launcher picks 128 (default) or force_block_n (tuning).
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
    }
    _invoke_moe_lora_expand_add(
        intermediate,
        lora_b_virtual,
        output,
        # kernel indexes topk_weights flat by virtual-token id in [0, bs*top_k).
        topk_weights.reshape(-1),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight,
        fuse_sum_all_reduce,
        force_block_size_n=force_block_n,
    )
    return output


def ref_expand(
    intermediate, lora_b, topk_ids, topk_weights,
    mul_routed_weight=True, fuse_sum_all_reduce=True,
):
    bs, top_k = topk_ids.shape
    n = lora_b.shape[2]
    b = lora_b[0].float()  # [num_experts, N, R]
    inter = intermediate.float()  # [bs*top_k, R]
    rows = bs if fuse_sum_all_reduce else bs * top_k
    out = torch.zeros(rows, n, device=intermediate.device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            vt = m * top_k + k
            delta = inter[vt] @ b[e].t()  # [N]
            if mul_routed_weight:
                delta = delta * float(topk_weights[m, k].item())
            if fuse_sum_all_reduce:
                out[m] += delta
            else:
                out[vt] = delta
    return out


def bench_ms(fn, warmup=25, rep=100, cudagraph=True, inner=200):
    """Per-call milliseconds.

    With ``cudagraph``, capture ``inner`` back-to-back ``fn()`` calls in ONE graph and
    divide the measured replay time by ``inner``. A single fn()-per-graph
    ``do_bench(g.replay)`` floors at ~8-10us for ANY tiny op -- it measures the fixed
    per-replay launch/dispatch overhead, not the kernel. Amortizing over ``inner``
    back-to-back calls drives that overhead to ~0 and exposes the true device time.
    (Same technique as bench_shrink_splitk.py.)
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
    ap.add_argument(
        "--mode",
        choices=["bench", "correctness", "profile", "sweep"],
        default="bench",
    )
    # Model presets set (num_experts, n, top_k); per-flag overrides below win.
    ap.add_argument("--model", choices=["qwen35", "kimi-k25"], default="qwen35")
    # bench/profile config source: "production" reproduces what _get_stage_config launches
    # (the e2e kernel); "manual" uses the --block-m/--block-n/--group-m/--num-warps flags.
    ap.add_argument("--config", choices=["production", "manual"], default="production")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="down-proj output hidden")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=None,
                    help="force BLOCK_SIZE_N (overrides the launcher's N%%128==0 -> 128 rule)")
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    # qwen35: tp4/ep4 -> 64 local experts, N=2048. kimi-k25: TP8/no-EP -> 384 experts, N=7168.
    preset = {
        "qwen35": {"num_experts": 64, "n": 2048, "top_k": 8},
        "kimi-k25": {"num_experts": 384, "n": 7168, "top_k": 8},
    }[args.model]
    if args.num_experts is None:
        args.num_experts = preset["num_experts"]
    if args.n is None:
        args.n = preset["n"]
    if args.top_k is None:
        args.top_k = preset["top_k"]

    dev = "cuda"

    if args.mode == "sweep":
        # P0: bs=64, rank=16. Tunable knobs: BLOCK_SIZE_M / BLOCK_SIZE_N / GROUP_SIZE_M /
        # num_warps (num_stages is hardcoded to 1 in the launcher). BLOCK_SIZE_N is swept via
        # force_block_size_n (the production path forces 128 for N%128==0); only divisors of N
        # are tried so no tile is wasted.
        topk_ids, tkw, tlm, inter, lora_b = make_inputs(
            args.bs, args.num_experts, args.top_k, args.n, args.rank,
            torch.bfloat16, dev,
        )
        block_ns = [bn for bn in [64, 128, 256, 512] if args.n % bn == 0]
        best = None
        for block_m in [16, 32, 64]:
            routing = build_v1_routing(topk_ids, tlm, args.num_experts, block_m)
            for block_n in block_ns:
                for group_m in [1, 4, 8]:
                    for nw in [2, 4, 8]:
                        f = lambda bm=block_m, bn=block_n, gm=group_m, w=nw, r=routing: expand(
                            inter, lora_b, topk_ids, tkw, r, bm,
                            group_m=gm, num_warps=w, force_block_n=bn,
                        )
                        try:
                            us = bench_ms(f, warmup=15, rep=60) * 1000
                        except Exception:
                            continue
                        tag = f"block_m={block_m} block_n={block_n} group_m={group_m} warps={nw}"
                        if best is None or us < best[0]:
                            best = (us, tag)
                        print(f"  {us:7.2f} us  {tag}")
        print(f"\nBEST bs={args.bs} r={args.rank} N={args.n} experts={args.num_experts}: "
              f"{best[0]:.2f} us  {best[1]}")
        return

    topk_ids, tkw, tlm, inter, lora_b = make_inputs(
        args.bs, args.num_experts, args.top_k, args.n, args.rank,
        torch.bfloat16, dev,
    )
    # bench/profile: by default reproduce the config production launches (so the headline
    # number matches the e2e kernel); --config manual uses the explicit flags instead.
    if args.config == "production":
        eff_block_m, eff_group_m, eff_warps = production_config(
            args.num_experts, args.n, args.rank, args.bs, torch.bfloat16
        )
        eff_force_block_n = None  # let the launcher force 128 (N % 128 == 0), as production does
    else:
        eff_block_m, eff_group_m, eff_warps = args.block_m, args.group_m, args.num_warps
        eff_force_block_n = args.block_n
    routing = build_v1_routing(topk_ids, tlm, args.num_experts, eff_block_m)
    fn = lambda: expand(
        inter, lora_b, topk_ids, tkw, routing, eff_block_m,
        group_m=eff_group_m, num_warps=eff_warps, force_block_n=eff_force_block_n,
    )

    if args.mode == "correctness":
        # Guard the routing/tiling block-size contract: _invoke_moe_lora_expand_add tiles
        # expert_ids with config["BLOCK_SIZE_M"] (one entry per M-block); routing must be
        # aligned with the SAME block. Build routing AND config with the same block_m per
        # iteration, but SWEEP block_m so any hardcoded value diverges from the routing at
        # the other block sizes; sweep bs too so the overrun is deterministic enough to
        # fault. Reference accumulates in fp32, kernel accumulates per-expert bf16
        # atomic-adds -> generous abs tol.
        block_ms = sorted({args.block_m, 16, 32, 64})
        batch_sizes = sorted({args.bs, 16, 64})
        failures = 0
        for bs in batch_sizes:
            tk, tkw_b, tlm_b, inter_b, lb = make_inputs(
                bs, args.num_experts, args.top_k, args.n, args.rank,
                torch.bfloat16, dev,
            )
            ref = ref_expand(inter_b, lb, tk, tkw_b)
            for bm in block_ms:
                routing_bm = build_v1_routing(tk, tlm_b, args.num_experts, bm)
                out = expand(
                    inter_b, lb, tk, tkw_b, routing_bm, bm,
                    group_m=args.group_m, num_warps=args.num_warps,
                    force_block_n=args.block_n,
                ).float()
                err = float((out - ref).abs().max().item())
                rel = err / float(ref.abs().max().item() + 1e-9)
                ok = err <= args.tol
                failures += int(not ok)
                print(
                    f"{'PASS' if ok else 'FAIL'} bs={bs:<3d} block_m={bm:<2d} "
                    f"max_abs_err={err:.4e} rel={rel:.2e}"
                )
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
        eff_block_n = eff_force_block_n if eff_force_block_n is not None else (
            128 if args.n % 128 == 0 else args.block_n
        )
        print(
            f"BENCH expand_add down model={args.model} config={args.config} bs={args.bs} "
            f"r={args.rank} N={args.n} experts={args.num_experts} top_k={args.top_k} "
            f"block_m={eff_block_m} block_n={eff_block_n} group_m={eff_group_m} "
            f"warps={eff_warps}: {ms * 1000:.2f} us (amortized true device time)"
        )


if __name__ == "__main__":
    main()
