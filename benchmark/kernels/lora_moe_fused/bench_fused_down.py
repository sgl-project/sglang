"""Standalone dev bench for a FUSED MoE LoRA *down* kernel (shrink + expand in one launch).

Baseline (2 kernels): _moe_lora_shrink_splitk (down) -> intermediate[512,16] in HBM ->
_moe_lora_expand_add (down, mul_routed_weight + fuse_sum_all_reduce) -> output[64,2048].

Fused (1 kernel): per owned (token,expert) m-block, compute the rank-16 shrink
s = act @ A_down[e]^T in registers, then expand delta = s @ B_down[e]^T over the full
N=2048, scale by routed weight, atomic-add into the token row. The rank-16 intermediate
never touches HBM and one kernel launch + one routing reuse are removed.

Shapes reproduce Qwen3.5-35B-A3B-FP8 tp4/ep4 decode bs64, single rank-16 adapter
(SHAPECAP 2026-06-04), matching benchmark/kernels/lora_csgmv (shrink) +
benchmark/kernels/lora_moe_expand (expand) exactly:
  act [512,512] (per-(token,expert)), A_down [256,16,512], B_down [256,2048,16],
  topk_ids [64,8], output [64,2048]. EP: 64 owned experts of 256, non-owned -> -1.

  python3 bench_fused_down.py --mode correctness
  python3 bench_fused_down.py --mode bench           # fused vs 2-kernel baseline
  python3 bench_fused_down.py --mode profile --iters 4   # for ncu
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.kernel_utils import get_pdl_launch_metadata
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    _invoke_moe_lora_shrink_splitk,
)
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add

QWEN35_EP4 = {
    "num_experts": 256,
    "local_num_experts": 64,
    "local_expert_offset": 0,
    "top_k": 8,
}
# down: act per-(token,expert) [bs*top_k, K], A_down [E, R, K], B_down [E, N, R].
SPEC = {"n": 2048, "k": 512, "rank": 16}


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_moe_lora_down_kernel(
    act_ptr,  # [num_tokens(=bs*top_k), K]
    a_ptr,  # [E, R, K]  LoRA-A (down)
    b_ptr,  # [E, N, R]  LoRA-B (down)
    out_ptr,  # [bs, N]   atomic-add target (the real MoE output)
    topk_weights_ptr,  # [bs*top_k]
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_a_e,
    stride_a_r,
    stride_a_k,
    stride_b_e,
    stride_b_n,
    stride_b_r,
    stride_om,
    stride_on,
    router_topk: tl.constexpr,
    R: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_TILES_PER_BLOCK: tl.constexpr,
):
    """Fused rank-R shrink+expand for the LoRA down projection (1-adapter EP decode).

    Grid = (num_m_blocks, num_n_blocks). Each program owns one expert m-block and a
    span of ``N_TILES_PER_BLOCK`` consecutive N tiles. The rank-R shrink s[BLOCK_M, R]
    is computed ONCE (in registers) and reused across this program's N tiles, then each
    tile is expanded and atomic-added (weighted) into the token row. With
    N_TILES_PER_BLOCK = ceil(N/BLOCK_N) and num_n_blocks=1 the shrink is computed once
    per expert (no redundancy) while the internal N loop gives software-pipelined
    memory-level parallelism to hide DRAM latency at the ~93-block (under one wave) grid.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_r = tl.arange(0, R).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # ---- shrink (once): s[BLOCK_M, R] = act[offs_token, :] @ A_down[e]^T ----
    a_act_ptrs = act_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_w_ptrs = (
        a_ptr
        + off_expert * stride_a_e
        + (offs_k[:, None] * stride_a_k + offs_r[None, :] * stride_a_r)
    )
    s = tl.zeros((BLOCK_SIZE_M, R), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k0 * BLOCK_SIZE_K
        kmask = offs_k < k_remaining
        act = tl.load(
            a_act_ptrs, mask=token_mask[:, None] & kmask[None, :], other=0.0
        )
        aw = tl.load(a_w_ptrs, mask=kmask[:, None], other=0.0)
        s += tl.dot(act, aw.to(act.dtype))
        a_act_ptrs += BLOCK_SIZE_K * stride_ak
        a_w_ptrs += BLOCK_SIZE_K * stride_a_k
    s = s.to(b_ptr.dtype.element_ty)

    moe_w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    offs_out = offs_token // router_topk

    # ---- expand: loop this program's N tiles, reusing s ----
    for t in range(0, N_TILES_PER_BLOCK):
        n_base = (pid_n * N_TILES_PER_BLOCK + t) * BLOCK_SIZE_N
        offs_n = n_base + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        n_mask = offs_n < N
        b_ptrs = (
            b_ptr
            + off_expert * stride_b_e
            + (offs_r[:, None] * stride_b_r + offs_n[None, :] * stride_b_n)
        )
        bw = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        delta = tl.dot(s, bw, out_dtype=tl.float32)
        delta = delta * moe_w[:, None]
        out_ptrs = out_ptr + offs_out[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = token_mask[:, None] & n_mask[None, :]
        tl.atomic_add(out_ptrs, delta.to(out_ptr.dtype.element_ty), mask=out_mask)


def invoke_fused_down(
    act,
    a_down,
    b_down,
    output,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    n_split: int = 0,
):
    N = b_down.shape[1]
    K = a_down.shape[2]
    R = a_down.shape[1]
    block_k = min(256, K)
    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], block_m)
    total_n_tiles = triton.cdiv(N, block_n)
    # n_split = number of grid blocks along N; each handles total_n_tiles/n_split tiles
    # internally (shrink computed once per grid block, reused across its tiles).
    # n_split=0 -> one grid block per expert (shrink once, full internal N loop).
    num_n_blocks = n_split if n_split > 0 else 1
    n_tiles_per_block = triton.cdiv(total_n_tiles, num_n_blocks)
    grid = (num_m_blocks, num_n_blocks)
    _fused_moe_lora_down_kernel[grid](
        act,
        a_down,
        b_down,
        output,
        topk_weights.reshape(-1),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        topk_ids.numel(),
        act.stride(0),
        act.stride(1),
        a_down.stride(0),
        a_down.stride(1),
        a_down.stride(2),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        output.stride(0),
        output.stride(1),
        router_topk=topk_ids.shape[1],
        R=R,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        N_TILES_PER_BLOCK=n_tiles_per_block,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ---------------------------------------------------------------------------
# Inputs / routing (mirror the two production benches)
# ---------------------------------------------------------------------------
def make_inputs(bs, ep, dtype, device, seed=0, routing="uniform", skew_a=0.9):
    gen = torch.Generator(device=device).manual_seed(seed)
    if routing == "skewed":
        pop = torch.arange(1, ep["num_experts"] + 1, dtype=torch.float32, device=device).pow(-skew_a)
        pop = pop[torch.randperm(ep["num_experts"], generator=gen, device=device)]
        topk_ids = torch.multinomial(
            pop.expand(bs, -1), ep["top_k"], replacement=False, generator=gen
        ).to(torch.int32)
    else:
        scores = torch.rand(bs, ep["num_experts"], generator=gen, device=device)
        topk_ids = torch.topk(scores, k=ep["top_k"], dim=1).indices.to(torch.int32)
    topk_weights = (
        torch.rand(bs, ep["top_k"], generator=gen, device=device, dtype=torch.float32) * 0.9 + 0.1
    )
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    act = torch.randn(bs * ep["top_k"], SPEC["k"], generator=gen, device=device, dtype=dtype) * 0.1
    a_down = torch.randn(ep["num_experts"], SPEC["rank"], SPEC["k"], generator=gen, device=device, dtype=dtype) * 0.1
    b_down = torch.randn(ep["num_experts"], SPEC["n"], SPEC["rank"], generator=gen, device=device, dtype=dtype) * 0.1
    return topk_ids, topk_weights, tlm, act, a_down, b_down


def build_ep_routing(topk_ids, tlm, ep, block_m):
    max_loras = 1
    virtual_topk_ids, _, vne = _fused_virtual_topk_ids(
        topk_ids, tlm, ep["num_experts"], shared_outer=False, max_loras=max_loras,
        local_expert_offset=ep["local_expert_offset"], local_num_experts=ep["local_num_experts"],
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        virtual_topk_ids, block_m, vne
    )
    num_tokens = topk_ids.numel()
    populated = ep["local_num_experts"] * max_loras + 1
    max_nonempty = min(num_tokens, populated)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    return sorted_token_ids[:tight], expert_ids[: tight // block_m], num_tokens_post_padded


# ---------------------------------------------------------------------------
# Reference (composed: fp32 shrink then fp32 expand-add)
# ---------------------------------------------------------------------------
def ref_fused_down(act, a_down, b_down, topk_ids, topk_weights, ep):
    bs, top_k = topk_ids.shape
    lo, hi = ep["local_expert_offset"], ep["local_expert_offset"] + ep["local_num_experts"]
    A = a_down.float()
    B = b_down.float()
    out = torch.zeros(bs, SPEC["n"], device=act.device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            if not (lo <= e < hi):
                continue
            vt = m * top_k + k
            s = act[vt].float() @ A[e].t()           # [R]
            delta = s @ B[e].t()                       # [N]
            out[m] += float(topk_weights[m, k].item()) * delta
    return out


# ---------------------------------------------------------------------------
# Baseline (2-kernel) invocation
# ---------------------------------------------------------------------------
def baseline_two_kernel_call(act, a_down, b_down, output, topk_weights, topk_ids, routing):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    a_cfg = {"BLOCK_SIZE_M": 16, "num_warps": 2, "num_stages": 3}
    b_cfg = {"BLOCK_SIZE_M": 16, "num_warps": 4, "num_stages": 1, "GROUP_SIZE_M": 1, "BLOCK_SIZE_N": 64}
    inter = torch.zeros(act.shape[0], SPEC["rank"], device=act.device, dtype=act.dtype)

    def run():
        inter.zero_()
        _invoke_moe_lora_shrink_splitk(
            act, a_down, inter, topk_ids, sorted_token_ids, expert_ids,
            num_tokens_post_padded, 1, a_cfg,
        )
        _invoke_moe_lora_expand_add(
            inter, b_down, output, topk_weights, topk_ids, sorted_token_ids,
            expert_ids, num_tokens_post_padded, b_cfg, True, True,
        )

    return run


def auto_num_groups(group_bytes, l2_mult, min_g, max_g):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    return max(min_g, min(math.ceil(l2 * l2_mult / max(group_bytes, 1)), max_g))


def touched_bytes(topk_ids, ep):
    lo = ep["local_expert_offset"]
    owned = (topk_ids >= lo) & (topk_ids < lo + ep["local_num_experts"])
    pairs = int(owned.sum().item())
    uniq = int(topk_ids[owned.bool()].unique().numel())
    act_b = pairs * SPEC["k"]
    a_b = uniq * SPEC["rank"] * SPEC["k"]
    b_b = uniq * SPEC["n"] * SPEC["rank"]
    out_b = topk_ids.shape[0] * SPEC["n"]
    return 2 * (act_b + a_b + b_b + out_b)


def bench_call(call, rep_ms):
    call()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(call, rep=rep_ms)
    return float(ms) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "bench", "profile"], default="bench")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routing", choices=["uniform", "skewed"], default="skewed")
    ap.add_argument("--skew-a", type=float, default=0.9)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--n-split", type=int, default=0, help="grid blocks along N (0=shrink-once full internal N loop)")
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--rep-ms", type=int, default=400)
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--min-groups", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    ep = QWEN35_EP4

    if args.mode == "correctness":
        for seed in range(4):
            for routing in ("uniform", "skewed"):
                topk_ids, topk_weights, tlm, act, a_down, b_down = make_inputs(
                    args.bs, ep, dtype, device, seed=seed, routing=routing, skew_a=args.skew_a
                )
                routing_pack = build_ep_routing(topk_ids, tlm, ep, args.block_m)
                out = torch.zeros(args.bs, SPEC["n"], device=device, dtype=dtype)
                invoke_fused_down(
                    act, a_down, b_down, out, topk_weights, topk_ids, *routing_pack,
                    args.block_m, args.block_n, args.num_warps, args.num_stages, args.n_split,
                )
                ref = ref_fused_down(act, a_down, b_down, topk_ids, topk_weights, ep)
                got = out.float()
                abs_err = (got - ref).abs().max().item()
                rel = abs_err / (ref.abs().max().item() + 1e-9)
                ok = abs_err < 2e-2 or rel < 2e-2
                print(f"seed={seed} routing={routing}: max_abs={abs_err:.4e} rel={rel:.4e} {'PASS' if ok else 'FAIL'}")
                assert ok, "fused down correctness FAILED"
        print("ALL PASS")
        return

    # bench / profile: rotate L2-sized groups
    topk_ids0, _, _, _, _, _ = make_inputs(args.bs, ep, dtype, device, seed=args.seed, routing=args.routing, skew_a=args.skew_a)
    gbytes = touched_bytes(topk_ids0, ep)
    num_groups = auto_num_groups(gbytes, args.l2_mult, args.min_groups, args.max_groups)
    groups = [make_inputs(args.bs, ep, dtype, device, seed=g, routing=args.routing, skew_a=args.skew_a) for g in range(num_groups)]
    routings = [build_ep_routing(g[0], g[2], ep, args.block_m) for g in groups]
    outs = [torch.zeros(args.bs, SPEC["n"], device=device, dtype=dtype) for _ in range(num_groups)]

    fused_calls = []
    base_calls = []
    for (topk_ids, topk_weights, tlm, act, a_down, b_down), routing, out in zip(groups, routings, outs):
        fused_calls.append(
            (lambda act, a_down, b_down, out, topk_weights, topk_ids, routing: (
                lambda: invoke_fused_down(
                    act, a_down, b_down, out, topk_weights, topk_ids, *routing,
                    args.block_m, args.block_n, args.num_warps, args.num_stages, args.n_split,
                )
            ))(act, a_down, b_down, out, topk_weights, topk_ids, routing)
        )
        base_calls.append(baseline_two_kernel_call(act, a_down, b_down, out, topk_weights, topk_ids, routing))

    def run_all(calls):
        for c in calls:
            c()

    if args.mode == "profile":
        for _ in range(2):
            run_all(fused_calls)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            run_all(fused_calls)
        torch.cuda.synchronize()
        print(f"PROFILE fused down: {args.iters} x {num_groups} groups done")
        return

    fused_us = bench_call(lambda: run_all(fused_calls), args.rep_ms) / num_groups
    base_us = bench_call(lambda: run_all(base_calls), args.rep_ms) / num_groups
    pad = int(routings[0][2].item())
    print(
        f"BENCH down bs={args.bs} routing={args.routing} padded={pad} groups={num_groups} "
        f"block_m={args.block_m} block_n={args.block_n} warps={args.num_warps} stages={args.num_stages}"
    )
    print(f"  baseline (shrink+expand, 2 kernels): {base_us:.2f} us")
    print(f"  fused (1 kernel):                    {fused_us:.2f} us   speedup {base_us/fused_us:.2f}x")


if __name__ == "__main__":
    main()
