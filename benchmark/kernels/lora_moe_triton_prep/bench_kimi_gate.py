"""Self-contained perf-bench + correctness-test for kimi_k2_moe_fused_gate
(sgl_kernel::kimi_k2_moe_fused_gate, the Kimi routing gate, ~5us), on the EP8
bs64 Kimi-K2.5-NVFP4 decode path.

Production decode shapes (SHAPE_REPORT.md, decode bs64):
  IN gating_output (64,384) f32, correction_bias (384,) f32, topk=8,
  num_expert_group=1, renormalize=True, routed_scaling_factor=2.827
  OUT topk_weights (64,8) f32, topk_ids (64,8) i32.

correctness: torch reference (sigmoid score + bias for selection, renormalized
sigmoid score as weight). The expert SELECTION is asserted exactly (bitwise) per
row; the WEIGHTS are asserted to match within a tiny numerical tolerance (the
kernel's sigmoid/renormalize is fp32 but not bit-identical to torch's).

Timing + cold-L2 buffer rotation: see common_bench.

Usage (on the GPU pod):
  python3 bench_kimi_gate.py --mode bench
  python3 bench_kimi_gate.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
from common_bench import bench_kernel, pick_n_sets, report_sets, set_bytes
from sgl_kernel import kimi_k2_moe_fused_gate


def make_input_set(bs, num_experts, device):
    return {
        "gating": torch.randn(bs, num_experts, device=device, dtype=torch.float32),
        "bias": torch.randn(num_experts, device=device, dtype=torch.float32) * 0.1,
    }


def gate(s, topk, renormalize, rsf):
    """kimi_k2_moe_fused_gate, mirroring the biased_grouped_topk_gpu call site."""
    return kimi_k2_moe_fused_gate(
        s["gating"],
        s["bias"],
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=rsf,
        apply_routed_scaling_factor_on_output=False,
    )


def ref_gate(gating, bias, topk, renormalize):
    """fp reference: sigmoid score, select top-k by (score+bias), weight = score at
    selected, renormalized. (num_expert_group=1 -> no real grouping; rsf is NOT
    folded in because apply_routed_scaling_factor_on_output=False.)"""
    scores = torch.sigmoid(gating.float())
    idx = torch.topk(scores + bias.float(), topk, dim=-1).indices
    w = torch.gather(scores, -1, idx)
    if renormalize:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness"], default="bench")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--rsf", type=float, default=2.827)
    ap.add_argument("--budget-gb", type=float, default=16.0)
    ap.add_argument("--n-sets", type=int, default=0, help="0 = auto (fill --budget-gb)")
    args = ap.parse_args()
    dev = "cuda"
    mk = lambda: make_input_set(args.bs, args.num_experts, dev)

    if args.mode == "correctness":
        s = mk()
        w, ids = gate(s, args.topk, True, args.rsf)
        rw, rids = ref_gate(s["gating"], s["bias"], args.topk, True)
        # PAIRED check: sort each row by expert id, then compare. If the id sets match the
        # sorted-id tensors are bitwise-equal, AND the per-position weights are then paired to
        # their own expert id -> this catches a kernel that returns right ids + right weight
        # multiset but the WRONG id<->weight pairing (a real MoE bug a set/sorted check misses).
        ki, ri = torch.argsort(ids, -1), torch.argsort(rids.to(torch.int32), -1)
        ids_s = torch.gather(ids, -1, ki)
        rids_s = torch.gather(rids.to(torch.int32), -1, ri)
        w_s = torch.gather(w, -1, ki)
        rw_s = torch.gather(rw, -1, ri)
        id_ok = bool(torch.equal(ids_s, rids_s))  # expert ids bitwise (as sorted sets)
        werr = float((w_s - rw_s).abs().max().item())  # weights PAIRED to their ids
        ok = id_ok and werr <= 1e-3
        print(
            f"{'PASS' if ok else 'FAIL'} gate: expert_ids_bitwise={id_ok} "
            f"paired_max_weight_err={werr:.4e} (tol 1e-3)"
        )
        raise SystemExit(0 if ok else 1)

    per = set_bytes(mk())
    n_sets = pick_n_sets(per, args.budget_gb, args.n_sets)
    S = [mk() for _ in range(n_sets)]
    call = lambda i: gate(S[i], args.topk, True, args.rsf)
    us = bench_kernel(call, n_sets) * 1000
    print(
        f"BENCH kimi_k2_moe_fused_gate bs={args.bs} experts={args.num_experts} topk={args.topk}"
    )
    print(f"  per_set={per/1e3:.1f}KB {report_sets(per, n_sets)}")
    print(f"  kimi_k2_moe_fused_gate        = {us:7.2f} us")


if __name__ == "__main__":
    main()
