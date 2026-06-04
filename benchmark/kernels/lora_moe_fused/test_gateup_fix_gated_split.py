"""Fix validation: does the gated split (up output reads up_A shrink [R:2R]) make the
up half match PEFT ground truth?

Builds the full 2R-wide shrink intermediate ([0:R]=x@gate_A^T, [R:2R]=x@up_A^T) from the
REAL qwen35 adapter, then runs the rank-specialized expand kernel TWICE:
  - gated_a_half=0      (current production: up reads gate_A shrink)  -> expect up MISMATCH
  - gated_a_half=N//2   (proposed fix: up reads up_A shrink)          -> expect up MATCH
and compares both halves to PEFT (gate=x@gA^T@gB^T, up=x@uA^T@uB^T).

  python3 test_gateup_fix_gated_split.py [--experts 64] [--bs 16]
"""

from __future__ import annotations

import argparse

import torch
import triton
from safetensors import safe_open

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import _fused_virtual_topk_ids
from sglang.srt.lora.trtllm_moe.specialized_expand import _moe_lora_expand_add_kernel

TOP_K = 8
RANK = 16
K = 2048
INTER = 512
N = 2 * INTER


def build_routing(topk_ids, tlm, num_experts, local_e, block_m):
    vt, _, vne = _fused_virtual_topk_ids(
        topk_ids, tlm, num_experts, shared_outer=False, max_loras=1,
        local_expert_offset=0, local_num_experts=local_e,
    )
    sti, eid, ntp = moe_align_block_size(vt, block_m, vne)
    num_tokens = topk_ids.numel()
    populated = local_e + 1
    max_nonempty = min(num_tokens, populated)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    return sti[:tight], eid[: tight // block_m], ntp


def run_expand(intermediate, weight, output, topk_weights, topk_ids, routing, gated_a_half, block_n=128):
    sti, eid, ntp = routing
    block_m = 16
    R = weight.shape[2]
    grid = (triton.cdiv(sti.shape[0], block_m) * triton.cdiv(N, block_n),)
    output.zero_()
    _moe_lora_expand_add_kernel[grid](
        intermediate, weight, output, topk_weights, sti, eid, ntp,
        N, R, topk_ids.numel(),
        intermediate.stride(0), intermediate.stride(1),
        weight.stride(0), weight.stride(1), weight.stride(2),
        output.stride(-2), output.stride(-1),
        router_topk=topk_ids.shape[1],
        MUL_ROUTED_WEIGHT=False, FUSE_SUM_ALL_REDUCE=False,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_R=triton.next_power_of_2(R),
        GROUP_SIZE_M=1, GATED_A_HALF=gated_a_half, num_warps=4, num_stages=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/data/qwen35_35b_lora_alpha/adapter_model.safetensors")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0)
    args = ap.parse_args()
    device, dtype, E = "cuda", torch.bfloat16, args.experts

    f = safe_open(args.adapter, framework="pt")
    gA = torch.empty(E, RANK, K, dtype=dtype); uA = torch.empty(E, RANK, K, dtype=dtype)
    gB = torch.empty(E, INTER, RANK, dtype=dtype); uB = torch.empty(E, INTER, RANK, dtype=dtype)
    for e in range(E):
        p = f"base_model.model.model.layers.{args.layer}.mlp.experts.{e}"
        gA[e] = f.get_tensor(f"{p}.gate_proj.lora_A.weight")
        uA[e] = f.get_tensor(f"{p}.up_proj.lora_A.weight")
        gB[e] = f.get_tensor(f"{p}.gate_proj.lora_B.weight")
        uB[e] = f.get_tensor(f"{p}.up_proj.lora_B.weight")
    gA, uA, gB, uB = (t.to(device) for t in (gA, uA, gB, uB))

    # lora_b expand weight [E, 2*INTER, R] = [gate_B; up_B] * scaling
    weight = torch.empty(E, N, RANK, device=device, dtype=dtype)
    weight[:, 0:INTER, :] = gB * args.scaling
    weight[:, INTER:N, :] = uB * args.scaling

    torch.manual_seed(0)
    bs = args.bs
    hidden = torch.randn(bs, K, device=device, dtype=dtype) * 0.1
    scores = torch.rand(bs, E, device=device)
    topk_ids = torch.topk(scores, k=TOP_K, dim=1).indices.to(torch.int32)
    topk_weights = torch.ones(bs, TOP_K, device=device, dtype=torch.float32)
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    routing = build_routing(topk_ids, tlm, E, E, 16)

    # Full 2R-wide shrink intermediate: [0:R]=x@gate_A^T, [R:2R]=x@up_A^T, per (token, slot).
    inter = torch.zeros(bs * TOP_K, 2 * RANK, device=device, dtype=dtype)
    for m in range(bs):
        for k in range(TOP_K):
            e = int(topk_ids[m, k].item())
            vt = m * TOP_K + k
            inter[vt, 0:RANK] = (hidden[m].float() @ gA[e].float().t()).to(dtype)
            inter[vt, RANK:2 * RANK] = (hidden[m].float() @ uA[e].float().t()).to(dtype)

    # PEFT ground truth
    xf = hidden.float()
    def verdict(out, tag):
        ge = up = gd = ud = 0.0
        for m in range(bs):
            for k in range(TOP_K):
                e = int(topk_ids[m, k].item())
                gref = (xf[m] @ gA[e].float().t()) @ (gB[e].float() * args.scaling).t()
                uref = (xf[m] @ uA[e].float().t()) @ (uB[e].float() * args.scaling).t()
                got = out[m, k].float()
                ge = max(ge, (got[:INTER] - gref).abs().max().item()); gd = max(gd, gref.abs().max().item())
                up = max(up, (got[INTER:] - uref).abs().max().item()); ud = max(ud, uref.abs().max().item())
        print(f"  [{tag}] GATE rel={ge/(gd+1e-9):.2%}  UP rel={up/(ud+1e-9):.2%}  "
              f"-> GATE {'MATCH' if ge/(gd+1e-9)<0.05 else 'MISMATCH'} | UP {'MATCH' if up/(ud+1e-9)<0.05 else 'MISMATCH'}")

    out = torch.zeros(bs, TOP_K, N, device=device, dtype=dtype)
    print(f"experts={E} bs={bs} scaling={args.scaling}")
    run_expand(inter, weight, out, topk_weights, topk_ids, routing, gated_a_half=0)
    verdict(out, "gated_a_half=0 (current prod)")
    run_expand(inter, weight, out, topk_weights, topk_ids, routing, gated_a_half=INTER)
    verdict(out, f"gated_a_half={INTER} (proposed fix)")


if __name__ == "__main__":
    main()
