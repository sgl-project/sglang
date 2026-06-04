"""Definitive check: does the real sglang MoE gate_up LoRA dispatch match PEFT ground truth?

Loads the REAL qwen35 adapter's per-expert gate_proj / up_proj LoRA (gate_A != up_A),
builds the stacked buffers exactly as mem_pool does (lora_a [E,2R,K] = [gate_A; up_A],
lora_b [E,2*inter,R] = [gate_B; up_B]), runs the production
``merged_experts_fused_moe_lora_add`` and compares both output halves against the PEFT
reference:  gate_out = x @ gate_A^T @ gate_B^T ;  up_out = x @ up_A^T @ up_B^T.

If the up half diverges while the gate half matches, the expand is reading the gate
shrink for the up output (the suspected bug).

  python3 test_gateup_peft_groundtruth.py [--adapter /data/qwen35_35b_lora_alpha] [--layer 0] [--experts 64]
"""

from __future__ import annotations

import argparse

import torch
from safetensors import safe_open

from sglang.srt.lora.triton_ops.virtual_experts import merged_experts_fused_moe_lora_add

TOP_K = 8
RANK = 16
K = 2048
INTER = 512  # gate/up output dim each; full gate_up N = 2*INTER


def load_expert_lora(f, layer, e):
    p = f"base_model.model.model.layers.{layer}.mlp.experts.{e}"
    gA = f.get_tensor(f"{p}.gate_proj.lora_A.weight")  # [R,K]
    uA = f.get_tensor(f"{p}.up_proj.lora_A.weight")
    gB = f.get_tensor(f"{p}.gate_proj.lora_B.weight")  # [inter,R]
    uB = f.get_tensor(f"{p}.up_proj.lora_B.weight")
    return gA, uA, gB, uB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/data/qwen35_35b_lora_alpha/adapter_model.safetensors")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=64, help="num owned experts to build")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0, help="adapter scaling (alpha/r); applied to lora_b like prod")
    args = ap.parse_args()
    device, dtype = "cuda", torch.bfloat16
    E = args.experts

    f = safe_open(args.adapter, framework="pt")
    gA = torch.empty(E, RANK, K, dtype=dtype)
    uA = torch.empty(E, RANK, K, dtype=dtype)
    gB = torch.empty(E, INTER, RANK, dtype=dtype)
    uB = torch.empty(E, INTER, RANK, dtype=dtype)
    for e in range(E):
        a_g, a_u, b_g, b_u = load_expert_lora(f, args.layer, e)
        gA[e], uA[e], gB[e], uB[e] = a_g, a_u, b_g, b_u
    gA, uA, gB, uB = (t.to(device) for t in (gA, uA, gB, uB))

    # Stacked buffers exactly as mem_pool builds them.
    # lora_a [1, E, 2R, K]: [0:R]=gate_A, [R:2R]=up_A
    lora_a = torch.empty(1, E, 2 * RANK, K, device=device, dtype=dtype)
    lora_a[0, :, 0:RANK, :] = gA
    lora_a[0, :, RANK : 2 * RANK, :] = uA
    # lora_b [1, E, 2*INTER, R]: [0:INTER]=gate_B, [INTER:2*INTER]=up_B (scaling folded in, as prod)
    lora_b = torch.empty(1, E, 2 * INTER, RANK, device=device, dtype=dtype)
    lora_b[0, :, 0:INTER, :] = gB * args.scaling
    lora_b[0, :, INTER : 2 * INTER, :] = uB * args.scaling

    torch.manual_seed(0)
    bs = args.bs
    hidden = (torch.randn(bs, K, device=device, dtype=dtype) * 0.1)
    # route every token's top-k to owned experts [0:E) so all are exercised.
    scores = torch.rand(bs, E, device=device)
    topk_ids = torch.topk(scores, k=TOP_K, dim=1).indices.to(torch.int32)
    topk_weights = torch.ones(bs, TOP_K, device=device, dtype=torch.float32)
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)

    output = torch.zeros(bs, TOP_K, 2 * INTER, device=device, dtype=dtype)
    merged_experts_fused_moe_lora_add(
        output, hidden, lora_a, lora_b, topk_ids, topk_weights, tlm,
        False, False, False,
        fuse_add_to_output=False, fuse_sum_all_reduce=False,
        use_direct_expand_add=True, local_expert_offset=0, local_num_experts=E,
    )

    # PEFT ground truth per (token, slot).
    gAf, uAf, gBf, uBf = gA.float(), uA.float(), (gB * args.scaling).float(), (uB * args.scaling).float()
    xf = hidden.float()
    gate_err = up_err = gate_den = up_den = 0.0
    for m in range(bs):
        for k in range(TOP_K):
            e = int(topk_ids[m, k].item())
            gate_ref = (xf[m] @ gAf[e].t()) @ gBf[e].t()  # [INTER]
            up_ref = (xf[m] @ uAf[e].t()) @ uBf[e].t()
            got = output[m, k].float()
            gate_err = max(gate_err, (got[:INTER] - gate_ref).abs().max().item())
            up_err = max(up_err, (got[INTER:] - up_ref).abs().max().item())
            gate_den = max(gate_den, gate_ref.abs().max().item())
            up_den = max(up_den, up_ref.abs().max().item())
    print(f"experts={E} bs={bs} scaling={args.scaling}")
    print(f"  GATE half: max_abs_err={gate_err:.3e}  ref_max={gate_den:.3e}  rel={gate_err/(gate_den+1e-9):.2%}")
    print(f"  UP   half: max_abs_err={up_err:.3e}  ref_max={up_den:.3e}  rel={up_err/(up_den+1e-9):.2%}")
    # Relative-error verdict (these tensors are tiny in magnitude, so an absolute
    # tolerance falsely passes; a wrong low-rank delta still has rel >> 1).
    gate_rel = gate_err / (gate_den + 1e-9)
    up_rel = up_err / (up_den + 1e-9)
    gate_ok = gate_rel < 0.05
    up_ok = up_rel < 0.05
    print(f"  GATE {'MATCH' if gate_ok else 'MISMATCH'} | UP {'MATCH' if up_ok else 'MISMATCH'} vs PEFT")
    if gate_ok and not up_ok:
        print("  => CONFIRMED BUG: gate half correct, up half wrong (up output uses gate_A shrink).")


if __name__ == "__main__":
    main()
