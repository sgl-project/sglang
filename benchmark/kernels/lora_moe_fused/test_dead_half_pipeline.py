"""Production-pipeline correctness for the gate_up shrink dead-half opt.

Drives the real ``merged_experts_fused_moe_lora_add`` dispatch on the gate_up MoE-LoRA
shapes with SGLANG_OPT_LORA_MOE_SHRINK_DEAD_HALF OFF vs ON and asserts the final output
is numerically identical (atomic-add / split-K reassociation noise floor). This is the
authoritative guard for the wiring in virtual_experts.py: the dead-half must not change
the model-visible result.

gate_up case: lora_a [1, E, 2*rank, K], lora_b [1, E, 2*inter, rank], single adapter,
EP4 (64 owned of 256), mul_routed_weight=False, fuse_sum_all_reduce=False,
use_direct_expand_add=True.

  python3 test_dead_half_pipeline.py
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.lora.triton_ops.virtual_experts import (
    merged_experts_fused_moe_lora_add,
)

E = 256
LOCAL_E = 64
TOP_K = 8
K = 2048
INTER = 512  # moe intermediate; gate_up output N = 2*INTER = 1024
RANK = 16


def build(bs, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(bs, E, generator=gen, device=device)
    topk_ids = torch.topk(scores, k=TOP_K, dim=1).indices.to(torch.int32)
    topk_weights = torch.rand(bs, TOP_K, generator=gen, device=device, dtype=torch.float32) * 0.9 + 0.1
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    hidden = torch.randn(bs, K, generator=gen, device=device, dtype=dtype) * 0.1
    # gate_up: lora_a rank-dim = 2*RANK (gate_A; up_A merged), lora_b N = 2*INTER, rank RANK.
    lora_a = torch.randn(1, E, 2 * RANK, K, generator=gen, device=device, dtype=dtype) * 0.1
    lora_b = torch.randn(1, E, 2 * INTER, RANK, generator=gen, device=device, dtype=dtype) * 0.1
    return topk_ids, topk_weights, tlm, hidden, lora_a, lora_b


def run_once(hidden, lora_a, lora_b, topk_ids, topk_weights, tlm, bs, device, dtype):
    output = torch.zeros(bs, TOP_K, 2 * INTER, device=device, dtype=dtype)
    merged_experts_fused_moe_lora_add(
        output,
        hidden,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        tlm,
        False,  # mul_routed_weight
        False,  # experts_shared_outer_loras_a
        False,  # experts_shared_outer_loras_b
        fuse_add_to_output=False,
        fuse_sum_all_reduce=False,
        use_direct_expand_add=True,
        local_expert_offset=0,
        local_num_experts=LOCAL_E,
    )
    return output


def main():
    device, dtype, bs = "cuda", torch.bfloat16, 64
    fails = 0
    for seed in range(4):
        topk_ids, topk_weights, tlm, hidden, lora_a, lora_b = build(bs, dtype, device, seed=seed)

        envs.SGLANG_OPT_LORA_MOE_SHRINK_DEAD_HALF.set(False)
        out_off = run_once(hidden, lora_a, lora_b, topk_ids, topk_weights, tlm, bs, device, dtype)

        envs.SGLANG_OPT_LORA_MOE_SHRINK_DEAD_HALF.set(True)
        out_on = run_once(hidden, lora_a, lora_b, topk_ids, topk_weights, tlm, bs, device, dtype)

        envs.SGLANG_OPT_LORA_MOE_SHRINK_DEAD_HALF.set(False)

        abs_err = (out_on.float() - out_off.float()).abs().max().item()
        denom = out_off.float().abs().max().item() + 1e-9
        rel = abs_err / denom
        ok = abs_err < 1e-2 or rel < 1e-2
        fails += int(not ok)
        print(f"seed={seed}: max_abs={abs_err:.4e} rel={rel:.4e} ref_max={denom:.3e} {'PASS' if ok else 'FAIL'}")
    if fails:
        raise SystemExit(f"{fails} dead-half pipeline mismatches")
    print("ALL PASS (dead-half pipeline numerically identical)")


if __name__ == "__main__":
    main()
