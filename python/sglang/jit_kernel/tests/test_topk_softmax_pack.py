"""Bitwise unit test: JIT topk_softmax_pack vs AOT topk_softmax + triton fused_pack_topk.

Run on a GPU pod with the lora-fused-topk-pack branch installed:
    python3 test_topk_softmax_pack.py
"""
import torch

from sgl_kernel import topk_softmax  # AOT reference
from sglang.jit_kernel.flashinfer_trtllm_moe.topk_pack import fused_pack_topk  # ref pack
from sglang.jit_kernel.topk_softmax_pack import topk_softmax_pack  # JIT under test

torch.manual_seed(0)
dev = "cuda"
fails = 0

def case(num_tokens, num_experts, k, dtype, renormalize, ntnp_val):
    global fails
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    # ref: AOT kernel
    w_ref = torch.empty(num_tokens, k, dtype=torch.float32, device=dev)
    i_ref = torch.empty(num_tokens, k, dtype=torch.int32, device=dev)
    topk_softmax(w_ref, i_ref, gating.clone(), renormalize)
    # jit: fused pack
    w_jit = torch.empty_like(w_ref)
    i_jit = torch.empty_like(i_ref)
    packed = torch.empty(num_tokens, k, dtype=torch.int32, device=dev)
    ntnp = (
        torch.tensor([ntnp_val], dtype=torch.int32, device=dev)
        if ntnp_val is not None
        else None
    )
    topk_softmax_pack(w_jit, i_jit, packed, gating, renormalize, num_token_non_padded=ntnp)
    torch.cuda.synchronize()
    # reference packed: mask ids like _mask_topk_ids_padded_region, then triton pack
    ids_masked = i_ref.clone()
    if ntnp_val is not None:
        ids_masked[ntnp_val:, :] = -1
    p_ref = fused_pack_topk(ids_masked, w_ref)
    torch.cuda.synchronize()
    ok_w = torch.equal(w_jit, w_ref)
    ok_i = torch.equal(i_jit, i_ref)
    ok_p = torch.equal(packed, p_ref)
    tag = f"M={num_tokens} E={num_experts} k={k} {dtype} renorm={renormalize} ntnp={ntnp_val}"
    if ok_w and ok_i and ok_p:
        print(f"PASS  {tag}")
    else:
        fails += 1
        print(f"FAIL  {tag}  w={ok_w} i={ok_i} p={ok_p}")
        if not ok_p:
            bad = (packed != p_ref).nonzero()[:5]
            for b in bad:
                r, c = b.tolist()
                print(f"    [{r},{c}] jit={packed[r,c].item():#010x} ref={p_ref[r,c].item():#010x} "
                      f"id_ref={ids_masked[r,c].item()} w_ref={w_ref[r,c].item():.6f}")

# Qwen3.5 shape: 256 experts, k=8, bf16 router logits
for M in (1, 64, 333):
    for renorm in (True, False):
        case(M, 256, 8, torch.bfloat16, renorm, None)
case(64, 256, 8, torch.bfloat16, True, 40)     # padded region
case(64, 256, 8, torch.bfloat16, True, 64)     # ntnp == M (no padding)
case(64, 256, 8, torch.bfloat16, True, 0)      # all padded
case(64, 256, 8, torch.float32, True, None)    # fp32 gating
case(64, 256, 8, torch.float16, True, None)    # fp16 gating
case(64, 128, 6, torch.bfloat16, True, 17)     # other pow2 + odd k
case(7, 512, 4, torch.bfloat16, True, None)    # 512 experts
case(64, 8, 2, torch.bfloat16, True, None)     # tiny experts

print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILURES")
raise SystemExit(1 if fails else 0)
