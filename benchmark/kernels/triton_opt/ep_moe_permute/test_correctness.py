"""Standalone correctness test for the optimized deepep_permute_triton_kernel.

Tests the optimized config (BLOCK_SIZE=1024, num_warps=8) against PyTorch reference
at multiple sizes including edge cases.
"""
import os
import sys

import torch
import triton
import triton.language as tl

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_3"


@triton.jit
def deepep_permute_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    src_ptr = input_ptr + src_idx * hidden_size

    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)

        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


def pytorch_reference(input_tensor, src2dst, topk, hidden_size, out_dtype):
    num_tokens = input_tensor.shape[0]
    max_dst = int(src2dst.max().item()) + 1 if (src2dst >= 0).any() else 1
    out = torch.zeros(max_dst, hidden_size, device=input_tensor.device, dtype=out_dtype)
    for t in range(num_tokens):
        for k in range(topk):
            dst_idx = src2dst[t, k].item()
            if dst_idx >= 0:
                out[dst_idx] = input_tensor[t].to(out_dtype)
    return out


def run_test(num_tokens, hidden_size, topk, device="cuda:0"):
    input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    src2dst = torch.arange(num_tokens * topk, device=device, dtype=torch.int64).reshape(num_tokens, topk)
    # Mark ~10% as invalid
    invalid_mask = torch.rand(num_tokens, topk, device=device) < 0.1
    src2dst[invalid_mask] = -1
    max_dst = int(src2dst.max().item()) + 1
    topk_ids = torch.randint(0, 64, (num_tokens, topk), device=device, dtype=torch.int64)

    # Reference
    ref = pytorch_reference(input_tensor, src2dst, topk, hidden_size, torch.bfloat16)

    # Triton with optimized config
    gateup_input = torch.zeros(max_dst, hidden_size, device=device, dtype=torch.bfloat16)
    grid = (num_tokens,)
    deepep_permute_triton_kernel[grid](
        input_tensor, gateup_input, src2dst, topk_ids, None,
        topk, hidden_size,
        BLOCK_SIZE=1024,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # Compare valid entries
    valid_dst_indices = src2dst[src2dst >= 0].long()
    if len(valid_dst_indices) == 0:
        return True

    triton_out = gateup_input[valid_dst_indices]
    ref_out = ref[valid_dst_indices]

    exact = (triton_out.view(torch.uint8) == ref_out.view(torch.uint8)).float().mean().item()
    max_diff = (triton_out.float() - ref_out.float()).abs().max().item()

    if exact < 0.99:
        print(f"  FAIL: exact={exact:.4f}, max_diff={max_diff}")
        return False
    return True


def main():
    test_cases = [
        # (num_tokens, hidden_size, topk)
        (1, 2048, 2),
        (1, 4096, 6),
        (64, 2048, 2),
        (64, 4096, 6),
        (64, 7168, 8),
        (256, 2048, 2),
        (256, 4096, 6),
        (256, 7168, 8),
        (1024, 2048, 2),
        (1024, 4096, 6),
        (1024, 7168, 8),
        (4096, 2048, 2),
        (4096, 4096, 6),
        (4096, 7168, 8),
        # Edge: hidden_size not a multiple of BLOCK_SIZE
        (64, 3000, 4),
        (256, 5000, 6),
    ]

    all_passed = True
    for nt, hs, tk in test_cases:
        ok = run_test(nt, hs, tk)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] num_tokens={nt}, hidden_size={hs}, topk={tk}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll correctness tests PASSED.")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
