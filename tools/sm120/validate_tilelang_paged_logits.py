#!/usr/bin/env python3
"""
Validate the SM120 fix for tilelang_fp8_paged_mqa_logits.

Run this ON THE RTX PRO 6000 box (needs CUDA + the patched tilelang + sglang).
It compiles the tilelang paged-MQA-logits kernel and compares its output against
the pure-torch reference (fp8_paged_mqa_logits_torch_sm120), which is the ground
truth. If they match within fp8 tolerance, the tilelang fix is correct.

    python validate_tilelang_paged_logits.py

Exit code 0 = PASS, 1 = FAIL/compile-error.
"""
import sys
import torch

FP8 = torch.float8_e4m3fn

# Shapes matching GLM-5.2 / DeepSeek DSA indexer.
BATCH = 8          # query rows (decode: one per request)
NUM_HEADS = 32     # index_n_heads
HEAD_DIM = 128     # index_head_dim
BLOCK = 64         # page_size
MAX_SEQ = 4096     # > index_topk so the kernel path is exercised
NUM_PAGES = BATCH * (MAX_SEQ // BLOCK) + 8


def build_inputs(device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(BATCH, 1, NUM_HEADS, HEAD_DIM, device=device).to(FP8)
    weight = torch.randn(BATCH, NUM_HEADS, device=device, dtype=torch.float32)
    # KV cache: [num_pages, BLOCK, 1, HEAD_DIM + 4]  (128 fp8 vals + 1 fp32 scale)
    head_dim_sf = HEAD_DIM + 4
    kv = torch.zeros(NUM_PAGES, BLOCK, 1, head_dim_sf, device=device, dtype=torch.uint8)
    kv[..., :HEAD_DIM] = torch.randint(
        0, 255, (NUM_PAGES, BLOCK, 1, HEAD_DIM), device=device, dtype=torch.uint8
    )
    scales = (torch.rand(NUM_PAGES, BLOCK, 1, 1, device=device) + 0.1).to(torch.float32)
    kv[..., HEAD_DIM:] = scales.view(torch.uint8)
    seq_lens = torch.full((BATCH,), MAX_SEQ - 7, device=device, dtype=torch.int32)
    max_pages = (MAX_SEQ + BLOCK - 1) // BLOCK
    page_table = torch.arange(
        BATCH * max_pages, device=device, dtype=torch.int32
    ).view(BATCH, max_pages) % NUM_PAGES
    return q, kv, weight, seq_lens, page_table


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); return 1
    major, minor = torch.cuda.get_device_capability()
    print(f"Device cc: sm_{major}{minor}  ({torch.cuda.get_device_name()})")

    from sglang.srt.layers.attention.dsa.tilelang_kernel import (
        tilelang_fp8_paged_mqa_logits,
    )
    from sglang.srt.layers.attention.dsv4.indexer import (
        fp8_paged_mqa_logits_torch_sm120,
    )

    q, kv, weight, seq_lens, page_table = build_inputs()

    # Reference (ground truth) — pure torch.
    ref = fp8_paged_mqa_logits_torch_sm120(
        q, kv, weight, seq_lens, page_table, None, MAX_SEQ, clean_logits=False
    )

    # Tilelang kernel under test — this is what must compile + match on SM120.
    try:
        out = tilelang_fp8_paged_mqa_logits(
            q, kv, weight, seq_lens, page_table, None, MAX_SEQ, clean_logits=False
        )
    except Exception as e:  # noqa: BLE001
        print("TILELANG COMPILE/RUN FAILED:")
        print(repr(e))
        return 1

    # Compare only the valid (unmasked) region per row: [0, seq_len).
    ok = True
    max_abs = 0.0
    for b in range(BATCH):
        n = int(seq_lens[b].item())
        a = out[b, :n].float()
        r = ref[b, :n].float()
        d = (a - r).abs().max().item()
        max_abs = max(max_abs, d)
        # fp8 inputs => loose tolerance; structure/scale must match.
        if not torch.allclose(a, r, atol=1e-2, rtol=1e-2):
            ok = False
    print(f"max abs diff (valid region) = {max_abs:.4g}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
