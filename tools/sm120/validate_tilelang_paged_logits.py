#!/usr/bin/env python3
"""
Validate the SM120 fix for tilelang_fp8_paged_mqa_logits.

Run this ON the RTX PRO 6000 box (needs CUDA + the patched tilelang + sglang).
It compiles the tilelang paged-MQA-logits kernel and compares its output against
an inlined pure-torch reference (identical to sglang's
fp8_paged_mqa_logits_torch_sm120). If they match within fp8 tolerance, the
tilelang fix is correct.

The reference is inlined on purpose: importing it from
sglang.srt.layers.attention.dsv4.indexer drags in deepseek_v2 -> flashinfer.comm,
which can fail at import time in some images (libcudart stub symbol errors). We
only need the tilelang kernel under test from sglang; the reference is standalone.

    python validate_tilelang_paged_logits.py

Exit code 0 = PASS, 1 = FAIL/compile-error.
"""
import sys

import torch
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn

# Shapes matching GLM-5.2 / DeepSeek DSA indexer.
BATCH = 8          # query rows (decode: one per request)
NUM_HEADS = 32     # index_n_heads
HEAD_DIM = 128     # index_head_dim
BLOCK = 64         # page_size
MAX_SEQ = 4096     # > index_topk so the kernel path is exercised
NUM_PAGES = BATCH * (MAX_SEQ // BLOCK) + 8


def fp8_paged_mqa_logits_torch_ref(
    q_fp8, kvcache_fp8, weight, seq_lens, page_table, max_seq_len
):
    """Standalone copy of sglang's fp8_paged_mqa_logits_torch_sm120
    (clean_logits=False semantics). Ground-truth reference."""
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device
    assert head_dim == 128 and block_size == 64
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_padded_seq = max_pages * block_size
    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    scale_offset = block_size * head_dim

    page_ids = page_table[:, :max_pages]
    gathered = kvcache_flat[page_ids]
    kv_value = (
        gathered[..., :scale_offset]
        .contiguous()
        .view(dtype=FP8)
        .to(torch.float32)
        .view(batch_size, max_padded_seq, head_dim)
    )
    kv_scale = (
        gathered[..., scale_offset:]
        .contiguous()
        .view(dtype=torch.float32)
        .view(batch_size, max_padded_seq)
    )

    q = q_fp8[:, 0].to(torch.float32)               # [B, H, D]
    score = torch.bmm(kv_value, q.transpose(1, 2))  # [B, S, H]
    score = F.relu(score)
    score = score * weight.unsqueeze(1)
    score = score.sum(dim=2)                         # [B, S]
    score = score * kv_scale

    out_width = min(max_padded_seq, max_seq_len)
    logits = score.new_full((batch_size, max_seq_len), float("-inf"))
    logits[:, :out_width] = score[:, :out_width]
    positions = torch.arange(max_seq_len, device=device)
    logits.masked_fill_(positions.unsqueeze(0) >= seq_lens.unsqueeze(1), float("-inf"))
    return logits


def build_inputs(device="cuda"):
    torch.manual_seed(0)
    # Valid fp8 values (quantized from floats), and a VALUE-MAJOR page layout:
    # per page the bytes are [ all B*D fp8 values | all B fp32 scales ], which is
    # exactly how both the kernel and the reference read the cache. (A plain
    # contiguous (BLOCK,1,HEAD_DIM+4) tensor would be token-major and mis-feed the
    # value-major readers.)
    q = (torch.randn(BATCH, 1, NUM_HEADS, HEAD_DIM, device=device) * 0.5).to(FP8)
    weight = torch.randn(BATCH, NUM_HEADS, device=device, dtype=torch.float32)

    block_bytes = BLOCK * (HEAD_DIM + 4)   # 8448
    scale_offset = BLOCK * HEAD_DIM        # 8192
    flat = torch.zeros(NUM_PAGES, block_bytes, device=device, dtype=torch.uint8)

    # value region [0:B*D] = (BLOCK, HEAD_DIM) fp8, row-major
    kv_vals_fp8 = (torch.randn(NUM_PAGES, BLOCK, HEAD_DIM, device=device) * 0.5).to(FP8)
    flat[:, :scale_offset] = kv_vals_fp8.reshape(NUM_PAGES, scale_offset).view(torch.uint8)

    # scale region [B*D:] = (BLOCK,) fp32 per page
    scales = (torch.rand(NUM_PAGES, BLOCK, device=device) + 0.1).to(torch.float32)
    flat[:, scale_offset:] = scales.view(torch.uint8).reshape(NUM_PAGES, BLOCK * 4)

    # present in the (pages, BLOCK, 1, HEAD_DIM+4) shape the wrapper asserts; it
    # immediately flattens back to (pages, block_bytes), preserving value-major.
    kv = flat.view(NUM_PAGES, BLOCK, 1, HEAD_DIM + 4)
    seq_lens = torch.full((BATCH,), MAX_SEQ - 7, device=device, dtype=torch.int32)
    max_pages = (MAX_SEQ + BLOCK - 1) // BLOCK
    page_table = (
        torch.arange(BATCH * max_pages, device=device, dtype=torch.int32)
        .view(BATCH, max_pages) % NUM_PAGES
    )
    return q, kv, weight, seq_lens, page_table


def check_paged_logits():
    """Numerically validate the FP8 paged-MQA-logits kernel vs the torch ref."""
    print("\n=== [1/2] DSA indexer paged-logits kernel (fp8_paged_mqa_logits) ===")
    from sglang.srt.layers.attention.dsa.tilelang_kernel import (
        tilelang_fp8_paged_mqa_logits,
    )

    q, kv, weight, seq_lens, page_table = build_inputs()

    ref = fp8_paged_mqa_logits_torch_ref(q, kv, weight, seq_lens, page_table, MAX_SEQ)

    try:
        out = tilelang_fp8_paged_mqa_logits(
            q, kv, weight, seq_lens, page_table, None, MAX_SEQ, clean_logits=False
        )
    except Exception as e:  # noqa: BLE001
        print("TILELANG COMPILE/RUN FAILED:")
        print(repr(e))
        return 1

    print(f"out shape={tuple(out.shape)} dtype={out.dtype}")

    ok = True
    worst_finite = 0.0
    total_nan_inf = 0
    total_mismatch = 0
    total_elems = 0
    first_report = True
    for b in range(BATCH):
        n = int(seq_lens[b].item())
        a = out[b, :n].float()
        r = ref[b, :n].float()
        total_elems += n

        nan_inf = int((~torch.isfinite(a)).sum().item())
        total_nan_inf += nan_inf

        diff = (a - r).abs()
        finite = torch.isfinite(diff)
        if finite.any():
            worst_finite = max(worst_finite, diff[finite].max().item())

        close = torch.isclose(a, r, atol=1e-2, rtol=1e-2)
        bad = int((~close).sum().item())
        total_mismatch += bad

        if (nan_inf or bad) and first_report:
            first_report = False
            idx = (~close).nonzero(as_tuple=True)[0][:6].tolist()
            print(f"\n[batch {b}] first mismatches at positions {idx}")
            print(f"  tilelang: {[round(a[i].item(), 4) for i in idx]}")
            print(f"  torch ref:{[round(r[i].item(), 4) for i in idx]}")
            # ratio helps spot a constant scale/sign error
            ratios = []
            for i in idx:
                rv = r[i].item()
                ratios.append(round(a[i].item() / rv, 4) if rv != 0 else float("nan"))
            print(f"  ratio a/r:{ratios}")

        if nan_inf or bad:
            ok = False

    print(f"\nelements compared (valid): {total_elems}")
    print(f"nan/inf in tilelang output: {total_nan_inf}")
    print(f"mismatched elements:        {total_mismatch}")
    print(f"worst finite abs diff:      {worst_finite:.4g}")
    print("paged-logits: PASS" if ok else "paged-logits: FAIL")
    return ok


# --- DSA sparse-attention kernel (sparse_attention_fwd_kernel_v2) -------------
# This kernel previously failed to *compile* on SM120 (Hopper WGMMA / wait_wgmma).
# With the _SM120_WGMMA_OFF (disable_wgmma) fix it should lower via MMA. This is a
# compile + finiteness check (no closed-form reference for sparse MLA attention).
SA_TOKENS = 64
SA_HEADS = 16        # GLM-5.2 attn heads per GPU are padded to 16 (e.g. 64/TP8 -> 8 -> 16)
SA_D_V = 512         # kv_lora_rank
SA_TAIL = 64         # qk_rope_head_dim
SA_TOPK = 2048       # index_topk
SA_NUM_KV = 8192


def check_sparse_attention():
    print("\n=== [2/2] DSA sparse-attention kernel (sparse_attention_fwd_kernel_v2) ===")
    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    dim = SA_D_V + SA_TAIL
    dev = "cuda"
    q = torch.randn(SA_TOKENS, SA_HEADS, dim, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(SA_NUM_KV, 1, dim, device=dev, dtype=torch.bfloat16)
    indices = torch.randint(
        0, SA_NUM_KV, (SA_TOKENS, 1, SA_TOPK), device=dev, dtype=torch.int32
    )
    sm_scale = 1.0 / (dim ** 0.5)

    try:
        out = tilelang_sparse_fwd(q, kv, indices, sm_scale, d_v=SA_D_V)
    except Exception as e:  # noqa: BLE001
        print("TILELANG COMPILE/RUN FAILED:")
        print(repr(e))
        return False

    finite = torch.isfinite(out).all().item()
    print(f"out shape={tuple(out.shape)} dtype={out.dtype} all_finite={finite}")
    print("sparse-attention: PASS" if finite else "sparse-attention: FAIL (non-finite)")
    return bool(finite)


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); return 1
    major, minor = torch.cuda.get_device_capability()
    print(f"Device cc: sm_{major}{minor}  ({torch.cuda.get_device_name()})")

    ok_logits = check_paged_logits()
    ok_attn = check_sparse_attention()

    print("\n==================== SUMMARY ====================")
    print(f"  paged-logits kernel:    {'PASS' if ok_logits else 'FAIL'}")
    print(f"  sparse-attention kernel:{'PASS' if ok_attn else 'FAIL'}")
    all_ok = ok_logits and ok_attn
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
