"""TQ4 CUDA native decode attention — Python wrapper + correctness test.

Compiles the CUDA kernel and provides a drop-in replacement for
_tq_decode_grouped_att_m_fwd's stage1 call.
"""

import os
import math
import torch
from torch.utils.cpp_extension import load

_module = None

def _get_module():
    global _module
    if _module is None:
        src_dir = os.path.join(os.path.dirname(__file__), "csrc", "tq_decode")
        _module = load(
            name="tq4_decode_cuda",
            sources=[os.path.join(src_dir, "tq4_decode_stage1.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90"],
            verbose=False,
        )
    return _module


def tq4_decode_stage1_cuda(
    q, k_packed, v_packed, k_dscale, v_dscale,
    k_centroids, v_centroids, att_out, att_lse,
    kv_indptr, kv_indices, num_kv_splits, max_kv_splits,
    sm_scale,
):
    """CUDA native TQ4 decode stage1 — drop-in replacement for Triton version."""
    mod = _get_module()
    mod.tq4_decode_stage1(
        q, k_packed, v_packed, k_dscale, v_dscale,
        kv_indptr, kv_indices, num_kv_splits,
        att_out, att_lse,
        k_centroids, v_centroids,
        sm_scale, max_kv_splits,
    )


def test_correctness(batch=32, seq_len=512, q_heads=32, kv_heads=8, head_dim=128):
    """Compare CUDA kernel output against Triton kernel output."""
    import sys
    sys.path.insert(0, '/workspace/sglang/python')

    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        tq_decode_attention_fwd,
    )
    from sglang.srt.layers.quantization.kv_turboquant import build_codebook

    device = "cuda"
    total_tokens = batch * seq_len
    packed_dim = head_dim // 2
    max_kv_splits = 8

    torch.manual_seed(42)
    q = torch.randn(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_packed = torch.randint(0, 255, (total_tokens, kv_heads, packed_dim), dtype=torch.uint8, device=device)
    v_packed = torch.randint(0, 255, (total_tokens, kv_heads, packed_dim), dtype=torch.uint8, device=device)
    k_dscale = (torch.randn(total_tokens, kv_heads, dtype=torch.bfloat16, device=device).abs() + 0.1)
    v_dscale = (torch.randn(total_tokens, kv_heads, dtype=torch.bfloat16, device=device).abs() + 0.1)

    centroids, _ = build_codebook(4, head_dim)
    k_centroids = torch.tensor(centroids, dtype=torch.float32, device=device)
    v_centroids = torch.tensor(centroids, dtype=torch.float32, device=device)

    kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, total_tokens, dtype=torch.int32, device=device)
    num_kv_splits_t = torch.full((batch,), 8, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # --- Triton reference ---
    o_triton = torch.zeros(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    att_logits_triton = torch.zeros(batch, q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device)
    att_lse_triton = torch.zeros(batch, q_heads, max_kv_splits, dtype=torch.float32, device=device)

    tq_decode_attention_fwd(
        q, k_packed, v_packed, k_dscale, v_dscale,
        k_centroids, v_centroids, o_triton,
        kv_indptr, kv_indices, att_logits_triton, att_lse_triton,
        num_kv_splits_t, max_kv_splits, sm_scale,
        k_bit_width=4, v_bit_width=4,
    )

    # --- CUDA kernel ---
    att_logits_cuda = torch.zeros(batch, q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device)
    att_lse_cuda = torch.zeros(batch, q_heads, max_kv_splits, dtype=torch.float32, device=device)

    tq4_decode_stage1_cuda(
        q, k_packed, v_packed, k_dscale, v_dscale,
        k_centroids, v_centroids,
        att_logits_cuda, att_lse_cuda,
        kv_indptr, kv_indices, num_kv_splits_t, max_kv_splits,
        sm_scale,
    )

    # Compare
    valid_mask = att_lse_triton > -1e30
    if valid_mask.any():
        diff_logits = (att_logits_triton - att_logits_cuda).abs()
        diff_lse = (att_lse_triton - att_lse_cuda).abs()
        max_diff_logits = diff_logits[valid_mask.unsqueeze(-1).expand_as(diff_logits)].max().item()
        max_diff_lse = diff_lse[valid_mask].max().item()
        mean_diff_logits = diff_logits[valid_mask.unsqueeze(-1).expand_as(diff_logits)].mean().item()
    else:
        max_diff_logits = max_diff_lse = mean_diff_logits = 0

    print(f"=== Correctness Test (batch={batch}, seq_len={seq_len}) ===")
    print(f"  att_logits max diff:  {max_diff_logits:.6f}")
    print(f"  att_logits mean diff: {mean_diff_logits:.6f}")
    print(f"  att_lse max diff:     {max_diff_lse:.6f}")
    print(f"  valid splits: {valid_mask.sum().item()}/{valid_mask.numel()}")
    ok = max_diff_logits < 0.01 and max_diff_lse < 0.1
    print(f"  PASS: {ok}")
    return ok


def benchmark(batch=32, seq_len=1024, q_heads=32, kv_heads=8, head_dim=128,
              warmup=10, repeat=100):
    """Timing comparison: CUDA vs Triton."""
    import sys
    sys.path.insert(0, '/workspace/sglang/python')

    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        tq_decode_attention_fwd,
    )
    from sglang.srt.layers.quantization.kv_turboquant import build_codebook

    device = "cuda"
    total_tokens = batch * seq_len
    packed_dim = head_dim // 2
    max_kv_splits = 8

    torch.manual_seed(42)
    q = torch.randn(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_packed = torch.randint(0, 255, (total_tokens, kv_heads, packed_dim), dtype=torch.uint8, device=device)
    v_packed = torch.randint(0, 255, (total_tokens, kv_heads, packed_dim), dtype=torch.uint8, device=device)
    k_dscale = (torch.randn(total_tokens, kv_heads, dtype=torch.bfloat16, device=device).abs() + 0.1)
    v_dscale = (torch.randn(total_tokens, kv_heads, dtype=torch.bfloat16, device=device).abs() + 0.1)

    centroids, _ = build_codebook(4, head_dim)
    k_centroids = torch.tensor(centroids, dtype=torch.float32, device=device)
    v_centroids = torch.tensor(centroids, dtype=torch.float32, device=device)

    kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, total_tokens, dtype=torch.int32, device=device)
    num_kv_splits_t = torch.full((batch,), 8, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(head_dim)

    o = torch.zeros(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    al = torch.zeros(batch, q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device)
    lse = torch.zeros(batch, q_heads, max_kv_splits, dtype=torch.float32, device=device)

    def run_triton():
        tq_decode_attention_fwd(q, k_packed, v_packed, k_dscale, v_dscale,
                                k_centroids, v_centroids, o, kv_indptr, kv_indices,
                                al, lse, num_kv_splits_t, max_kv_splits, sm_scale)

    def run_cuda():
        tq4_decode_stage1_cuda(q, k_packed, v_packed, k_dscale, v_dscale,
                               k_centroids, v_centroids, al, lse,
                               kv_indptr, kv_indices, num_kv_splits_t, max_kv_splits, sm_scale)

    for fn, name in [(run_triton, "Triton"), (run_cuda, "CUDA")]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        print(f"{name:8s}: {start.elapsed_time(end) / repeat:.3f} ms/iter")


if __name__ == "__main__":
    print("Compiling CUDA kernel...")
    _get_module()
    print("Compiled. Running correctness test...")
    test_correctness(batch=4, seq_len=64)
    test_correctness(batch=32, seq_len=512)
    print("\nRunning benchmark...")
    benchmark(batch=32, seq_len=1024)
