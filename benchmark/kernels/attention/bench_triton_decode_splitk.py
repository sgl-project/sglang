"""A/B Benchmark: Baseline (Legacy Log2) vs Dynamic Chunk-Driven (Ours).

Calls real SGLang Triton Decode Kernels (_decode_grouped_att_m_fwd +
_decode_softmax_reducev_fwd) to measure latency across context lengths under
both production default cap (max_kv_splits=8) and high cap (max_kv_splits=32).
"""

import math

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd_grouped
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton


@triton.jit
def get_num_kv_splits_triton_legacy(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    """Original legacy heuristic log2 split calculation from SGLang main."""
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # Legacy heuristic log2 scaling hack
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def benchmark_real_triton_decode(
    batch_size: int = 1,
    context_len: int = 8192,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    max_kv_splits: int = 8,
    device_core_count: int | None = None,
    num_iters: int = 50,
):
    device = "cuda"
    dtype = torch.float16
    sm_scale = 1.0 / math.sqrt(head_dim)
    page_size = 1
    if device_core_count is None:
        device_core_count = torch.cuda.get_device_properties(0).multi_processor_count

    # 1. Allocate tensors
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k_buffer = torch.randn(
        context_len * batch_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_buffer = torch.randn(
        context_len * batch_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    o = torch.empty_like(q)

    kv_indptr = torch.tensor([0, context_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(context_len, dtype=torch.int64, device=device)

    # 2. Benchmark Legacy Log2 vs Ours under the SAME cap
    results = {}
    for mode in ["Baseline (Legacy Log2)", "Dynamic (Ours)"]:
        num_kv_splits = torch.empty(batch_size, dtype=torch.int32, device=device)
        seq_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        if mode == "Baseline (Legacy Log2)":
            get_num_kv_splits_triton_legacy[(1,)](
                num_kv_splits,
                seq_lens,
                batch_size,
                1,
                num_heads,
                num_kv_heads,
                max_kv_splits,
                device_core_count,
                MAX_NUM_SEQ=32,
            )
        else:
            get_num_kv_splits_triton[(1,)](
                num_kv_splits,
                seq_lens,
                batch_size,
                1,
                num_heads,
                num_kv_heads,
                max_kv_splits,
                device_core_count,
                MAX_NUM_SEQ=32,
            )

        actual_splits = int(num_kv_splits[0].item())
        attn_logits = torch.empty(
            batch_size,
            num_heads,
            max_kv_splits,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        attn_lse = torch.empty(
            batch_size,
            num_heads,
            max_kv_splits,
            dtype=torch.float32,
            device=device,
        )

        # Warmup
        for _ in range(10):
            decode_attention_fwd_grouped(
                q,
                k_buffer,
                v_buffer,
                o,
                kv_indptr,
                kv_indices,
                attn_logits,
                attn_lse,
                num_kv_splits,
                max_kv_splits,
                sm_scale,
                1.0,
                page_size=page_size,
            )
        torch.cuda.synchronize()

        # Timed execution with CUDA Events
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        for _ in range(num_iters):
            decode_attention_fwd_grouped(
                q,
                k_buffer,
                v_buffer,
                o,
                kv_indptr,
                kv_indices,
                attn_logits,
                attn_lse,
                num_kv_splits,
                max_kv_splits,
                sm_scale,
                1.0,
                page_size=page_size,
            )
        end_ev.record()
        torch.cuda.synchronize()

        avg_latency = start_ev.elapsed_time(end_ev) / num_iters
        results[mode] = (actual_splits, avg_latency)

    return results


def run_full_benchmark():
    if not torch.cuda.is_available():
        print("CUDA is required for Triton benchmark.")
        return

    device_name = torch.cuda.get_device_name(0)
    print("\n" + "=" * 78)
    print(f"  Real SGLang Triton Decode Kernel Microbenchmark on {device_name}")
    print("=" * 78)

    for cap in [8, 32]:
        cap_title = (
            f"--- Cap: max_kv_splits={cap} "
            f"({'Production Default' if cap == 8 else 'High-Cap Mode'}) ---"
        )
        print(f"\n{cap_title}")
        header = (
            f"{'Context Len':<12} | {'Legacy Log2':<18} | "
            f"{'Dynamic (Ours)':<18} | {'Speedup'}"
        )
        print(header)
        print("-" * 78)

        for ctx in [256, 512, 1024, 2048, 4096, 8192]:
            res = benchmark_real_triton_decode(
                batch_size=1, context_len=ctx, max_kv_splits=cap
            )
            old_splits, old_ms = res["Baseline (Legacy Log2)"]
            new_splits, new_ms = res["Dynamic (Ours)"]

            speedup_val = ((old_ms - new_ms) / old_ms) * 100
            speedup_str = (
                f"+{speedup_val:.1f}%" if speedup_val > 0 else f"{speedup_val:.1f}%"
            )
            out_line = (
                f"{ctx:<12} | {old_ms:.4f} ms ({old_splits}sp)   | "
                f"{new_ms:.4f} ms ({new_splits}sp)   | {speedup_str}"
            )
            print(out_line)

    print("=" * 78)


if __name__ == "__main__":
    run_full_benchmark()
