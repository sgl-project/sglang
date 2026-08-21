"""A/B Benchmark: Baseline (Static 8 Splits) vs Dynamic Split-K (Ours).

Calls real SGLang Triton Decode Kernels (_decode_grouped_att_m_fwd +
_decode_softmax_reducev_fwd) to measure latency across context lengths.
"""

import math

import torch

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd_grouped
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton


def benchmark_real_triton_decode(
    batch_size: int = 1,
    context_len: int = 8192,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    max_kv_splits: int = 32,
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

    # 2. Benchmark Old Static / Conservative 8 Splits vs Dynamic Model
    results = {}
    for mode, force_splits in [
        ("Old Static (8 splits)", 8),
        ("Dynamic Split-K (Ours)", max_kv_splits),
    ]:
        num_kv_splits = torch.empty(batch_size, dtype=torch.int32, device=device)
        seq_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        if mode == "Old Static (8 splits)":
            num_kv_splits.fill_(8)
            cur_max_splits = 8
        else:
            # Run our dynamic hardware-aware split calculator
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
            cur_max_splits = max_kv_splits

        actual_splits = int(num_kv_splits[0].item())
        attn_logits = torch.empty(
            batch_size,
            num_heads,
            cur_max_splits,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        attn_lse = torch.empty(
            batch_size,
            num_heads,
            cur_max_splits,
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
                cur_max_splits,
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
                cur_max_splits,
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
    header = (
        f"{'Context Len':<12} | {'Old (8 splits)':<18} | "
        f"{'Dynamic (Ours)':<18} | {'Speedup'}"
    )
    print(header)

    print("-" * 78)

    for ctx in [256, 1024, 2048, 4096, 8192]:
        res = benchmark_real_triton_decode(batch_size=1, context_len=ctx)
        old_splits, old_ms = res["Old Static (8 splits)"]
        new_splits, new_ms = res["Dynamic Split-K (Ours)"]

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
