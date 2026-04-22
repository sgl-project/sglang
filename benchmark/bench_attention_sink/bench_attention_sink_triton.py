import argparse

import torch
import triton

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd

# gpt oss
head_num = 64
head_dim = 64
head_kv_num = 8


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],  # sequence length on x-axis
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="B",  # batch size as different lines
        line_vals=[1, 8, 32, 128],
        line_names=["B=1", "B=8", "B=32", "B=128"],
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("red", "-"),
            ("cyan", "-"),
        ],
        ylabel="TFLOPS",
        plot_name="attention-sink-triton-decode",
        args={},
    )
)
def benchmark_decode(B, S, H_Q, H_KV, D):
    D_V = D
    dtype = torch.bfloat16
    seq_len = S
    total_tokens = B * seq_len
    device = torch.device("cuda")
    sm_scale = 1.0 / (D**0.5)
    max_kv_splits = 8
    num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="cuda")

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

    b_seq_len = torch.full((B,), seq_len, device="cuda")

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len, dim=0)
    kv_indices = torch.arange(total_tokens, device="cuda")

    attn_logits1 = torch.empty(
        (B, H_Q, max_kv_splits, D_V),
        dtype=torch.float32,
        device="cuda",
    )
    attn_lse1 = torch.empty(
        (B, H_Q, max_kv_splits, D_V),
        dtype=torch.float32,
        device="cuda",
    )
    sink = torch.randn(H_Q, device=device, dtype=torch.float32)

    # warmup
    for _ in range(5):
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits1,
            attn_lse1,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=0.0,
            sinks=sink,
        )

    # benchmark
    run_step = 500
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(run_step):
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits1,
            attn_lse1,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=0.0,
            sinks=sink,
        )
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / run_step
    tflops = lambda ms: (2 * B * S * H_Q * D) * 1e-9 / ms  # must be causal
    return tflops(ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],  # sequence length on x-axis
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="B",  # batch size as different lines
        line_vals=[1, 8, 32, 128],
        line_names=["B=1", "B=8", "B=32", "B=128"],
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("red", "-"),
            ("cyan", "-"),
        ],
        ylabel="TFLOPS",
        plot_name="attention-sink-triton-extend",
        args={},
    )
)
def benchmark_extend(B, S, H_Q, H_KV, D):
    # S here represents N_CTX from the test
    dtype = torch.bfloat16
    device = "cuda"

    # Split S into prefix and extend lengths
    prefill_len = S // 2  # Similar to test's N_CTX // 2
    extend_len = S // 4  # Make extend length smaller than prefix

    # Calculate total tokens and extend tokens
    total_extend_tokens = B * extend_len
    total_prefix_tokens = B * prefill_len

    # Create query, key, value tensors for extension
    q_extend = torch.randn(total_extend_tokens, H_Q, D, dtype=dtype, device=device)
    k_extend = torch.randn(total_extend_tokens, H_KV, D, dtype=dtype, device=device)
    v_extend = torch.randn(total_extend_tokens, H_KV, D, dtype=dtype, device=device)
    o_extend = torch.empty_like(q_extend)

    # Create key-value buffers for prefix
    k_buffer = torch.randn(total_prefix_tokens, H_KV, D, dtype=dtype, device=device)
    v_buffer = torch.randn(total_prefix_tokens, H_KV, D, dtype=dtype, device=device)

    # Create index pointers
    qo_indptr = torch.arange(0, (B + 1) * extend_len, extend_len, device=device).to(
        torch.int32
    )
    kv_indptr = torch.arange(0, (B + 1) * prefill_len, prefill_len, device=device).to(
        torch.int32
    )
    kv_indices = torch.arange(0, total_prefix_tokens, device=device).to(torch.int32)

    sm_scale = 1.0 / (D**0.5)
    # sliding_window = 128  # From GPT-OSS config, skip for now
    sliding_window = -1

    sink = torch.randn(H_Q, device=device, dtype=torch.float32)

    # warmup
    for _ in range(5):
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=extend_len,
            sm_scale=sm_scale,
            sliding_window_size=sliding_window,
            sinks=sink,
        )

    # benchmark
    run_step = 500
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(run_step):
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=extend_len,
            sm_scale=sm_scale,
            sliding_window_size=sliding_window,
            sinks=sink,
        )
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / run_step

    # FLOPS calculation: each attention operation requires 2 multiplications per element
    total_flops = 2 * total_extend_tokens * H_Q * (prefill_len + extend_len / 2) * D
    tflops = lambda ms: total_flops * 1e-12 / (ms * 1e-3)  # convert to TFLOPS
    return tflops(ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="all", help="all, extend, decode")
    args = parser.parse_args()

    kwargs = {
        "H_Q": head_num,
        "H_KV": head_kv_num,
        "D": head_dim,
    }

    if args.bench in ["all", "decode"]:
        benchmark_decode.run(print_data=True, show_plots=False, **kwargs)

    if args.bench in ["all", "extend"]:
        benchmark_extend.run(print_data=True, show_plots=False, **kwargs)

    print("Benchmark finished!")
