"""
Benchmark DSV4 inverse RoPE plus FP8 quantization.

Compares the old path, which applies inverse RoPE in place and then materializes
a contiguous grouped layout before quantization, against the packed inverse RoPE
CUDA JIT path that directly emits the contiguous quantization input layout.

Run:
    python test/registered/jit/benchmark/bench_dsv4_rope_pack.py
"""

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.dsv4 import (
    fused_rope_inplace,
    fused_rope_pack,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-benchmark-1-gpu-large")

DSV4_NUM_ATTENTION_HEADS = 64
DSV4_TP_SIZE = 8
DSV4_O_GROUPS = 8
DSV4_LOCAL_HEADS = DSV4_NUM_ATTENTION_HEADS // DSV4_TP_SIZE
DSV4_LOCAL_GROUPS = DSV4_O_GROUPS // DSV4_TP_SIZE

DSV4_HEAD_DIM = 576
ROPE_DIM = 64

TOKEN_RANGE = get_benchmark_range(
    full_range=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    ci_range=[4, 128, 1024],
)


def _make_inputs(num_tokens: int):
    full_heads = DSV4_NUM_ATTENTION_HEADS
    head_offset = 0
    q_base = torch.randn(
        (num_tokens, full_heads, DSV4_HEAD_DIM),
        device=DEFAULT_DEVICE,
        dtype=torch.bfloat16,
    )
    q = q_base[:, head_offset : head_offset + DSV4_LOCAL_HEADS, :]

    positions = torch.arange(num_tokens, device=DEFAULT_DEVICE, dtype=torch.int32)
    angles = torch.randn(
        (num_tokens, ROPE_DIM // 2), device=DEFAULT_DEVICE, dtype=torch.float32
    )
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return q, freqs_cis, positions


def _packed_rope_quant(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
):
    o = fused_rope_pack(q, freqs_cis, positions, DSV4_LOCAL_GROUPS, inverse=True)
    num_tokens, num_groups, packed_dim = o.shape
    o_quant = o.view(num_tokens * num_groups, packed_dim)
    return sglang_per_token_group_quant_fp8(o_quant, group_size=128)


def _old_rope_materialize_quant(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
):
    fused_rope_inplace(
        q[..., -ROPE_DIM:],
        None,
        freqs_cis,
        positions=positions,
        inverse=True,
    )
    o = q.view(q.shape[0], DSV4_LOCAL_GROUPS, -1)
    num_tokens, num_groups, packed_dim = o.shape
    o_quant = o.reshape(num_tokens * num_groups, packed_dim).contiguous()
    return sglang_per_token_group_quant_fp8(o_quant, group_size=128)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=TOKEN_RANGE,
        line_arg="provider",
        line_vals=["packed_rope_quant", "old_materialize_quant"],
        line_names=[
            "Packed inverse RoPE + quant",
            "Old inverse RoPE + materialize + quant",
        ],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="dsv4-rope-pack-quant-performance",
        args={},
    )
)
def benchmark(num_tokens: int, provider: str):
    q, freqs_cis, positions = _make_inputs(num_tokens)

    if provider == "packed_rope_quant":
        fn = lambda: _packed_rope_quant(q, freqs_cis, positions)
    else:
        fn = lambda: _old_rope_materialize_quant(q, freqs_cis, positions)

    # Exclude CUDA JIT build, quant JIT build, and allocator warmup from timing.
    fn()
    torch.cuda.synchronize()

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
