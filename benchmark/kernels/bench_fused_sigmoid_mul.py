"""Benchmark fused_sigmoid_mul: auto-dispatch vs PyTorch eager.

The auto-dispatch path uses a strided Triton kernel for Qwen3.5 MoE attention
output gates.

Both paths start from a strided 3D gate (from torch.chunk) to ensure
a fair comparison — the reshape/contiguous cost is included.
"""

import torch
import triton

from sglang.kernels.ops.layernorm.elementwise import fused_sigmoid_mul

NUM_HEADS = 32
HEAD_DIM = 256
HIDDEN_DIM = NUM_HEADS * HEAD_DIM  # 8192


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[1, 2, 4, 8, 16, 1024, 2048, 4096, 8192],
        line_arg="impl",
        line_vals=["auto", "auto_inplace", "pytorch_from_strided"],
        line_names=[
            "fused_sigmoid_mul (auto)",
            "fused_sigmoid_mul (auto, inplace)",
            "PyTorch eager (incl. reshape)",
        ],
        styles=[("blue", "-"), ("green", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="fused_sigmoid_mul_qwen3_5_moe_target",
        args={},
    )
)
def bench(num_tokens, impl, dtype=torch.bfloat16):
    q_gate = torch.randn(
        num_tokens, NUM_HEADS, 2 * HEAD_DIM, dtype=dtype, device="cuda"
    )
    _, gate_strided = torch.chunk(q_gate, 2, dim=-1)
    attn_output = torch.randn(num_tokens, HIDDEN_DIM, dtype=dtype, device="cuda")

    if impl == "auto":
        fn = lambda: fused_sigmoid_mul(attn_output, gate_strided, inplace=False)
    elif impl == "auto_inplace":
        fn = lambda: fused_sigmoid_mul(attn_output, gate_strided, inplace=True)
    else:
        # Fair comparison: include reshape cost in every iteration
        def fn():
            g = gate_strided.contiguous().view(num_tokens, HIDDEN_DIM)
            return attn_output * torch.sigmoid(g)

    ms = triton.testing.do_bench(fn, warmup=100, rep=200)
    return ms * 1000


if __name__ == "__main__":
    bench.run(print_data=True)
