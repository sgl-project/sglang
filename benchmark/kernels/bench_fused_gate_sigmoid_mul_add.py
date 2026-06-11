"""Benchmark fused_gate_sigmoid_mul_add: Triton kernel vs PyTorch eager.

Compares the fused Triton kernel against a plain PyTorch implementation
over a sweep of (num_tokens, hidden_dim) shapes typical for Qwen2 MoE.
"""

import torch
import triton

from sglang.srt.layers.elementwise import fused_gate_sigmoid_mul_add


def _pytorch_reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = hidden_states @ gate_weight
    final_hidden_states += torch.sigmoid(gate).unsqueeze(1) * shared_output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="impl",
        line_vals=["triton", "pytorch"],
        line_names=["Triton fused", "PyTorch eager"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="fused_gate_sigmoid_mul_add",
        args={},
    )
)
def bench(num_tokens, impl, hidden_dim=3584, dtype=torch.bfloat16):
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gate_weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    shared_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    final_hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=dtype, device="cuda"
    )

    if impl == "triton":
        fn = lambda: fused_gate_sigmoid_mul_add(
            hidden_states, gate_weight, shared_output, final_hidden_states
        )
    else:
        fn = lambda: _pytorch_reference(
            hidden_states, gate_weight, shared_output, final_hidden_states
        )

    ms = triton.testing.do_bench(fn, warmup=100, rep=200)
    return ms * 1000  # convert to us


if __name__ == "__main__":
    bench.run(print_data=True)
