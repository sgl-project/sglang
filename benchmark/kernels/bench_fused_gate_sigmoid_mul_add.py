"""Benchmark fused_gate_sigmoid_mul_add: Triton kernel vs PyTorch eager.

Compares the fused Triton kernel against a plain PyTorch implementation
over the Qwen3.5 MoE target hidden size.
"""

import torch
import triton

from sglang.srt.layers.elementwise import fused_gate_sigmoid_mul_add

HIDDEN_DIMS = [4096]


def _pytorch_reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = hidden_states @ gate_weight
    final_hidden_states += torch.sigmoid(gate).unsqueeze(1) * shared_output


def make_bench(hidden_dim):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=[1, 2, 4, 8, 16, 1024, 2048, 4096, 8192],
            line_arg="impl",
            line_vals=["triton", "pytorch"],
            line_names=["Triton fused", "PyTorch eager"],
            styles=[("blue", "-"), ("orange", "--")],
            ylabel="us",
            plot_name=f"fused_gate_sigmoid_mul_add-hidden{hidden_dim}",
            args={"hidden_dim": hidden_dim},
        )
    )
    def bench(num_tokens, impl, hidden_dim, dtype=torch.bfloat16):
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

    return bench


if __name__ == "__main__":
    for d in HIDDEN_DIMS:
        print(f"\n===== hidden_dim={d} =====")
        make_bench(d).run(print_data=True)
