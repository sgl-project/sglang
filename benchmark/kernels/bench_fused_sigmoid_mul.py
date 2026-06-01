"""Benchmark fused_sigmoid_mul: Triton kernel vs PyTorch eager.

Compares the fused Triton kernel against a plain PyTorch implementation
over a sweep of num_tokens for attention output gating.
"""

import torch
import triton

from sglang.srt.layers.elementwise import fused_sigmoid_mul

HIDDEN_DIMS = [2048, 3072, 4096, 6144]


def _pytorch_reference(attn_output, gate):
    return attn_output * torch.sigmoid(gate)


def make_bench(hidden_dim):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=[1, 4, 16, 64, 512, 1024, 2048, 8192],
            line_arg="impl",
            line_vals=["triton", "pytorch"],
            line_names=["Triton fused", "PyTorch eager"],
            styles=[("blue", "-"), ("orange", "--")],
            ylabel="us",
            plot_name=f"fused_sigmoid_mul-hidden{hidden_dim}",
            args={"hidden_dim": hidden_dim},
        )
    )
    def bench(num_tokens, impl, hidden_dim, dtype=torch.bfloat16):
        attn_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
        gate = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")

        if impl == "triton":
            fn = lambda: fused_sigmoid_mul(attn_output, gate)
        else:
            fn = lambda: _pytorch_reference(attn_output, gate)

        ms = triton.testing.do_bench(fn, warmup=100, rep=200)
        return ms * 1000  # convert to us

    return bench


if __name__ == "__main__":
    for d in HIDDEN_DIMS:
        print(f"\n===== hidden_dim={d} =====")
        make_bench(d).run(print_data=True)
