import os

import torch

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)
import triton

from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel

# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [64, 128]  # Only test 2 values in CI
else:
    batch_sizes = [64, 128, 256, 512, 640, 768, 1024, 2048, 4096]

configs = [(bs,) for bs in batch_sizes]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton Kernel"],
        styles=[("orange", "-")],
        ylabel="us",
        plot_name="ep-moe-post-reorder-performance",
        args={},
    )
)
def benchmark(batch_size, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    hidden_size, topk, start_expert_id, end_expert_id, block_size = 4096, 8, 0, 255, 512

    def alloc_tensors():
        down_output = torch.randn(
            batch_size * topk, hidden_size, dtype=dtype, device=device
        )
        output = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)
        src2dst = torch.randint(
            0, batch_size * topk, (batch_size, topk), dtype=torch.int32, device=device
        )
        topk_ids = torch.randint(
            start_expert_id,
            end_expert_id + 1,
            (batch_size, topk),
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.rand(batch_size, topk, dtype=dtype, device=device)
        return down_output, output, src2dst, topk_ids, topk_weights

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        d_out, out, s2d, tk_ids, tk_weights = alloc_tensors()

        def run_triton():
            post_reorder_triton_kernel[(batch_size,)](
                d_out.view(-1),
                out.view(-1),
                s2d.view(-1),
                tk_ids.view(-1),
                tk_weights.view(-1),
                start_expert_id,
                end_expert_id,
                topk,
                hidden_size,
                0,
                block_size,
            )

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            run_triton, quantiles=quantiles
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
