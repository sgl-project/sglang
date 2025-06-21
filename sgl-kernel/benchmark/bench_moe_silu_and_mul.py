import itertools

import torch
import triton
from sgl_kernel import ep_moe_silu_and_mul

from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_triton_kernel

batch_size_range = [64, 128, 256, 512, 640, 768, 1024, 2048, 4096]
hidden_size_range = [1024, 2048, 4096, 8192]
block_size_range = [128, 256, 512]
configs = list(itertools.product(batch_size_range, hidden_size_range, block_size_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size", "block_size"],
        x_vals=[list(cfg) for cfg in configs],
        line_arg="provider",
        line_vals=["cuda", "triton"],
        line_names=["CUDA Kernel", "Triton Kernel"],
        styles=[("green", "-"), ("orange", "-")],
        ylabel="us",
        plot_name="ep-moe-silu-and-mul-performance",
        args={},
    )
)
def benchmark(batch_size, hidden_size, block_size, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    half_hidden_size = hidden_size // 2
    start_expert_id, end_expert_id = 0, 255
    block_size = 512
    quantiles = [0.5, 0.2, 0.8]

    def alloc_tensors():
        gateup_output = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        down_input = torch.empty(
            batch_size, half_hidden_size, dtype=dtype, device=device
        )
        reorder_topk_ids = torch.randint(
            start_expert_id,
            end_expert_id + 1,
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        scales = torch.rand(
            end_expert_id - start_expert_id + 1, dtype=torch.float32, device=device
        )
        return gateup_output, down_input, reorder_topk_ids, scales

    if provider == "cuda":
        gateup, down, ids, scales = alloc_tensors()

        def run_cuda():
            ep_moe_silu_and_mul(
                gateup,
                down,
                ids,
                scales,
                start_expert_id,
                end_expert_id,
            )

        ms, min_ms, max_ms = triton.testing.do_bench(run_cuda, quantiles=quantiles)

    elif provider == "triton":
        gateup, down, ids, scales = alloc_tensors()

        def run_triton():
            silu_and_mul_triton_kernel[(batch_size,)](
                gateup.view(-1),
                down.view(-1),
                hidden_size,
                ids,
                scales,
                start_expert_id,
                end_expert_id,
                block_size,
            )

        ms, min_ms, max_ms = triton.testing.do_bench(run_triton, quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
