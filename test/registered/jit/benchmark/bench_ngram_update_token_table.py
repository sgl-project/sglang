import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.jit_kernel.ngram_embedding import (
    update_token_table,
    update_token_table_decode,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=15, stage="jit-kernel-benchmark", runner_config="amd")

MAX_CONTEXT_LEN = 4096
BATCH_SIZE_LIST = get_benchmark_range(
    full_range=[1, 2, 8, 32, 128, 512, 1024, 2048, 4096],
    ci_range=[32, 1024],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZE_LIST,
        line_arg="provider",
        line_vals=["general", "decode"],
        line_names=["general update_token_table", "decode fast path"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="us",
        plot_name="ngram-update-token-table",
        args={},
    )
)
def benchmark(batch_size: int, provider: str):
    max_running_reqs = batch_size + 8
    tokens = torch.arange(batch_size, dtype=torch.int32, device=DEFAULT_DEVICE)
    token_table = torch.empty(
        (max_running_reqs, MAX_CONTEXT_LEN), dtype=torch.int32, device=DEFAULT_DEVICE
    )
    row_indices = torch.arange(batch_size, dtype=torch.int64, device=DEFAULT_DEVICE)
    column_starts = torch.randint(
        0, MAX_CONTEXT_LEN, (batch_size,), dtype=torch.int32, device=DEFAULT_DEVICE
    )
    req_lens = torch.ones(batch_size, dtype=torch.int32, device=DEFAULT_DEVICE)

    if provider == "general":

        def fn():
            update_token_table(
                tokens,
                token_table,
                row_indices,
                column_starts,
                req_lens,
                None,
            )

    else:

        def fn():
            update_token_table_decode(
                tokens,
                token_table,
                row_indices,
                column_starts,
            )

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
