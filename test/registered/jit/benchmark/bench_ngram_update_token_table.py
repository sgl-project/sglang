import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.ngram_embedding import (
    update_token_table,
    update_token_table_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

MAX_CONTEXT_LEN = 4096


@marker.parametrize(
    "batch_size",
    [1, 2, 8, 32, 128, 512, 1024, 2048, 4096],
    [32, 1024],
)
@marker.benchmark("provider", ["general", "decode"])
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
        return marker.do_bench(
            update_token_table,
            input_args=(
                tokens,
                token_table,
                row_indices,
                column_starts,
                req_lens,
                None,
            ),
            # token_table is large + write-only; do not clone it
            graph_clone_args=(0, 2, 3, 4),
            memory_args=(tokens, row_indices, column_starts, req_lens),
            memory_output=(tokens,),  # scatter into token_table; bytes ~ tokens
        )

    return marker.do_bench(
        update_token_table_decode,
        input_args=(
            tokens,
            token_table,
            row_indices,
            column_starts,
        ),
        # token_table is large + write-only; do not clone it
        graph_clone_args=(0, 2, 3),
        memory_args=(tokens, row_indices, column_starts),
        memory_output=(tokens,),  # scatter into token_table; bytes ~ tokens
    )


if __name__ == "__main__":
    benchmark.run()
