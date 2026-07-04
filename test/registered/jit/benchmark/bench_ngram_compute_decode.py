import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.ngram_embedding import (
    compute_n_gram_ids,
    compute_n_gram_ids_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NE_N = 8
NE_K = 2
VOCAB_SIZE = 32000
MAX_CONTEXT_LEN = 1024


def _make_ngram_params():
    ne_weights = torch.zeros([NE_N - 1, NE_K, NE_N], dtype=torch.int32)
    ne_mods = torch.zeros([NE_N - 1, NE_K], dtype=torch.int32)
    exclusive_sums = torch.zeros([(NE_N - 1) * NE_K + 1], dtype=torch.int32)

    for n in range(2, NE_N + 1):
        for k in range(NE_K):
            config_id = (n - 2) * NE_K + k
            mod = 65537 + 2 * config_id
            ne_mods[n - 2][k] = mod
            exclusive_sums[config_id + 1] = exclusive_sums[config_id] + mod
            for delta in range(NE_N):
                ne_weights[n - 2][k][delta] = pow(VOCAB_SIZE, delta, mod)

    return (
        ne_weights.to(DEFAULT_DEVICE),
        ne_mods.to(DEFAULT_DEVICE),
        exclusive_sums.to(DEFAULT_DEVICE),
    )


@marker.parametrize(
    "batch_size",
    [1, 2, 8, 32, 128, 512, 1024, 2048, 4096],
    [32, 1024],
)
@marker.benchmark("provider", ["general", "decode"])
def benchmark(batch_size: int, provider: str):
    num_configs = (NE_N - 1) * NE_K
    max_running_reqs = batch_size + 8
    ne_weights, ne_mods, exclusive_sums = _make_ngram_params()
    ne_token_table = torch.randint(
        0,
        VOCAB_SIZE,
        (max_running_reqs, MAX_CONTEXT_LEN),
        dtype=torch.int32,
        device=DEFAULT_DEVICE,
    )
    row_indices = torch.arange(batch_size, dtype=torch.int64, device=DEFAULT_DEVICE)
    column_starts = torch.randint(
        0, MAX_CONTEXT_LEN, (batch_size,), dtype=torch.int32, device=DEFAULT_DEVICE
    )
    n_gram_ids = torch.empty(
        (batch_size, num_configs), dtype=torch.int32, device=DEFAULT_DEVICE
    )

    if provider == "general":
        tokens = torch.empty(batch_size, dtype=torch.int32, device=DEFAULT_DEVICE)
        exclusive_req_len_sums = torch.arange(
            batch_size + 1, dtype=torch.int32, device=DEFAULT_DEVICE
        )
        return marker.do_bench(
            compute_n_gram_ids,
            input_args=(
                NE_N,
                NE_K,
                ne_weights,
                ne_mods,
                exclusive_sums,
                tokens,
                exclusive_req_len_sums,
                ne_token_table,
                row_indices,
                column_starts,
                n_gram_ids,
            ),
            memory_output=(n_gram_ids,),  # inplace write to n_gram_ids
        )

    return marker.do_bench(
        compute_n_gram_ids_decode,
        input_args=(
            NE_N,
            NE_K,
            ne_weights,
            ne_mods,
            exclusive_sums,
            ne_token_table,
            row_indices,
            column_starts,
            n_gram_ids,
        ),
        memory_output=(n_gram_ids,),  # inplace write to n_gram_ids
    )


if __name__ == "__main__":
    benchmark.run()
