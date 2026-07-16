import sys

import pytest
import torch

from sglang.jit_kernel.ngram_embedding import (
    compute_n_gram_ids,
    compute_n_gram_ids_decode,
    update_token_table,
    update_token_table_decode,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, stage="jit-kernel-unit", runner_config="amd")


def _make_ngram_params(ne_n: int, ne_k: int, vocab_size: int):
    ne_weights = torch.zeros([ne_n - 1, ne_k, ne_n], dtype=torch.int32)
    ne_mods = torch.zeros([ne_n - 1, ne_k], dtype=torch.int32)
    exclusive_sums = torch.zeros([(ne_n - 1) * ne_k + 1], dtype=torch.int32)

    for n in range(2, ne_n + 1):
        for k in range(ne_k):
            config_id = (n - 2) * ne_k + k
            mod = 65537 + 2 * config_id
            ne_mods[n - 2][k] = mod
            exclusive_sums[config_id + 1] = exclusive_sums[config_id] + mod
            for delta in range(ne_n):
                ne_weights[n - 2][k][delta] = pow(vocab_size, delta, mod)

    return (
        ne_weights.cuda(),
        ne_mods.cuda(),
        exclusive_sums.cuda(),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 17, 128, 1024])
def test_compute_n_gram_ids_decode_matches_general(batch_size: int) -> None:
    ne_n = 8
    ne_k = 2
    vocab_size = 32000
    eos_token_id = vocab_size
    max_context_len = 1024
    max_running_reqs = batch_size + 8
    num_configs = (ne_n - 1) * ne_k

    ne_weights, ne_mods, exclusive_sums = _make_ngram_params(ne_n, ne_k, vocab_size)
    ne_token_table = torch.randint(
        0,
        vocab_size,
        (max_running_reqs, max_context_len),
        dtype=torch.int32,
        device="cuda",
    )
    row_indices = torch.randperm(max_running_reqs, device="cuda")[:batch_size].to(
        torch.int64
    )
    column_starts = torch.randint(
        0, max_context_len, (batch_size,), dtype=torch.int32, device="cuda"
    )
    tokens = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device="cuda"
    )
    exclusive_req_len_sums = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    n_gram_ids_general = torch.empty(
        (batch_size, num_configs), dtype=torch.int32, device="cuda"
    )
    n_gram_ids_decode = torch.empty_like(n_gram_ids_general)

    compute_n_gram_ids(
        ne_n=ne_n,
        ne_k=ne_k,
        ne_weights=ne_weights,
        ne_mods=ne_mods,
        exclusive_ne_embedder_size_sums=exclusive_sums,
        tokens=tokens,
        exclusive_req_len_sums=exclusive_req_len_sums,
        ne_token_table=ne_token_table,
        row_indices=row_indices,
        column_starts=column_starts,
        n_gram_ids=n_gram_ids_general,
        eos_token_id=eos_token_id,
    )
    compute_n_gram_ids_decode(
        ne_n=ne_n,
        ne_k=ne_k,
        ne_weights=ne_weights,
        ne_mods=ne_mods,
        exclusive_ne_embedder_size_sums=exclusive_sums,
        ne_token_table=ne_token_table,
        row_indices=row_indices,
        column_starts=column_starts,
        n_gram_ids=n_gram_ids_decode,
        eos_token_id=eos_token_id,
    )

    torch.testing.assert_close(n_gram_ids_decode, n_gram_ids_general, atol=0, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 2, 17, 128, 1024])
def test_update_token_table_decode_matches_general(batch_size: int) -> None:
    max_context_len = 4096
    max_running_reqs = batch_size + 8
    tokens = torch.arange(batch_size, dtype=torch.int32, device="cuda") + 100
    row_indices = torch.randperm(max_running_reqs, device="cuda")[:batch_size].to(
        torch.int64
    )
    column_starts = torch.randint(
        0, max_context_len, (batch_size,), dtype=torch.int32, device="cuda"
    )
    req_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    token_table_general = torch.full(
        (max_running_reqs, max_context_len), -1, dtype=torch.int32, device="cuda"
    )
    token_table_decode = token_table_general.clone()

    update_token_table(
        tokens=tokens,
        ne_token_table=token_table_general,
        row_indices=row_indices,
        column_starts=column_starts,
        req_lens=req_lens,
        ignore_tokens=None,
    )
    update_token_table_decode(
        tokens=tokens,
        ne_token_table=token_table_decode,
        row_indices=row_indices,
        column_starts=column_starts,
    )

    torch.testing.assert_close(token_table_decode, token_table_general, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
