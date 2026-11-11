import sys

import pytest
import torch

from sglang.kernels.ops.embeddings.ngram_embedding import (
    compute_n_gram_ids,
    update_token_table,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
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

    return ne_weights, ne_mods, exclusive_sums


def _reference_n_gram_ids(
    ne_n, ne_k, weights, mods, sums, table, rows, starts, lens, eos_token_id
):
    """Each token's (n, k) id: the windowed modular hash of the n tokens ending at
    it, stopping at the request start, an ignored (negative) token or a past eos."""
    ids = []
    for row, start, length in zip(rows, starts, lens):
        for pos in range(start, start + length):
            token_ids = []
            for n in range(ne_n - 1):
                for k in range(ne_k):
                    mod = mods[n][k]
                    acc = 0
                    for j in range(n + 2):
                        if pos - j < 0:
                            break
                        token = table[row][pos - j]
                        if token < 0 or (token == eos_token_id and j > 0):
                            break
                        acc += token * weights[n][k][j] % mod
                    token_ids.append(acc % mod + sums[n * ne_k + k])
            ids.append(token_ids)
    return torch.tensor(ids, dtype=torch.int32)


@pytest.mark.parametrize("batch_size", [1, 17, 128])
def test_update_table_and_compute_n_gram_ids_match_reference(batch_size: int) -> None:
    ne_n = 8
    ne_k = 2
    vocab_size = 1000
    eos_token_id = 7
    ignore_token_id = 11
    max_context_len = 256
    max_running_reqs = batch_size + 8
    gen = torch.Generator().manual_seed(batch_size)

    ne_weights, ne_mods, exclusive_sums = _make_ngram_params(ne_n, ne_k, vocab_size)
    row_indices = torch.randperm(max_running_reqs, generator=gen)[:batch_size]
    req_lens = torch.randint(1, 33, (batch_size,), generator=gen, dtype=torch.int32)
    column_starts = torch.stack(
        [
            torch.randint(0, max_context_len - int(n) + 1, (1,), generator=gen)[0]
            for n in req_lens
        ]
    ).to(torch.int32)
    # Small ids so eos and the ignored token show up inside the n-gram windows.
    tokens = torch.randint(0, 16, (int(req_lens.sum()),), generator=gen).to(torch.int32)
    table = torch.randint(
        0, vocab_size, (max_running_reqs, max_context_len), generator=gen
    ).to(torch.int32)

    expected_table = table.clone()
    offset = 0
    for row, start, length in zip(
        row_indices.tolist(), column_starts.tolist(), req_lens.tolist()
    ):
        chunk = tokens[offset : offset + length]
        expected_table[row, start : start + length] = torch.where(
            chunk == ignore_token_id, -chunk, chunk
        )
        offset += length

    device_table = table.cuda()
    update_token_table(
        tokens=tokens.cuda(),
        ne_token_table=device_table,
        row_indices=row_indices.cuda(),
        column_starts=column_starts.cuda(),
        req_lens=req_lens.cuda(),
        ignore_tokens=torch.tensor([ignore_token_id], dtype=torch.int32).cuda(),
    )
    torch.testing.assert_close(device_table.cpu(), expected_table, atol=0, rtol=0)

    n_gram_ids = torch.empty(
        (tokens.numel(), (ne_n - 1) * ne_k), dtype=torch.int32, device="cuda"
    )
    compute_n_gram_ids(
        ne_n=ne_n,
        ne_k=ne_k,
        ne_weights=ne_weights.cuda(),
        ne_mods=ne_mods.cuda(),
        exclusive_ne_embedder_size_sums=exclusive_sums.cuda(),
        tokens=tokens.cuda(),
        exclusive_req_len_sums=torch.cat(
            (req_lens.new_zeros(1), req_lens.cumsum(0).to(torch.int32))
        ).cuda(),
        ne_token_table=device_table,
        row_indices=row_indices.cuda(),
        column_starts=column_starts.cuda(),
        n_gram_ids=n_gram_ids,
        eos_token_id=eos_token_id,
    )
    expected = _reference_n_gram_ids(
        ne_n,
        ne_k,
        ne_weights.tolist(),
        ne_mods.tolist(),
        exclusive_sums.tolist(),
        expected_table.tolist(),
        row_indices.tolist(),
        column_starts.tolist(),
        req_lens.tolist(),
        eos_token_id,
    )
    torch.testing.assert_close(n_gram_ids.cpu(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
