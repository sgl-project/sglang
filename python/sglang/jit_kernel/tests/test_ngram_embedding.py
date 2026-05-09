import sys

import pytest
import torch

from sglang.jit_kernel.ngram_embedding import (
    update_token_table,
    update_token_table_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")


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
