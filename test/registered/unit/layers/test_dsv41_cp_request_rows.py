"""Hopper prefill must index the same request pools after interleaved CP."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from sglang.srt.layers.attention.dsv4.v41_indexer.types import (
    materialize_prefill_request_rows,
)
from sglang.srt.layers.cp.interleave import interleave_rows_per_request


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "lengths", [[1, 0, 2, 13], [3, 5, 7, 19], [0, 0, 0, 0], [1, 1, 1, 1]]
)
def test_pool_ids_follow_global_sharding(cp_size, lengths):
    pools = torch.tensor([31, 4, 67, 9], dtype=torch.int32)
    global_rows = pools.repeat_interleave(torch.tensor(lengths))
    for rank in range(cp_size):
        expected = global_rows[rank::cp_size]
        counts = interleave_rows_per_request(lengths, rank, cp_size)
        inputs = SimpleNamespace(
            req_rows=None,
            req_pool_indices=pools,
            positions=torch.empty(expected.numel(), dtype=torch.int64),
            rows_per_request=counts,
            rows_per_request_device=torch.tensor(counts, dtype=torch.int32),
        )
        actual = materialize_prefill_request_rows(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_tail_counts_exclude_padding_and_zero_row_requests():
    inputs = SimpleNamespace(
        req_rows=None,
        req_pool_indices=torch.tensor([8, 41, 3, 27]),
        positions=torch.arange(5),
        rows_per_request=[0, 2, 0, 3],
        rows_per_request_device=torch.tensor([0, 2, 0, 3]),
    )
    torch.testing.assert_close(
        materialize_prefill_request_rows(inputs), torch.tensor([41, 41, 27, 27, 27])
    )


def test_regular_prefill_reuses_its_existing_rows():
    rows = torch.tensor([5, 5, 17])
    assert materialize_prefill_request_rows(SimpleNamespace(req_rows=rows)) is rows


def test_rejects_padded_rows_as_real_queries():
    inputs = SimpleNamespace(
        req_rows=None,
        req_pool_indices=torch.tensor([8, 41]),
        positions=torch.arange(6),
        rows_per_request=[2, 3],
        rows_per_request_device=torch.tensor([2, 3]),
    )
    with pytest.raises(AssertionError):
        materialize_prefill_request_rows(inputs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
