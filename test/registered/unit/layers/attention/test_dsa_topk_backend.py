import sys

import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import _topk_unfused
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _reference_topk(score, lengths, topk, row_starts):
    result = torch.full((score.shape[0], topk), -1, dtype=torch.int32)
    for row in range(score.shape[0]):
        start = int(row_starts[row])
        end = start + int(lengths[row])
        valid = score[row, start:end]
        count = min(topk, valid.numel())
        if count:
            indices = torch.topk(valid, count).indices.to(torch.int32)
            result[row, :count] = indices
    return result


@pytest.mark.parametrize(
    "batch_size,topk",
    [(4096, 16), (4097, 8)],
    ids=["chunk-boundary", "first-chunked-row"],
)
def test_topk_unfused_chunks_wide_batches(batch_size, topk):
    generator = torch.Generator().manual_seed(0)
    score = torch.randn(batch_size, 16, generator=generator)
    lengths = torch.randint(
        0, 17, (batch_size,), generator=generator, dtype=torch.int32
    )
    row_starts = torch.randint(
        0, 5, (batch_size,), generator=generator, dtype=torch.int32
    )

    actual = _topk_unfused(score, lengths, topk, row_starts=row_starts)
    expected = _reference_topk(score, lengths, topk, row_starts)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
