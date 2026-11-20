"""TokenIdRows: each scored row reports its own requested token ids."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    OutputLogprobProcessor,
    get_token_ids_logprobs_chunk,
)
from sglang.srt.managers.req_types import TokenIdRows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROWS = [[], [4, 1], [3], [], [5, 8, 2], [15, 9], [6]]


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7])
@pytest.mark.parametrize("normalized", [False, True])
def test_rows_across_logprob_chunks(chunk_size, normalized):
    torch.manual_seed(0)
    logits = torch.randn(7, 17)
    dense = logits.log_softmax(-1)
    values, indices, split = [], [], 0
    for offset in range(0, 7, chunk_size):
        chunk = (logits if normalized else dense)[offset : offset + chunk_size]
        maximum = chunk.max(-1).values
        normalizer = (
            (maximum, torch.logsumexp(chunk - maximum[:, None], -1))
            if normalized
            else None
        )
        split = get_token_ids_logprobs_chunk(
            chunk, [TokenIdRows(ROWS)], [7], values, indices, split, normalizer
        )
    assert indices == [ROWS]
    for row, token_ids in enumerate(ROWS):
        assert values[0][row] == pytest.approx(dense[row, token_ids].tolist(), abs=1e-6)


def test_mixed_batch():
    torch.manual_seed(1)
    logprobs = torch.randn(6, 13).log_softmax(-1)
    values, indices = [], []
    get_token_ids_logprobs_chunk(
        logprobs, [None, TokenIdRows([[3], [8, 1]]), [4, 7]], [2, 2, 2], values, indices
    )
    assert values[0] == [] and indices[0] == []
    assert indices[1] == [[3], [8, 1]]
    assert values[1][0] == pytest.approx(logprobs[2, [3]].tolist())
    assert values[1][1] == pytest.approx(logprobs[3, [8, 1]].tolist())
    assert indices[2] == [[4, 7], [4, 7]]
    assert values[2][1] == pytest.approx(logprobs[5, [4, 7]].tolist())


def test_pinned_copy_matches_plain_rows():
    torch.manual_seed(2)
    logprobs = torch.randn(9, 13).log_softmax(-1)
    ids = [TokenIdRows([[], [2, 5], [7], [], [1, 3, 4], [6]]), [4, 7], None]
    pruned = [6, 2, 1]

    def gather(copy_to_pinned_cpu):
        values, indices = [], []
        kwargs = dict(copy_to_pinned_cpu=copy_to_pinned_cpu)
        rest = get_token_ids_logprobs_chunk(
            logprobs[:4], ids[:1], pruned[:1], values, indices, **kwargs
        )
        get_token_ids_logprobs_chunk(
            logprobs[4:], ids, pruned, values, indices, rest, **kwargs
        )
        return values, indices

    plain, plain_indices = gather(False)
    spans, span_indices = gather(True)
    output = LogitsProcessorOutput(
        next_token_logits=None,
        input_token_ids_logprobs_val=spans,
        input_logprobs_copy_done=SimpleNamespace(synchronize=lambda: None),
    )
    output.finalize_input_logprobs()
    assert output.input_token_ids_logprobs_val == plain
    assert span_indices == plain_indices
    assert plain[0][4] == pytest.approx(logprobs[4, [1, 3, 4]].tolist())


def test_sampled_token_reports_none():
    torch.manual_seed(3)
    logprobs = torch.randn(3, 13).log_softmax(-1)
    ids = [TokenIdRows([[2, 5]]), None, [4, 7]]
    result = OutputLogprobProcessor().compute_logprobs(
        logprobs, [0, 0, 0], ids, torch.tensor([1, 2, 3])
    )
    values = [value.tolist() for value in result.token_ids_logprobs_val]
    assert values[:2] == [[], []]
    assert values[2] == pytest.approx(logprobs[2, [4, 7]].tolist())
    assert result.token_ids_logprobs_idx == [[], [], [4, 7]]
    assert (
        OutputLogprobProcessor().compute_logprobs_only(
            logprobs[:1], [0], ids[:1], preprocess_fn=lambda logits: logits
        )
        is None
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
