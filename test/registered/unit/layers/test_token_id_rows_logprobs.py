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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("copy_to_pinned_cpu", [False, True])
def test_mixed_batch_across_logprob_chunks(normalized, copy_to_pinned_cpu):
    torch.manual_seed(0)
    logits = torch.randn(9, 13)
    dense = logits.log_softmax(-1)
    rows = [[], [2, 5], [7], [], [1, 3, 4], [6]]
    ids, pruned = [TokenIdRows(rows), [4, 7], None], [6, 2, 1]
    values, indices, split = [], [], 0
    # The first chunk ends inside the TokenIdRows sequence.
    for start, end, n in ((0, 4, 1), (4, 9, 3)):
        chunk = (logits if normalized else dense)[start:end]
        maximum = chunk.max(-1).values
        normalizer = (
            (maximum, torch.logsumexp(chunk - maximum[:, None], -1))
            if normalized
            else None
        )
        split = get_token_ids_logprobs_chunk(
            chunk,
            ids[:n],
            pruned[:n],
            values,
            indices,
            split,
            normalizer,
            copy_to_pinned_cpu=copy_to_pinned_cpu,
        )
    if copy_to_pinned_cpu:
        output = LogitsProcessorOutput(
            next_token_logits=None,
            input_token_ids_logprobs_val=values,
            input_logprobs_copy_done=SimpleNamespace(synchronize=lambda: None),
        )
        output.finalize_input_logprobs()
        values = output.input_token_ids_logprobs_val
    assert indices == [rows, [[4, 7], [4, 7]], []]
    expected = [
        [dense[row, token_ids].tolist() for row, token_ids in enumerate(rows)],
        [dense[6, [4, 7]].tolist(), dense[7, [4, 7]].tolist()],
        [],
    ]
    assert [len(v) for v in values] == [len(e) for e in expected]
    for got, want in zip(values, expected):
        for got_row, want_row in zip(got, want):
            assert got_row == pytest.approx(want_row, abs=1e-6)


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
