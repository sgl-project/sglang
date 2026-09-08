import sys

import pytest
import torch

from sglang.srt.layers.logprob_processor import (
    LogprobStage,
    OutputLogprobProcessor,
    get_token_ids_logprobs_chunk,
    get_token_ids_logprobs_raw,
)
from sglang.srt.logprob_types import PerPositionTokenIds
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def test_sparse_raw_matches_dense_with_empty_prompt_and_shifted_positions():
    torch.manual_seed(1)
    logprobs = torch.randn(6, 13).log_softmax(-1)
    rows = [[], [], [2, 5], [7], [3, 8], [11, 1]]
    ids = PerPositionTokenIds(rows, start=1)
    vals, indices = get_token_ids_logprobs_raw(
        logprobs, [ids], stage=LogprobStage.PREFILL, extend_logprob_pruned_lens_cpu=[6]
    )
    expected_rows = rows[1:] + [[]]
    assert indices == [expected_rows]
    for i, candidates in enumerate(expected_rows):
        assert vals[0][i] == pytest.approx(logprobs[i, candidates].tolist())
    # Scoring-only requests do not request candidates for a new generated token.
    values, indices = get_token_ids_logprobs_raw(
        logprobs[:1], [ids], stage=LogprobStage.DECODE
    )
    assert values == [[]] and indices == [[]]


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7])
@pytest.mark.parametrize("normalized", [False, True])
def test_sparse_chunk_stitching_matches_dense(chunk_size, normalized):
    torch.manual_seed(2)
    logits = torch.randn(7, 17)
    dense = logits.log_softmax(-1)
    positions = [[], [4, 1], [3], [], [5, 8, 2], [15, 9], [6]]
    ids = PerPositionTokenIds(positions)
    values, indices, split = [], [], 0
    for offset in range(0, 7, chunk_size):
        chunk = (
            logits[offset : offset + chunk_size]
            if normalized
            else dense[offset : offset + chunk_size]
        )
        maximum = chunk.max(-1).values
        normalizer = (
            (maximum, torch.logsumexp(chunk - maximum[:, None], -1))
            if normalized
            else None
        )
        split = get_token_ids_logprobs_chunk(
            chunk, [ids], [7], values, indices, split, normalizer
        )
    assert indices == [positions]
    for row, candidates in enumerate(positions):
        assert values[0][row] == pytest.approx(
            dense[row, candidates].tolist(), abs=1e-6
        )


def test_sparse_and_flat_requests_share_one_batch_without_position_drift():
    logprobs = torch.randn(6, 13).log_softmax(-1)
    ids = [None, PerPositionTokenIds([[3], [8, 1]]), [4, 7]]
    values, indices = get_token_ids_logprobs_raw(
        logprobs,
        ids,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=[2, 2, 2],
    )
    assert values[0] == []
    assert values[1][0] == pytest.approx(logprobs[2, [3]].tolist())
    assert values[1][1] == pytest.approx(logprobs[3, [8, 1]].tolist())
    assert values[2][0] == pytest.approx(logprobs[4, [4, 7]].tolist())
    assert indices[1] == [[3], [8, 1]]


def test_mixed_decode_keeps_empty_values_as_tensors_for_host_normalization():
    logprobs = torch.randn(3, 13).log_softmax(-1)
    ids = [PerPositionTokenIds([[2, 5]]), None, [4, 7]]
    result = OutputLogprobProcessor().compute_logprobs(
        logprobs, [0, 0, 0], ids, torch.tensor([1, 2, 3])
    )
    # Both prefill/decode result consumers call tolist() after asynchronous D2H.
    assert all(torch.is_tensor(value) for value in result.token_ids_logprobs_val)
    values = [value.tolist() for value in result.token_ids_logprobs_val]
    assert values[:2] == [[], []]
    assert values[2] == pytest.approx(logprobs[2, [4, 7]].tolist())
    assert result.token_ids_logprobs_idx == [[], [], [4, 7]]


@pytest.mark.parametrize("boundaries", [(12,), (24,), (30,), (12, 24, 30)])
@pytest.mark.parametrize("logprob_start", [0, 19])
def test_cross_prefill_passes_preserve_absolute_candidate_positions(
    boundaries, logprob_start
):
    torch.manual_seed(3)
    length, prompt_length, width = 40, 24, 5
    logprobs = torch.randn(length, 100).log_softmax(-1)
    positions = [[] for _ in range(prompt_length)] + [
        list(range(5 + row * width, 5 + (row + 1) * width))
        for row in range(length - prompt_length)
    ]
    union = sorted({token for row in positions for token in row})
    dense, _ = get_token_ids_logprobs_raw(
        logprobs[logprob_start:],
        [union],
        LogprobStage.PREFILL,
        [length - logprob_start],
    )
    gathered_values, gathered_ids = [], []
    starts = [logprob_start, *(b for b in boundaries if b > logprob_start), length]
    for start, end in zip(starts, starts[1:]):
        values, indices = get_token_ids_logprobs_raw(
            logprobs[start:end],
            [PerPositionTokenIds(positions, start=start + 1)],
            LogprobStage.PREFILL,
            [end - start],
        )
        gathered_values.extend(values[0])
        gathered_ids.extend(indices[0])
    # Response assembly discards the trailing next-token row.
    for row, (values, candidates) in enumerate(
        zip(gathered_values[:-1], gathered_ids[:-1])
    ):
        assert candidates == positions[logprob_start + row + 1]
        oracle = dict(zip(union, dense[0][row]))
        assert values == [oracle[token] for token in candidates]
    assert gathered_ids[-1] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
