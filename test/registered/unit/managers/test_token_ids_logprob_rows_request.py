"""token_ids_logprob_rows: malformed rows are rejected, and each extend of a
chunked or cache-resumed prefill scores its own rows."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.layers.logprob_processor import get_token_ids_logprobs_chunk
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import token_id_rows_for_extend
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MALFORMED_ROWS = {
    "too-few-rows": [[4, 2]],
    "out-of-vocabulary": [[4, 2], [8]],
    "negative-id": [[4, 2], [-1]],
    "non-list-row": [[4, 2], 5],
    "non-list-rows": 5,
}


def _request(**overrides):
    fields = dict(
        input_ids=[1, 2, 3, 4],
        token_ids_logprob_rows=[[4, 2], [5]],
        return_logprob=True,
        logprob_start_len=1,
        sampling_params={"max_new_tokens": 0},
    )
    request = GenerateReqInput(**{**fields, **overrides})
    request.normalize_batch_and_arguments()
    return request


def _validate(request):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.model_config = SimpleNamespace(
        token_id_vocab_size=8, is_encoder_decoder=False
    )
    manager.disaggregation_mode = DisaggregationMode.NULL
    manager._validate_token_ids_logprob_rows(request)


def test_valid_requests():
    _validate(_request())
    _validate(_request(logprob_start_len=0, token_ids_logprob_rows=[[1], [], [7]]))
    _validate(_request(logprob_start_len=3, token_ids_logprob_rows=[]))


@pytest.mark.parametrize("rows", MALFORMED_ROWS.values(), ids=MALFORMED_ROWS.keys())
def test_malformed_rows_are_rejected(rows):
    with pytest.raises(ValueError, match="token_ids_logprob_rows"):
        _validate(_request(token_ids_logprob_rows=rows))


@pytest.mark.parametrize("boundaries", [(), (12, 24, 30)])
@pytest.mark.parametrize("logprob_start_len", [0, 19, 23])
def test_each_extend_scores_its_own_rows(boundaries, logprob_start_len):
    torch.manual_seed(4)
    length, vocab = 40, 100
    logprobs = torch.randn(length, vocab).log_softmax(-1)
    rows = [
        [(7 * token + k) % vocab for k in range(1 + token % 4)]
        for token in range(logprob_start_len + 1, length)
    ]
    starts = [0, *boundaries, length]
    values, indices = [], []
    for prefix_len, end in zip(starts, starts[1:]):
        first_scored_pos = min(max(logprob_start_len, prefix_len), end)
        num_scored = end - first_scored_pos
        scored = token_id_rows_for_extend(
            rows,
            logprob_start_len=logprob_start_len,
            first_scored_pos=first_scored_pos,
            num_scored=num_scored,
        )
        val, idx = [], []
        get_token_ids_logprobs_chunk(
            logprobs[first_scored_pos:end], [scored], [num_scored], val, idx
        )
        values.extend(val[0])
        indices.extend(idx[0])
    # The last position predicts the token after the input, which has no row.
    assert indices == rows + [[]]
    for i, token_ids in enumerate(rows):
        position = logprob_start_len + i
        assert values[i] == pytest.approx(logprobs[position, token_ids].tolist())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
