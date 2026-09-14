"""Reject per-position requests whose positions cannot reach the scorer intact."""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.logprob_types import PerPositionTokenIds
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _request(**overrides):
    fields = dict(
        input_ids=[1, 2, 3],
        token_ids_logprob_positions=[[], [4, 2], [5]],
        return_logprob=True,
        logprob_start_len=0,
        sampling_params={"max_new_tokens": 0},
    )
    request = GenerateReqInput(**{**fields, **overrides})
    request.normalize_batch_and_arguments()
    return request


def _validate(request):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.model_config = SimpleNamespace(vocab_size=8)
    manager._validate_token_ids_logprob(request)


def test_normalized_scoring_request_keeps_absolute_rows():
    request = _request()
    _validate(request)
    assert request.token_ids_logprob == PerPositionTokenIds([[], [4, 2], [5]])
    _validate(request)


@pytest.mark.parametrize(
    "overrides",
    [
        {"token_ids_logprob": [4]},
        {"token_ids_logprob_positions": [[], [4]]},
        {"token_ids_logprob_positions": [[], [4, 4], [5]]},
        {"token_ids_logprob_positions": [[], [8], [5]]},
        {"token_ids_logprob_positions": [[], [-1], [5]]},
        {"token_ids_logprob_positions": [[], [True], [5]]},
        {"token_ids_logprob_positions": [[], 4, [5]]},
        {"return_logprob": False},
        {"sampling_params": {"max_new_tokens": 1}},
        {"logprob_start_len": -1},
        {"stream": True},
        {"multi_item_delimiter_indices": [1, 2]},
    ],
    ids=[
        "flat-and-position",
        "length",
        "duplicate",
        "out-of-vocabulary",
        "negative-id",
        "boolean-id",
        "non-list-row",
        "no-logprobs",
        "decode",
        "negative-start",
        "streaming",
        "multi-item",
    ],
)
def test_unsupported_scoring_request_is_rejected(overrides):
    with pytest.raises(ValueError):
        _validate(_request(**overrides))


def test_batched_positions_are_not_misread_as_flat_request_ids():
    with pytest.raises(ValueError, match="one input sequence"):
        _request(input_ids=[[1, 2], [3, 4]])


def test_flat_requested_ids_keep_the_existing_contract():
    request = _request(token_ids_logprob_positions=None, token_ids_logprob=[4, 2])
    _validate(request)
    assert request.token_ids_logprob == [4, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
