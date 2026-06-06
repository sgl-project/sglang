from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.sampling_params import SamplingParams


class FakeBatch:
    pass


def _sampling_params(**kwargs):
    params = SamplingParams(**kwargs)
    params.normalize(tokenizer=None)
    return params


def _req(params, *, eos_token_ids=None):
    req = Req(
        rid="rid",
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=params,
        eos_token_ids=eos_token_ids,
        vocab_size=128,
    )
    req.tokenizer = SimpleNamespace(
        eos_token_id=99,
        additional_stop_token_ids=None,
        decode=lambda ids: "stop",
    )
    return req


def test_min_new_tokens_penalizer_masks_all_token_stop_sources():
    params = _sampling_params(
        max_new_tokens=8,
        min_new_tokens=2,
        stop_token_ids=[42],
    )
    req = SimpleNamespace(
        sampling_params=params,
        tokenizer=SimpleNamespace(
            eos_token_id=43,
            additional_stop_token_ids={44},
        ),
        eos_token_ids={45},
    )
    batch = FakeBatch()
    batch.reqs = [req]
    batch.device = "cpu"

    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=128,
        batch=batch,
        penalizers={BatchedMinNewTokensPenalizer},
    )
    logits = torch.zeros((1, 128), dtype=torch.float32)

    orchestrator.apply(logits)

    for token_id in (42, 43, 44, 45):
        assert logits[0, token_id].item() == float("-inf")
    assert logits[0, 46].item() == 0


def test_update_finish_state_ignores_token_stop_before_min_new_tokens():
    req = _req(
        _sampling_params(
            max_new_tokens=8,
            min_new_tokens=2,
            stop_token_ids=[42],
        )
    )

    req.output_ids = [42]
    req.update_finish_state()
    assert req.finished_reason is None

    req.output_ids.append(42)
    req.update_finish_state()
    assert req.finished_reason.to_json() == {"type": "stop", "matched": 42}
    assert req.finished_len == 2


def test_update_finish_state_skips_early_stop_inside_speculative_acceptance():
    req = _req(
        _sampling_params(
            max_new_tokens=8,
            min_new_tokens=2,
            stop_token_ids=[42],
        )
    )

    req.output_ids = [42, 42]
    req.update_finish_state(new_accepted_len=2)

    assert req.finished_reason.to_json() == {"type": "stop", "matched": 42}
    assert req.finished_len == 2


def test_update_finish_state_does_not_carry_early_stop_to_later_token():
    req = _req(
        _sampling_params(
            max_new_tokens=8,
            min_new_tokens=2,
            stop_token_ids=[42],
        )
    )

    req.output_ids = [42, 7]
    req.update_finish_state(new_accepted_len=2)

    assert req.finished_reason is None


def test_update_finish_state_ignores_string_stop_before_min_new_tokens():
    req = _req(
        _sampling_params(
            max_new_tokens=8,
            min_new_tokens=2,
            stop="stop",
        )
    )

    req.output_ids = [7]
    req.update_finish_state()
    assert req.finished_reason is None

    req.output_ids.append(8)
    req.update_finish_state()
    assert req.finished_reason.to_json() == {"type": "stop", "matched": "stop"}
