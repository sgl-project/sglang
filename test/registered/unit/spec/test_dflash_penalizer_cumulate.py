import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import spec_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(
    *,
    repetition_penalty=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    min_new_tokens=0,
):
    return SimpleNamespace(
        sampling_params=SamplingParams(
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            min_new_tokens=min_new_tokens,
        ),
        tokenizer=SimpleNamespace(
            additional_stop_token_ids=None,
            eos_token_id=2,
        ),
        eos_token_ids=None,
        penalty_cumulated_len=0,
    )


def _make_batch(reqs):
    class FakeBatch:
        pass

    batch = FakeBatch()
    batch.reqs = reqs
    batch.device = torch.device("cpu")
    return batch


def _make_orchestrator(reqs, vocab_size=16):
    return BatchedPenalizerOrchestrator(
        vocab_size,
        _make_batch(reqs),
        {
            BatchedFrequencyPenalizer,
            BatchedMinNewTokensPenalizer,
            BatchedPresencePenalizer,
            BatchedRepetitionPenalizer,
        },
    )


class TestDFlashPenalizerCumulate(CustomTestCase):
    def test_multi_token_cumulate_updates_all_penalties_and_masks_padding(self):
        reqs = [
            _make_req(
                repetition_penalty=1.5,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                min_new_tokens=4,
            ),
            _make_req(
                repetition_penalty=1.5,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                min_new_tokens=4,
            ),
        ]
        orchestrator = _make_orchestrator(reqs)
        orchestrator.cumulate_output_tokens_multi(
            torch.tensor([[5, 7, 7], [9, 0, 0]], dtype=torch.int64),
            torch.tensor([3, 1], dtype=torch.int64),
        )

        repetition = orchestrator.penalizers[BatchedRepetitionPenalizer]
        frequency = orchestrator.penalizers[BatchedFrequencyPenalizer]
        presence = orchestrator.penalizers[BatchedPresencePenalizer]
        min_new = orchestrator.penalizers[BatchedMinNewTokensPenalizer]

        assert repetition.cumulated_repetition_penalties[0, 5] == 1.5
        assert repetition.cumulated_repetition_penalties[0, 7] == 1.5
        assert repetition.cumulated_repetition_penalties[1, 9] == 1.5
        assert repetition.cumulated_repetition_penalties[1, 0] == 1.0
        assert frequency.cumulated_frequency_penalties[0, 5] == 0.3
        assert frequency.cumulated_frequency_penalties[0, 7] == 0.6
        assert frequency.cumulated_frequency_penalties[1, 9] == 0.3
        assert frequency.cumulated_frequency_penalties[1, 0] == 0.0
        assert presence.cumulated_presence_penalties[0, 5] == 0.2
        assert presence.cumulated_presence_penalties[0, 7] == 0.2
        assert presence.cumulated_presence_penalties[1, 9] == 0.2
        assert presence.cumulated_presence_penalties[1, 0] == 0.0
        assert torch.equal(
            min_new.len_output_tokens[:, 0], torch.tensor([3, 1], dtype=torch.int32)
        )

    def test_multi_token_matches_single_token_cumulate(self):
        reqs = [
            _make_req(
                repetition_penalty=1.5,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                min_new_tokens=4,
            ),
            _make_req(
                repetition_penalty=1.5,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                min_new_tokens=4,
            ),
        ]
        multi = _make_orchestrator(reqs)
        single = _make_orchestrator(
            [
                _make_req(
                    repetition_penalty=1.5,
                    frequency_penalty=0.3,
                    presence_penalty=0.2,
                    min_new_tokens=4,
                ),
                _make_req(
                    repetition_penalty=1.5,
                    frequency_penalty=0.3,
                    presence_penalty=0.2,
                    min_new_tokens=4,
                ),
            ]
        )
        ids = torch.tensor([[5, 7], [9, 9]], dtype=torch.int64)
        multi.cumulate_output_tokens_multi(ids, torch.tensor([2, 2]))
        for column in range(ids.shape[1]):
            single.cumulate_output_tokens(ids[:, column])

        for penalizer_type, attrs in (
            (
                BatchedRepetitionPenalizer,
                ("cumulated_repetition_penalties",),
            ),
            (BatchedFrequencyPenalizer, ("cumulated_frequency_penalties",)),
            (BatchedPresencePenalizer, ("cumulated_presence_penalties",)),
            (BatchedMinNewTokensPenalizer, ("len_output_tokens",)),
        ):
            for attr in attrs:
                assert torch.equal(
                    getattr(multi.penalizers[penalizer_type], attr),
                    getattr(single.penalizers[penalizer_type], attr),
                )

    def test_schedule_batch_penalty_cursor_feeds_only_new_tokens(self):
        req0 = SimpleNamespace(
            origin_input_ids=[10, 42],
            output_ids=[1, 2, 3],
            penalty_cumulated_len=0,
        )
        req1 = SimpleNamespace(
            origin_input_ids=[20, 43], output_ids=[9], penalty_cumulated_len=0
        )
        orchestrator = MagicMock()

        class FakeBatch:
            pass

        batch = FakeBatch()
        batch.reqs = [req0, req1]
        batch.device = torch.device("cpu")
        batch.sampling_info = SimpleNamespace(penalizer_orchestrator=orchestrator)

        ScheduleBatch.cumulate_penalty_output_tokens_since_last(batch)
        ids, num_valid = orchestrator.cumulate_output_tokens_multi.call_args.args
        assert torch.equal(ids, torch.tensor([[42, 1, 2, 3], [43, 9, 0, 0]]))
        assert torch.equal(num_valid, torch.tensor([4, 2]))

        orchestrator.reset_mock()
        req0.output_ids.extend([4, 5])
        ScheduleBatch.cumulate_penalty_output_tokens_since_last(batch)
        ids, num_valid = orchestrator.cumulate_output_tokens_multi.call_args.args
        assert torch.equal(ids, torch.tensor([[4, 5], [0, 0]]))
        assert torch.equal(num_valid, torch.tensor([2, 0]))

        orchestrator.reset_mock()
        ScheduleBatch.cumulate_penalty_output_tokens_since_last(batch)
        orchestrator.cumulate_output_tokens_multi.assert_not_called()

    def test_spec_prepare_for_decode_gates_penalty_cumulate(self):
        calls = []
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
            spec_info=SimpleNamespace(
                prepare_for_decode=lambda _: calls.append("prepare")
            ),
            cumulate_penalty_output_tokens_since_last=lambda: calls.append("cumulate"),
        )

        with patch.object(
            spec_utils,
            "get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
            ),
        ):
            spec_utils.spec_prepare_for_decode(batch)
        assert calls == ["prepare"]

        calls.clear()
        batch.sampling_info.penalizer_orchestrator.is_required = True
        with patch.object(
            spec_utils,
            "get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
            ),
        ):
            spec_utils.spec_prepare_for_decode(batch)
        assert calls == ["cumulate", "prepare"]


if __name__ == "__main__":
    unittest.main()
