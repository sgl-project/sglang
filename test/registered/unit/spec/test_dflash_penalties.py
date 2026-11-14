"""Sampling penalties under DFLASH-family speculative decoding — no server, no model."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.repetition_penalty_relay import RepetitionPenaltyRelay
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
    apply_repetition_penalty_with_prefix,
)
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 8
BLOCK = 3
BS = 2


class _ForwardSamplingInfo(SimpleNamespace):
    """The forward-only copy of SamplingBatchInfo seen by the verify step."""

    def __init__(self, **fields):
        values = dict(
            has_custom_logit_processor=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=None,
            grammar_mask=None,
            logit_bias=None,
        )
        values.update(fields)
        super().__init__(**values)

    def __len__(self):
        return BS


class _BanTokens:
    """Grammar mask stand-in: the same tokens are banned for every request."""

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def apply(self, logits):
        logits[:, self.token_ids] = float("-inf")


def _make_req(freq=0.0, repetition=1.0, output_ids=()):
    req = MagicMock()
    req.sampling_params.frequency_penalty = freq
    req.sampling_params.repetition_penalty = repetition
    req.output_ids = array("q", output_ids)
    req.penalizer_cumulated_len = 0
    return req


def _orchestrator(reqs):
    batch = MagicMock()
    batch.reqs = reqs
    batch.device = "cpu"
    return BatchedPenalizerOrchestrator(
        VOCAB_SIZE, batch, {BatchedFrequencyPenalizer, BatchedRepetitionPenalizer}
    )


def _verify_logits():
    # Mixed signs, so the scaling penalty divides some logits and multiplies others.
    return torch.randn(
        BS * BLOCK, VOCAB_SIZE, generator=torch.Generator().manual_seed(0)
    )


def _penalized(logits, additive, scaling):
    """Reference: apply request i's penalties to each row of its verify block."""
    rows = logits.clone().view(BS, BLOCK, VOCAB_SIZE) + additive[:, None, :]
    rows = torch.where(rows < 0, rows * scaling[:, None, :], rows / scaling[:, None, :])
    return rows.view(BS * BLOCK, VOCAB_SIZE)


class TestDFlashVerifyPenalties(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.additive = torch.zeros(BS, VOCAB_SIZE)
        self.additive[0, 1], self.additive[1, 4] = -2.0, -3.0
        self.scaling = torch.ones(BS, VOCAB_SIZE)
        self.scaling[0, 2], self.scaling[1, 5] = 1.5, 2.0

    def test_accumulated_penalties_reach_every_row_of_the_block(self):
        logits = _verify_logits()
        expected = _penalized(logits, self.additive, self.scaling)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=_ForwardSamplingInfo(
                acc_additive_penalties=self.additive,
                acc_scaling_penalties=self.scaling,
            ),
            draft_token_num=BLOCK,
        )
        torch.testing.assert_close(logits, expected)

    def test_scaling_penalty_is_kept_under_a_grammar_mask(self):
        logits = _verify_logits()
        expected = _penalized(logits, self.additive, self.scaling)
        expected[:, 7] = float("-inf")
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=_ForwardSamplingInfo(
                acc_additive_penalties=self.additive,
                acc_scaling_penalties=self.scaling,
                grammar_mask=_BanTokens([7]),
            ),
            draft_token_num=BLOCK,
        )
        torch.testing.assert_close(logits, expected)

    def test_live_orchestrator_is_used_without_accumulated_buffers(self):
        orch = _orchestrator([_make_req(freq=0.5), _make_req(repetition=2.0)])
        orch.cumulate_output_tokens(torch.tensor([3, 6]))
        logits = _verify_logits()
        expected = logits.clone().view(BS, BLOCK, VOCAB_SIZE)
        for position in range(BLOCK):
            rows = expected[:, position, :].clone()
            orch.apply(rows)
            expected[:, position, :] = rows
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=_ForwardSamplingInfo(penalizer_orchestrator=orch),
            draft_token_num=BLOCK,
        )
        torch.testing.assert_close(logits, expected.view(BS * BLOCK, VOCAB_SIZE))

    def test_dspark_fast_path_respects_accumulated_penalties(self):
        self.assertTrue(verify_logits_adjustments_are_noop(_ForwardSamplingInfo()))
        self.assertFalse(
            verify_logits_adjustments_are_noop(
                _ForwardSamplingInfo(acc_additive_penalties=self.additive)
            )
        )
        self.assertFalse(
            verify_logits_adjustments_are_noop(
                _ForwardSamplingInfo(acc_scaling_penalties=self.scaling)
            )
        )


class TestCommittedOutputTokens(CustomTestCase):
    def _batch(self, reqs):
        orch = _orchestrator(reqs)
        batch = SimpleNamespace(
            reqs=reqs,
            device="cpu",
            sampling_info=SimpleNamespace(penalizer_orchestrator=orch),
        )
        return batch, orch.penalizers[BatchedFrequencyPenalizer]

    def test_each_committed_token_is_fed_once(self):
        reqs = [
            _make_req(freq=1.0, output_ids=[5, 5, 2]),
            _make_req(freq=1.0, output_ids=[3]),
        ]
        batch, freq = self._batch(reqs)

        ScheduleBatch.cumulate_penalty_committed_output_tokens(batch)
        counts = freq.cumulated_frequency_penalties
        self.assertEqual(counts[0, [5, 2]].tolist(), [2.0, 1.0])
        self.assertEqual(counts[1, 3].item(), 1.0)
        self.assertEqual(counts[1, 0].item(), 0.0)  # padding id
        self.assertEqual([r.penalizer_cumulated_len for r in reqs], [3, 1])

        # Next step: row 0 commits nothing, row 1 commits a block of four.
        reqs[1].output_ids.extend([3, 4, 4, 4])
        ScheduleBatch.cumulate_penalty_committed_output_tokens(batch)
        self.assertEqual(counts[0, [5, 2]].tolist(), [2.0, 1.0])
        self.assertEqual(counts[1, [3, 4]].tolist(), [2.0, 3.0])
        self.assertEqual([r.penalizer_cumulated_len for r in reqs], [3, 5])

        # Nothing new anywhere: no change.
        ScheduleBatch.cumulate_penalty_committed_output_tokens(batch)
        self.assertEqual(counts.sum().item(), 8.0)

    def test_dflash_decode_prep_feeds_committed_tokens_when_penalized(self):
        exec_ns = SimpleNamespace(
            mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
        )
        for is_dflash_family, is_required, fed in (
            (True, True, True),
            (True, False, False),
        ):
            batch = MagicMock()
            batch.spec_algorithm.is_dflash_family.return_value = is_dflash_family
            batch.sampling_info.penalizer_orchestrator.is_required = is_required
            with patch.object(spec_utils, "get_exec", return_value=exec_ns):
                spec_utils.spec_prepare_for_decode(batch)
            self.assertEqual(batch.cumulate_penalty_committed_output_tokens.called, fed)
            batch.spec_info.prepare_for_decode.assert_called_once_with(batch)


class TestSequentialRepetitionPenalty(CustomTestCase):
    """Reference each verify row against the ordinary one-token penalizer."""

    def _history(self, histories, penalties):
        reqs = [_make_req(repetition=p) for p in penalties]
        orch = _orchestrator(reqs)
        for row, history in enumerate(histories):
            for token in history:
                ids = torch.zeros(len(reqs), dtype=torch.int64)
                ids[row] = token
                valid = torch.arange(len(reqs)) == row
                orch.cumulate_output_tokens_padded(ids[:, None], valid[:, None])
        return orch

    def _reference(self, raw, histories, penalties, inputs):
        orch = self._history(histories, penalties)
        result = raw.clone()
        for position in range(inputs.shape[1]):
            orch.cumulate_output_tokens(inputs[:, position])
            rows = result[:, position, :].clone()
            orch.apply(rows)
            result[:, position, :] = rows
        return result

    def test_matches_ordinary_decode_at_every_position(self):
        # Row zero sees 3 for the first time *inside* this block; row one
        # repeats historical token 5, which must never be penalized twice.
        inputs = torch.tensor([[2, 3, 3, 6], [4, 5, 5, 1]])
        histories = [[2], [5, 4]]
        for penalties in ([1.5, 2.0], [0.7, 0.5], [1.0, 1.6]):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(penalties=penalties, dtype=dtype):
                    raw = torch.linspace(-4, 4, 64).reshape(2, 4, 8).to(dtype)
                    expected = self._reference(raw, histories, penalties, inputs)
                    orch = self._history(histories, penalties)
                    state = orch.accumulate_scaling_penalties().clone()
                    info = _ForwardSamplingInfo(
                        acc_scaling_penalties=state,
                        repetition_penalties=torch.tensor(penalties)[:, None],
                    )
                    actual = raw.clone()
                    apply_dflash_verify_logits_adjustments(
                        next_token_logits=actual.view(-1, 8),
                        sampling_info=info,
                        draft_token_num=4,
                        verify_input_ids=inputs,
                    )
                    torch.testing.assert_close(actual, expected)
                    # Verify must not commit hypothetical drafts into history.
                    torch.testing.assert_close(
                        orch.accumulate_scaling_penalties(), state
                    )
                    torch.testing.assert_close(info.acc_scaling_penalties, state)
                    torch.testing.assert_close(
                        actual.float().softmax(-1), expected.float().softmax(-1)
                    )
                    self.assertEqual(
                        actual.argmax(-1).tolist(), expected.argmax(-1).tolist()
                    )

    def test_live_orchestrator_uses_the_same_position_histories(self):
        inputs = torch.tensor([[2, 3, 3], [4, 5, 5]])
        histories, penalties = [[2], [4]], [1.5, 0.7]
        raw = _verify_logits().view(2, 3, 8)
        expected = self._reference(raw, histories, penalties, inputs)
        orch = self._history(histories, penalties)
        actual = raw.clone()
        apply_dflash_verify_logits_adjustments(
            next_token_logits=actual.view(-1, 8),
            sampling_info=_ForwardSamplingInfo(penalizer_orchestrator=orch),
            draft_token_num=3,
            verify_input_ids=inputs,
        )
        torch.testing.assert_close(actual, expected)

    def test_anchor_and_bonus_do_not_see_future_drafts(self):
        raw = torch.tensor([[[4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]]]).repeat(
            2, 3, 1
        )
        histories, penalties = [[], []], [2.0, 0.5]
        inputs = torch.tensor([[0, 1, 1], [3, 2, 2]])
        actual = raw.clone()
        apply_repetition_penalty_with_prefix(
            actual, torch.ones(2, 8), torch.tensor(penalties), inputs
        )
        expected = self._reference(raw, histories, penalties, inputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[0, 0, 1].item(), raw[0, 0, 1].item())
        self.assertEqual(actual[0, 1, 1].item(), raw[0, 1, 1].item() / 2)
        self.assertEqual(actual[0, 2, 1].item(), raw[0, 2, 1].item() / 2)

    def test_overlap_matches_sync_after_any_accept_length(self):
        penalties = [1.5, 0.7]
        histories = [[2], [4]]
        # Emitted tokens include replacement/bonus. The suffix is not emitted
        # and may even contain invalid IDs: it must never enter the seen set.
        emitted = torch.tensor([[3, 6, -1, 999], [5, 1, 7, 2]])
        inputs = torch.tensor([[6, 3, 3], [2, 5, 5]])
        for lengths in ([1, 1], [2, 3], [2, 4]):
            with self.subTest(lengths=lengths):
                # Keep the invalid padding outside the committed prefix.
                committed = [
                    emitted[row, :length].tolist() for row, length in enumerate(lengths)
                ]
                inputs[:, 0] = torch.tensor([tokens[-1] for tokens in committed])
                fresh = [h + c for h, c in zip(histories, committed)]
                raw = _verify_logits().view(2, 3, 8)
                expected = self._reference(raw, fresh, penalties, inputs)
                relay = RepetitionPenaltyRelay(4, "cpu")
                indices = torch.tensor([3, 1])
                relay.stash(indices, emitted, torch.tensor(lengths))
                pending, lens = relay.resolve(indices)
                # CPU snapshot may contain none, part, or all of that block.
                # Membership union must give the same answer in every case.
                for snapshot_histories in (histories, fresh):
                    actual = raw.clone()
                    orch = self._history(snapshot_histories, penalties)
                    info = _ForwardSamplingInfo(
                        acc_scaling_penalties=orch.accumulate_scaling_penalties(),
                        repetition_penalties=torch.tensor(penalties),
                        repetition_penalty_tokens=pending,
                        repetition_penalty_lens=lens,
                    )
                    apply_dflash_verify_logits_adjustments(
                        next_token_logits=actual.view(-1, 8),
                        sampling_info=info,
                        draft_token_num=3,
                        verify_input_ids=inputs,
                    )
                    torch.testing.assert_close(actual, expected)

    def test_prefix_penalty_preserves_grammar_and_logit_bias(self):
        inputs = torch.tensor([[2, 3, 3], [4, 5, 5]])
        histories, penalties = [[2], [4]], [1.5, 0.7]
        raw = _verify_logits().view(2, 3, 8)
        bias = torch.arange(8).float().repeat(2, 1) / 10
        expected = self._reference(raw, histories, penalties, inputs)
        expected[:, :, 7] = float("-inf")
        expected += bias[:, None, :]
        orch = self._history(histories, penalties)
        actual = raw.clone()
        apply_dflash_verify_logits_adjustments(
            next_token_logits=actual.view(-1, 8),
            sampling_info=_ForwardSamplingInfo(
                acc_scaling_penalties=orch.accumulate_scaling_penalties(),
                repetition_penalties=torch.tensor(penalties),
                grammar_mask=_BanTokens([7]),
                logit_bias=bias,
            ),
            draft_token_num=3,
            verify_input_ids=inputs,
        )
        torch.testing.assert_close(actual, expected)


class TestRepetitionPenaltyRelay(CustomTestCase):
    def test_scheduler_result_reaches_the_next_forward_snapshot(self):
        from sglang.srt.managers.overlap_utils import FutureMap
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        for algorithm in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
            with self.subTest(algorithm=algorithm):
                # Use the real stash/resolve methods, without allocating model/KV
                # state. No CPU result processing occurs between the two calls.
                future = FutureMap.__new__(FutureMap)
                future.spec_algo = algorithm
                future.relay_bypass = False
                future._deferred_relay = None
                future._relay = None
                future.repetition_penalty_relay = RepetitionPenaltyRelay(5, "cpu")
                future.output_tokens_buf = torch.zeros(5, dtype=torch.int64)
                future._maybe_init_forward_bufs = lambda payload: None
                future._maybe_init_dsa_topk_indices_buf = lambda payload: None
                future.need_topk = future.need_hidden_states = False
                future.draft_probs_buf = future.dsa_topk_indices_buf = None
                scheduler = SimpleNamespace(
                    spec_algorithm=algorithm,
                    future_map=future,
                    beam_coordinator=MagicMock(),
                    chunked_req=None,
                    uocr_runtime=None,
                )
                reqs = [_make_req(repetition=1.5), _make_req(repetition=0.7)]
                indices = torch.tensor([4, 1])
                batch = SimpleNamespace(reqs=reqs, beam_tail=None)
                result = SimpleNamespace(
                    next_token_ids=torch.tensor([3, 2, -1, 5, 6, 4]),
                    accept_lens=torch.tensor([2, 3]),
                    next_draft_input=SimpleNamespace(
                        bonus_tokens=torch.tensor([2, 4]),
                        topk_p=None,
                        topk_index=None,
                        hidden_states=None,
                        dsa_topk_indices=None,
                    ),
                )
                Scheduler._relay_forward_payload(scheduler, batch, indices, result)
                self.assertEqual([len(req.output_ids) for req in reqs], [0, 0])

                # Next iteration reverses batch order. resolve_forward_inputs
                # runs before _forward_isolation/copy_for_forward.
                batch.req_pool_indices = indices.flip(0)
                batch.spec_info = None
                orch = _orchestrator(reqs[::-1])
                batch.sampling_info = SamplingBatchInfo(
                    temperatures=torch.ones(2, 1),
                    top_ps=torch.ones(2),
                    top_ks=torch.ones(2, dtype=torch.int32),
                    min_ps=torch.zeros(2),
                    is_all_greedy=True,
                    is_any_greedy=True,
                    need_top_p_sampling=False,
                    need_top_k_sampling=False,
                    need_min_p_sampling=False,
                    vocab_size=8,
                    penalizer_orchestrator=orch,
                    device="cpu",
                )
                FutureMap._resolve_spec_extras(future, batch)
                info = batch.sampling_info.copy_for_forward()
                self.assertIsNone(info.penalizer_orchestrator)
                self.assertEqual(info.repetition_penalty_lens.tolist(), [3, 2])
                raw = _verify_logits().view(2, 3, 8)
                inputs = torch.tensor([[4, 5, 5], [2, 3, 3]])
                expected = TestSequentialRepetitionPenalty()._reference(
                    raw, [[5, 6, 4], [3, 2]], [0.7, 1.5], inputs
                )
                actual = raw.clone()
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=actual.view(-1, 8),
                    sampling_info=info,
                    draft_token_num=3,
                    verify_input_ids=inputs,
                )
                torch.testing.assert_close(actual, expected)

    def test_filter_merge_grow_and_slot_reuse(self):
        relay = RepetitionPenaltyRelay(5, "cpu")
        # Prefill's sampled token, followed by a wider accepted decode block.
        relay.stash(torch.tensor([4]), torch.tensor([[6]]))
        relay.stash(torch.tensor([1]), torch.tensor([[2, 3, -1]]), torch.tensor([2]))
        tokens, lens = relay.resolve(torch.tensor([1, 4]))
        self.assertEqual(lens.tolist(), [2, 1])
        self.assertEqual(tokens[:, 0].tolist(), [2, 6])
        # Filtering/reordering uses request slots, not the previous batch order.
        kept, lengths = relay.resolve(torch.tensor([4]))
        self.assertEqual(kept[0, 0].item(), 6)
        self.assertEqual(lengths.tolist(), [1])
        # Slot 1 is reused without a penalty; old tokens must become invisible.
        relay.stash(torch.tensor([1]))
        _, lengths = relay.resolve(torch.tensor([1, 4]))
        self.assertEqual(lengths.tolist(), [0, 1])
        # An already resolved snapshot is independent of subsequent stashes.
        relay.stash(torch.tensor([4]), torch.tensor([[7]]))
        self.assertEqual(kept[0, 0].item(), 6)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_previous_block_is_visible_on_the_forward_stream(self):
        device = "cuda"
        relay = RepetitionPenaltyRelay(4, device)
        forward = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        indices = torch.tensor([3, 1], device=device)
        inputs = torch.tensor([[2, 3, 3], [4, 5, 5]], device=device)
        penalties = torch.tensor([1.5, 0.7], device=device)
        raw = _verify_logits().view(2, 3, 8).to(device)
        expected = raw.clone()
        history = torch.ones(2, 8, device=device)
        history[0, [2, 6]] = penalties[0]
        history[1, [4, 7]] = penalties[1]
        apply_repetition_penalty_with_prefix(expected, history, penalties, inputs)
        forward.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(forward):
            relay.stash(
                indices,
                torch.tensor([[6, 2, -1], [7, 4, -1]], device=device),
                torch.tensor([2, 2], device=device),
            )
        consumer.wait_stream(forward)
        with torch.cuda.stream(consumer):
            pending, lengths = relay.resolve(indices)
            actual = raw.clone()
            apply_repetition_penalty_with_prefix(
                actual,
                torch.ones(2, 8, device=device),
                penalties,
                inputs,
                pending,
                lengths,
            )
        torch.cuda.current_stream().wait_stream(consumer)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
