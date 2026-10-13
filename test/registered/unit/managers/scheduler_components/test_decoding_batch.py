import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler_components.decoding_batch import (  # noqa: E402
    DecodingBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.sampling.penaltylib.orchestrator import (  # noqa: E402
    BatchedPenalizerOrchestrator,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Batch:
    """Keeps the row order and the filter and merge rules of ScheduleBatch."""

    def __init__(self, *rids, finished=()):
        self.reqs = [
            SimpleNamespace(rid=rid, finished=lambda rid=rid: rid in finished)
            for rid in rids
        ]
        self.sampling_info = SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(batch=self)
        )

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        self.reqs = self.reqs + other.reqs

    def prepare_for_decode(self):
        pass

    def retract_decode(self):
        retracted, self.reqs = self.reqs[-1:], self.reqs[:-1]
        return retracted, 0.5, []


def _rids(batch):
    return [r.rid for r in batch.reqs]


class TestDecodingBatch(CustomTestCase):
    def test_join_appends_after_the_rows_and_empty_rows_take_the_batch(self):
        decoding_batch = DecodingBatch(_Batch())
        first = _Batch("a", "b")
        decoding_batch.join(first)
        self.assertIs(decoding_batch.rows, first)

        decoding_batch.join(_Batch("c"))
        self.assertIs(decoding_batch.rows, first)
        self.assertEqual(_rids(first), ["a", "b", "c"])

    def test_leave_drops_the_given_and_the_finished_rows(self):
        rows = _Batch("a", "b", "c", "d", finished={"c"})
        decoding_batch = DecodingBatch(rows)
        decoding_batch.leave([rows.reqs[0]])
        self.assertEqual(_rids(rows), ["b", "d"])

    def test_park_for_mix_hands_the_rows_to_the_mixed_step(self):
        rows = _Batch("a", "b")
        decoding_batch = DecodingBatch(rows)
        decoding_batch.park_for_mix()
        self.assertIsNot(decoding_batch.rows, rows)
        self.assertTrue(decoding_batch.rows.is_empty())
        self.assertEqual(_rids(rows), ["a", "b"])

    def test_in_place_methods_keep_the_rows_object(self):
        # select_decode_rows and the pipeline slots hold the rows object
        # across these calls.
        rows = _Batch("a", "b", "c", finished={"b"})
        decoding_batch = DecodingBatch(rows)
        decoding_batch.leave_finished()
        decoding_batch.advance()
        retracted, _, _ = decoding_batch.retract()
        self.assertEqual([r.rid for r in retracted], ["c"])
        decoding_batch.clear_for_pause()
        self.assertIs(decoding_batch.rows, rows)
        self.assertTrue(rows.is_empty())


def _schedule_batch(rids):
    batch = ScheduleBatch(
        reqs=[SimpleNamespace(rid=rid) for rid in rids],
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.tensor([5] * len(rids)),
        input_ids=torch.tensor([1] * len(rids)),
        return_logprob=True,
        top_logprobs_nums=[2] * len(rids),
        prefix_lens=[3] * len(rids),
        spec_info=object(),
    )
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=BatchedPenalizerOrchestrator(
            vocab_size=8, batch=batch, penalizers=set()
        )
    )
    return batch


def _assert_same_fields(test, batch, source):
    for field in dataclasses.fields(ScheduleBatch):
        if field.name != "reqs":
            test.assertIs(getattr(batch, field.name), getattr(source, field.name))
    test.assertIsNot(batch.reqs, source.reqs)
    test.assertEqual(batch.reqs, source.reqs)


class TestDecodingBatchRowCopy(CustomTestCase):
    def test_join_into_empty_rows_takes_the_copy_wholesale(self):
        launched = _schedule_batch(["a", "b"])
        copy = launched.shallow_copy()
        _assert_same_fields(self, copy, launched)

        decoding_batch = DecodingBatch(ScheduleBatch(reqs=[]))
        decoding_batch.join(copy)
        self.assertIs(decoding_batch.rows, copy)
        orchestrator = copy.sampling_info.penalizer_orchestrator
        self.assertIs(orchestrator.batch, copy)
        # The penalizers outlive the batch the rows were copied from.
        del launched
        self.assertEqual(orchestrator.reqs(), copy.reqs)

    def test_absorb_takes_back_what_the_decode_step_rebound(self):
        rows = _schedule_batch(["a", "b"])
        decoding_batch = DecodingBatch(rows)
        step = decoding_batch.decode_step()
        self.assertIsNot(step, rows)
        _assert_same_fields(self, step, rows)

        step.seq_lens = torch.tensor([6, 6])
        step.input_ids = None
        step.spec_info = object()
        decoding_batch.absorb()
        self.assertIs(decoding_batch.rows, rows)
        _assert_same_fields(self, rows, step)

        # Absorbed once: a later write to the step stays on the step.
        step.seq_lens = torch.tensor([7, 7])
        decoding_batch.absorb()
        self.assertEqual(rows.seq_lens.tolist(), [6, 6])

    def test_rows_stay_decode_after_the_dp_extend_view(self):
        rows = _schedule_batch(["a", "b"])
        decoding_batch = DecodingBatch(rows)
        step = decoding_batch.decode_step()
        step.forward_mode = ForwardMode.EXTEND
        step.prefix_lens = [4, 4]
        step.extend_lens = [1, 1]
        step.num_decode_rows = 2
        step.seq_lens = torch.tensor([6, 6])
        prefix_lens = rows.prefix_lens

        decoding_batch.absorb()
        self.assertEqual(rows.forward_mode, ForwardMode.DECODE)
        self.assertIs(rows.prefix_lens, prefix_lens)
        self.assertIsNone(rows.extend_lens)
        self.assertEqual(rows.num_decode_rows, 0)
        self.assertIs(rows.seq_lens, step.seq_lens)

    def test_debug_check_rejects_a_rows_write_during_the_step(self):
        rows = _schedule_batch(["a", "b"])
        decoding_batch = DecodingBatch(rows)
        with envs.SGLANG_DEBUG_INFLIGHT_STEPS.override(True):
            decoding_batch.decode_step()
            rows.reqs = rows.reqs[:1]
            with self.assertRaisesRegex(AssertionError, "rows changed"):
                decoding_batch.absorb()

    def test_debug_check_rejects_a_leaked_extend_mode(self):
        decoding_batch = DecodingBatch(_schedule_batch(["a", "b"]))
        with envs.SGLANG_DEBUG_INFLIGHT_STEPS.override(True):
            step = decoding_batch.decode_step()
            step.forward_mode = ForwardMode.TARGET_VERIFY
            with self.assertRaisesRegex(AssertionError, "came back as"):
                decoding_batch.absorb()

            # The DP extend view converts every row and passes.
            step = decoding_batch.decode_step()
            step.forward_mode = ForwardMode.EXTEND
            step.num_decode_rows = 2
            decoding_batch.absorb()

    def test_pause_drops_the_decode_step(self):
        rows = _schedule_batch(["a"])
        decoding_batch = DecodingBatch(rows)
        decoding_batch.decode_step()
        decoding_batch.clear_for_pause()
        decoding_batch.absorb()
        self.assertTrue(rows.is_empty())


if __name__ == "__main__":
    unittest.main()
