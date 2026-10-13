import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.decoding_batch import (  # noqa: E402
    DecodingBatch,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Batch:
    """Keeps the row order and the filter and merge rules of ScheduleBatch."""

    def __init__(self, *rids, finished=()):
        self.reqs = [
            SimpleNamespace(rid=rid, finished=lambda rid=rid: rid in finished)
            for rid in rids
        ]

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


if __name__ == "__main__":
    unittest.main()
