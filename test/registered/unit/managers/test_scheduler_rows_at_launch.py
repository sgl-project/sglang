"""The normal and overlap loops join a step's finished-prefill rows into the
running rows right after its launch, and drop the rows a commit finished right
after that commit, so a reader between steps sees exactly the rows still
decoding."""

import copy
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.decoding_batch import (  # noqa: E402
    DecodingBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.done = False

    def finished(self):
        return self.done


class _Batch:
    """Keeps the row order and the filter and merge rules of ScheduleBatch."""

    def __init__(self, reqs, mode=ForwardMode.DECODE, chunked_req=None):
        self.reqs = list(reqs)
        self.forward_mode = mode
        self.num_decode_rows = 0
        self.chunked_req = chunked_req
        self.sampling_info = SimpleNamespace(penalizer_orchestrator=SimpleNamespace())

    def is_empty(self):
        return not self.reqs

    def has_prefill_rows(self):
        return self.forward_mode.is_extend() and self.num_decode_rows < len(self.reqs)

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        self.reqs = self.reqs + other.reqs

    def shallow_copy(self):
        batch = copy.copy(self)
        batch.reqs = self.reqs[:]
        return batch

    copy = shallow_copy


class _StopLoop(Exception):
    pass


def _rids(reqs):
    return [r.rid for r in reqs]


class _Loop:
    """Drives a real event loop over scripted steps.

    Each step is (planned batch or None, rids its commit finishes). Records
    the running rows a reader sees at the top of each iteration, where intake
    (aborts, load queries, pause) runs.
    """

    def __init__(self, test, steps, rows=()):
        self.test = test
        s = MagicMock()
        self.s, self.readers, self.commits = s, [], []
        s.gracefully_exit = False
        s._engine_paused = False
        s.is_generation = True
        s.enable_hisparse = False
        s.dllm_config = None
        s.decoding_batch = DecodingBatch(_Batch(rows))
        s.is_disable_overlap_for_batch.return_value = False
        for name in (
            "_update_rows_at_launch",
            "_commit_due_step",
            "leave_finished_rows",
        ):
            setattr(s, name, MethodType(getattr(Scheduler, name), s))
        plans = iter(steps)
        finishes = {}

        def ingest_requests():
            self.readers.append(_rids(s.decoding_batch.rows.reqs))
            if len(self.readers) > len(steps):
                raise _StopLoop

        def get_next_batch_to_run(decoding_batch):
            test.assertIs(decoding_batch, s.decoding_batch)
            batch, finished = next(plans)
            if batch is not None:
                finishes[id(batch)] = finished
            return NextBatchPlan(batch_to_run=batch, running_batch=decoding_batch.rows)

        def run_batch(batch):
            return SimpleNamespace(key=id(batch))

        def process_batch_result(batch, result):
            finished = finishes[result.key]
            self.commits.append(_rids(batch.reqs))
            for r in batch.reqs:
                if r.rid in finished:
                    r.done = True

        s.ingest_requests.side_effect = ingest_requests
        s.get_next_batch_to_run.side_effect = get_next_batch_to_run
        s.run_batch.side_effect = run_batch
        s.process_batch_result.side_effect = process_batch_result

    def run(self, event_loop):
        with self.test.assertRaises(_StopLoop):
            event_loop(self.s)


class TestRowsAtLaunchAndCommit(CustomTestCase):
    def _steps(self):
        a, b, c, d = (_Req(rid) for rid in "abcd")
        return [
            # b is still chunking: it stays out of the rows. c finishes at its
            # prefill's commit.
            (_Batch([a, b, c], ForwardMode.EXTEND, chunked_req=b), {"c"}),
            # b's last chunk: it joins after a.
            (_Batch([b, d], ForwardMode.EXTEND), set()),
            (None, set()),
        ]

    def test_normal_loop_joins_at_launch_and_leaves_at_commit(self):
        loop = _Loop(self, self._steps())
        loop.run(Scheduler.event_loop_normal)
        self.assertEqual(loop.readers, [[], ["a"], ["a", "b", "d"], ["a", "b", "d"]])

    def test_overlap_loop_leaves_at_the_commit_one_step_later(self):
        loop = _Loop(self, self._steps())
        loop.run(Scheduler.event_loop_overlap)
        # c joins at its launch and leaves when the next iteration commits it.
        self.assertEqual(
            loop.readers, [[], ["a", "c"], ["a", "b", "d"], ["a", "b", "d"]]
        )
        self.assertEqual(loop.commits, [["a", "b", "c"], ["b", "d"]])

    def test_mixed_step_rows_replace_the_rows_at_launch(self):
        x, y, n = _Req("x"), _Req("y"), _Req("n")
        mixed = _Batch([n, x, y], ForwardMode.MIXED)
        mixed.num_decode_rows = 2
        for event_loop, readers in (
            (Scheduler.event_loop_normal, [["x", "y"], ["n", "x"], ["n", "x"]]),
            # Overlap commits the mixed step one iteration later.
            (Scheduler.event_loop_overlap, [["x", "y"], ["n", "x", "y"], ["n", "x"]]),
        ):
            with self.subTest(event_loop=event_loop.__name__):
                for r in (x, y, n):
                    r.done = False
                loop = _Loop(self, [(mixed, {"y"}), (None, set())], rows=[x, y])
                loop.run(event_loop)
                # The decode rows stay running through the mixed step; after
                # its launch the rows are its rows, new first.
                self.assertEqual(loop.readers, readers)


if __name__ == "__main__":
    unittest.main()
