"""A pipeline slot's running rows change when the ring commits the step the
slot launched: the output relay rebinds the step's fields at that commit, so
the rows join from the step as the commit left it, and the rows the commit
finished leave there. A request chunked across slots joins the slot whose step
ran its last chunk."""

import copy
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.decoding_batch import DecodingBatch
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.done = False

    def finished(self):
        return self.done


class _Batch:
    """Keeps the row order and the filter and merge rules of ScheduleBatch;
    spec_info stands for any field the output relay rebinds."""

    def __init__(self, reqs, extend=False, chunked_req=None, spec_info=None):
        self.reqs = list(reqs)
        self.forward_mode = SimpleNamespace(
            is_extend=lambda: extend, is_mixed=lambda: False
        )
        self.has_prefill_rows = lambda: extend
        self.chunked_req = chunked_req
        self.spec_info = spec_info
        self.sampling_info = SimpleNamespace(penalizer_orchestrator=SimpleNamespace())

    def is_empty(self):
        return not self.reqs

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


def _rids(batch):
    return [r.rid for r in batch.reqs]


def _scheduler(num_slots):
    s = MagicMock()
    s.enable_hisparse = False
    s.dllm_config = None
    s.decoding_batches = [DecodingBatch(_Batch([])) for _ in range(num_slots)]
    s.leave_finished_rows = MethodType(Scheduler.leave_finished_rows, s)
    s.update_rows_at_commit = MethodType(SchedulerPPMixin._pp_update_rows_at_commit, s)
    return s


class TestPPSlotRowsAtCommit(unittest.TestCase):
    def test_rows_join_the_step_as_its_commit_left_it(self):
        s = _scheduler(1)
        step = _Batch([_Req("a"), _Req("b")], extend=True, spec_info="launched")
        # The output relay of the commit rebinds the step's fields.
        step.spec_info = "relayed"

        s.update_rows_at_commit(0, step)

        rows = s.decoding_batches[0].rows
        self.assertEqual(_rids(rows), ["a", "b"])
        self.assertEqual(rows.spec_info, "relayed")
        self.assertIsNot(rows, step)

    def test_rows_the_commit_finished_leave(self):
        s = _scheduler(1)
        a, b = _Req("a"), _Req("b")
        s.decoding_batches[0] = DecodingBatch(_Batch([a, b]))
        step = s.decoding_batches[0].decode_step()
        step.spec_info = "relayed"
        b.done = True

        s.update_rows_at_commit(0, step)

        rows = s.decoding_batches[0].rows
        self.assertEqual(_rids(rows), ["a"])
        # The rows took back what the relay rebound on their decode step.
        self.assertEqual(rows.spec_info, "relayed")

    def test_chunked_request_joins_the_slot_of_its_last_chunk(self):
        s = _scheduler(2)
        c, x = _Req("c"), _Req("x")
        # Slot 0 ran a middle chunk of c, slot 1 its last chunk.
        s.update_rows_at_commit(0, _Batch([c, x], extend=True, chunked_req=c))
        self.assertEqual(_rids(s.decoding_batches[0].rows), ["x"])

        s.update_rows_at_commit(1, _Batch([c], extend=True))

        self.assertEqual(_rids(s.decoding_batches[0].rows), ["x"])
        self.assertEqual(_rids(s.decoding_batches[1].rows), ["c"])


class _EndLoop(Exception):
    pass


class TestPPLoopRowsAtCommit(unittest.TestCase):
    def test_a_prefilled_request_joins_its_slot_when_the_ring_commits_it(self):
        # pp_size 2, no async depth: two slots, and slot 0's step commits in
        # ring step 1, just before slot 0 replans in ring step 2.
        a = _Req("a")
        step = _Batch([a], extend=True, spec_info="launched")
        schedule = iter([step, None])
        rows_at_replan = []

        s = Scheduler.__new__(Scheduler)
        s.enable_overlap = False
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.device = "cpu"
        s.enable_hisparse = False
        s.dllm_config = None
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s._sched_idled = False
        s.on_idle = Mock()
        s._pp_recv_proxy_tensors = Mock()
        s._pp_commit_proxy_send_work = Mock()
        s._pp_launch_batch = Mock(return_value=(GenerationBatchResult(), Mock()))
        s._pp_commit_send_output_work_and_preprocess_output_tensors = Mock(
            return_value=(None, GenerationBatchResult(), Mock())
        )

        def relay_and_commit(batch, result):
            # The output relay rebinds a field of the step at its commit.
            batch.spec_info = "relayed"

        s._pp_process_batch_result = Mock(side_effect=relay_and_commit)

        def ingest():
            if s.inflight_steps.step == 2:
                rows_at_replan.append(s.decoding_batches[0].rows)
                raise _EndLoop
            return []

        s.ingest_requests = ingest
        s.get_next_batch_to_run = Mock(
            side_effect=lambda **kwargs: SimpleNamespace(batch_to_run=next(schedule))
        )

        with get_parallel().override(pp_size=2, pp_async_batch_depth=0):
            with self.assertRaises(_EndLoop):
                s.event_loop_pp()

        s._pp_process_batch_result.assert_called_once()
        (rows,) = rows_at_replan
        self.assertEqual(_rids(rows), ["a"])
        self.assertEqual(rows.spec_info, "relayed")
        self.assertEqual(_rids(s.decoding_batches[1].rows), [])


if __name__ == "__main__":
    unittest.main()
