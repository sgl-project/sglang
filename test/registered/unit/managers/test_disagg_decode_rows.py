"""PD decode joins the prebuilt rows into the running rows at the plan,
through a copy, and the normal and overlap loops decode a copy of the rows:
neither the prebuilt batch nor a launched step ever becomes the rows."""

import copy
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import (  # noqa: E402
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import request_queries  # noqa: E402
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

    def __init__(self, reqs, mode=ForwardMode.DECODE):
        self.reqs = list(reqs)
        self.forward_mode = mode
        self.num_decode_rows = 0
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

    copy = shallow_copy

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE


def _rids(reqs):
    return [r.rid for r in reqs]


def _plan_scheduler(prebuilt_reqs):
    """A scheduler whose PD decode plan finds prebuilt_reqs ready to join and
    selects every running row for decode."""
    s = MagicMock()
    s.chunked_req = None
    s.enable_hisparse = False
    s.last_batch = None
    s.disagg_decode_launches_copy = False
    s.get_next_disagg_decode_batch_to_run = MethodType(
        SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run, s
    )
    s.prebuilt = None

    def get_new_prebuilt_batch(running_batch):
        if not prebuilt_reqs:
            return None
        s.prebuilt = _Batch(prebuilt_reqs, ForwardMode.PREBUILT)
        return s.prebuilt

    s.get_new_prebuilt_batch.side_effect = get_new_prebuilt_batch
    s.update_running_batch.side_effect = lambda decoding_batch: decoding_batch.rows
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b: b
    return s


class TestPrebuiltJoin(CustomTestCase):
    def _plan(self, s, decoding_batch):
        with patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"):
            return s.get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch)

    def test_join_into_empty_rows_takes_a_copy(self):
        s = _plan_scheduler([_Req("p")])
        decoding_batch = DecodingBatch(_Batch([]))

        self._plan(s, decoding_batch)

        rows = decoding_batch.rows
        self.assertIsNot(rows, s.prebuilt)
        self.assertEqual(_rids(rows.reqs), ["p"])
        # The penalizers read their requests from the rows.
        self.assertIs(rows.sampling_info.penalizer_orchestrator.batch, rows)

    def test_join_into_running_rows_appends(self):
        s = _plan_scheduler([_Req("p")])
        decoding_batch = DecodingBatch(_Batch([_Req("a"), _Req("b")]))
        rows = decoding_batch.rows

        self._plan(s, decoding_batch)

        self.assertIs(decoding_batch.rows, rows)
        self.assertEqual(_rids(rows.reqs), ["a", "b", "p"])


class _StopLoop(Exception):
    pass


class _Scheduler(MagicMock):
    # The request queries read the rows through this alias.
    @property
    def running_batch(self):
        return self.decoding_batch.rows


class _Loop:
    """Drives a real PD decode loop and its real plan over scripted steps.

    Each step is a dict with optional keys: "prebuilt" (requests ready to
    join at its plan), "retract" (rids its decode selection retracts) and
    "finish" (rids its commit finishes). Records what intake, preallocation
    and on_idle read.
    """

    def __init__(self, test, steps, rows=()):
        self.test = test
        s = _Scheduler()
        self.s = s
        self.launched, self.intake, self.prealloc, self.idle = [], [], [], []
        s.gracefully_exit = False
        s._engine_paused = False
        s.enable_hisparse = False
        s.chunked_req = None
        s.last_batch = None
        s.waiting_queue = []
        s.disaggregation_mode = DisaggregationMode.DECODE
        s.decode_offload_manager = None
        s.enable_hierarchical_cache = False
        s.enable_lmcache = False
        s.dllm_manager.any_staging_reqs.return_value = False
        s.grammar_manager.grammar_queue = []
        s.decoding_batch = DecodingBatch(_Batch(rows))
        s.is_disable_overlap_for_batch.return_value = False
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b: b
        for name in (
            "get_next_disagg_decode_batch_to_run",
            "_commit_due_step",
            "_debug_check_commit_gate",
            "_debug_check_last_launched",
            "leave_finished_rows",
        ):
            setattr(s, name, MethodType(getattr(Scheduler, name), s))
        steps = iter(steps)
        finishes = {}

        def ingest_requests():
            rows = s.decoding_batch.rows
            self.intake.append((_rids(rows.reqs), getattr(rows, "launch", None)))
            if len(self.intake) > self.num_steps:
                raise _StopLoop

        def process_decode_queue():
            self.prealloc.append(_rids(s.decoding_batch.rows.reqs))

        def get_new_prebuilt_batch(running_batch):
            s.step = next(steps)
            reqs = s.step.get("prebuilt", [])
            return _Batch(reqs, ForwardMode.PREBUILT) if reqs else None

        def update_running_batch(decoding_batch):
            decoding_batch.leave([])
            retracted = [
                r
                for r in decoding_batch.rows.reqs
                if r.rid in s.step.get("retract", ())
            ]
            decoding_batch.leave(retracted)
            s.waiting_queue.extend(retracted)
            decoding_batch.advance()
            return decoding_batch.rows

        def run_batch(batch):
            self.launched.append(batch)
            # A run rebinds fields of the step it launches.
            batch.launch = len(self.launched)
            finishes[len(self.launched)] = s.step.get("finish", ())
            return SimpleNamespace(launch=batch.launch)

        def process_batch_result(batch, result):
            for r in batch.reqs:
                if r.rid in finishes[result.launch]:
                    r.done = True

        def on_idle():
            self.idle.append(
                (
                    request_queries.is_idle(s),
                    sorted(r.rid for r in request_queries.inflight_reqs(s)),
                )
            )

        s.ingest_requests.side_effect = ingest_requests
        s.process_decode_queue.side_effect = process_decode_queue
        s.get_new_prebuilt_batch.side_effect = get_new_prebuilt_batch
        s.update_running_batch.side_effect = update_running_batch
        s.run_batch.side_effect = run_batch
        s.process_batch_result.side_effect = process_batch_result
        s.on_idle.side_effect = on_idle

    def run(self, event_loop, num_steps):
        self.num_steps = num_steps
        with (
            patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"),
            self.test.assertRaises(_StopLoop),
        ):
            event_loop(self.s)


_LOOPS = (
    SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode,
    SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode,
)


class TestDecodeStepCopy(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))
        enter_scope(self, envs.SGLANG_DEBUG_INFLIGHT_STEPS.override(True))

    def test_rows_take_back_the_run_of_their_step_at_launch(self):
        for event_loop in _LOOPS:
            with self.subTest(event_loop=event_loop.__name__):
                loop = _Loop(self, [{}, {"prebuilt": [_Req("p")]}, {}])
                loop.run(event_loop, num_steps=3)
                rows = loop.s.decoding_batch.rows
                self.assertFalse(any(step is rows for step in loop.launched))
                self.assertEqual(
                    loop.intake,
                    [([], None), ([], None), (["p"], 1), (["p"], 2)],
                )

    def test_on_idle_after_the_last_row_finishes(self):
        # The step that decoded the row is a copy of the rows and stays
        # last_batch; it drops the row at the plan that finds nothing to run.
        loop = _Loop(self, [{"finish": {"a"}}, {}], rows=[_Req("a")])
        loop.run(
            SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode,
            num_steps=2,
        )
        self.assertEqual(loop.idle, [(True, [])])

    def test_on_idle_after_a_retract(self):
        # The retracted requests left the rows for the waiting queue: the
        # last step no longer lists them as running.
        loop = _Loop(self, [{}, {"retract": {"a", "b"}}], rows=[_Req("a"), _Req("b")])
        loop.run(
            SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode,
            num_steps=2,
        )
        self.assertEqual(loop.idle, [(False, [])])


if __name__ == "__main__":
    unittest.main()
