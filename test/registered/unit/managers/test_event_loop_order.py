"""Each CUDA event loop makes the same scheduler calls in the same order.

Drives every flat and pipeline loop a few iterations on a fake scheduler that
logs each call it receives (with its arguments, rendered when the call is made),
each write to a loop-owned attribute, each change to the in-flight queue, and
where each profiler label opens (``label{``) and closes (``}``).
The expected logs, one ``[scenario]`` block each in the sibling
event_loop_order_golden.txt, were recorded on the hand-written loops; a change
to a loop's call order fails here.

Re-recording is legitimate only for an intentional change to a loop's call
order, reviewed together with the diff of the data file: run this file with
SGLANG_TEST_REGEN_EVENT_LOOP_ORDER=1, which rewrites the scenarios that ran.
A change meant to keep the order (a refactor) never re-records; record a new
scenario on the code before such a change, then check it passes after.
"""

import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, NonCallableMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.inflight_queue import (  # noqa: E402
    InflightQueue,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_disagg, get_parallel  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Attribute writes the loops make between calls.
_WATCHED = frozenset(
    {
        "inflight_steps",
        "enable_skip_finishing_decode",
        "rows_join_before_plan",
        "enable_continuous_input_polling",
        "cur_batch_for_debug",
        "_sched_idled",
        "last_batch",
        "decoding_batch",
        "launch_event",
        "send_req_work",
        "pp_outputs",
    }
)
# Queue changes; its queries have no effect.
_QUEUE_CHANGES = ("push", "pop_due", "drain", "end_step")

_GOLDEN_PATH = Path(__file__).with_name("event_loop_order_golden.txt")
_GOLDEN_HEADER = (
    "# Expected call order of each scenario in test_event_loop_order.py, recorded\n"
    "# on the hand-written loops. Read that test's docstring before re-recording.\n"
)


def _read_golden():
    golden = {}
    for line in _GOLDEN_PATH.read_text().splitlines():
        if line.startswith("[") and line.endswith("]"):
            lines = golden[line[1:-1]] = []
        elif line and not line.startswith("#"):
            lines.append(line)
    return golden


def _write_golden(golden):
    blocks = [
        f"[{name}]\n" + "".join(line + "\n" for line in lines)
        for name, lines in golden.items()
    ]
    _GOLDEN_PATH.write_text(_GOLDEN_HEADER + "\n" + "\n".join(blocks))


GOLDEN = _read_golden()


_PREFETCH = "disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries("
# Entries that open an intake line; the ingest_requests call right after one
# stays on its line.
_INTAKE_OPENERS = (_PREFETCH, "recv_requests{")


class _Stop(Exception):
    pass


class _B:
    """A labelled stand-in for a batch or any other value the log names."""

    def __init__(self, label, mode=ForwardMode.EXTEND):
        self.label = label
        self.forward_mode = mode
        self.is_extend_in_batch = mode.is_extend()

    def copy(self):
        return _B(self.label + "c", self.forward_mode)


def _render(value):
    if isinstance(value, _B):
        return value.label
    if isinstance(value, InflightQueue):
        return f"IQ({value.depth},{value.blocking},{value.ring})"
    if isinstance(value, NonCallableMock):
        name = value._extract_mock_name()
        return name[2:] if name.startswith("s.") else name
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, SimpleNamespace):
        return "md"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render(v) for v in value) + "]"
    return repr(value)


def _render_call(name, args, kwargs):
    parts = [_render(a) for a in args]
    parts += [f"{k}={_render(v)}" for k, v in kwargs.items()]
    return f"{name}({','.join(parts)})"


class _LoggedMock(MagicMock):
    def _get_child_mock(self, **kwargs):
        return _LoggedMock(**kwargs)

    def __call__(self, *args, **kwargs):
        name = self._extract_mock_name()
        self._log_sink.append(_render_call(name[2:], args, kwargs))
        return super().__call__(*args, **kwargs)


class _FakeScheduler(_LoggedMock):
    def __setattr__(self, name, value):
        if name in _WATCHED:
            self._log_sink.append(f"{name}={_render(value)}")
        super().__setattr__(name, value)


def _make_scheduler(log):
    _LoggedMock._log_sink = log
    s = _FakeScheduler(name="s")
    s.gracefully_exit = False
    s._engine_paused = False
    s.is_generation = True
    s.enable_staging = True
    s.enable_unified_memory = True
    s._deferred_input_requests = []
    s.chunked_req = None
    s.last_batch = None
    s.ngram_embedding_manager.prepare_for_forward.side_effect = (
        lambda batch, chunked_req: batch
    )
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda batch: batch
    s.run_batch.side_effect = lambda batch: _B("r" + batch.label)
    s._commit_due_step.side_effect = MethodType(Scheduler._commit_due_step, s)
    del log[:]
    return s


def _script_ingest(s, n, paused=(), exit_at=None, inputs=None):
    """ingest_requests pauses the engine on the listed calls, asks the loop to
    exit after call exit_at, and stops the run after n calls."""
    calls = iter(range(n + 1))

    def ingest(*args, **kwargs):
        i = next(calls)
        if i == n:
            raise _Stop
        s._engine_paused = i in paused
        if exit_at is not None and i >= exit_at:
            s.gracefully_exit = True
        return (inputs or {}).get(i, [])

    s.ingest_requests.side_effect = ingest


def _plans(batches):
    plans = iter(batches)

    def plan(*args, **kwargs):
        return SimpleNamespace(batch_to_run=next(plans, None))

    return plan


def _values(values, default):
    it = iter(values)
    return lambda *args, **kwargs: next(it, default)


def _lines(log):
    """One line per loop iteration or pipeline slot: split at each intake, or
    at the prefetch or profiler label before it, and at each slot's row
    binding."""
    lines = []
    for entry in log:
        opened = bool(lines) and lines[-1][-1].startswith(_INTAKE_OPENERS)
        starts = entry.startswith(_INTAKE_OPENERS) or entry.startswith(
            "decoding_batch="
        )
        if not lines or starts or (entry.startswith("ingest_requests(") and not opened):
            lines.append([])
        lines[-1].append(entry)
    return ["; ".join(line) for line in lines]


class _QueueLog:
    def __init__(self, log):
        self.log = log
        self.patches = []
        for name in _QUEUE_CHANGES:
            orig = getattr(InflightQueue, name)

            def logged(q, *args, _name=name, _orig=orig, **kwargs):
                self.log.append(_render_call("iq." + _name, args, kwargs))
                return _orig(q, *args, **kwargs)

            self.patches.append(patch.object(InflightQueue, name, logged))

    def __enter__(self):
        for p in self.patches:
            p.__enter__()

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.__exit__(*exc)


class _ProfilerLabel:
    """Stands in for torch.profiler.record_function and logs the label span."""

    def __init__(self, log, label):
        self.log = log
        self.label = label

    def __enter__(self):
        self.log.append(self.label + "{")

    def __exit__(self, *exc):
        self.log.append("}")


class _LoopOrderCase(CustomTestCase):
    maxDiff = None

    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))
        enter_scope(self, envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1))
        self.log = []
        enter_scope(self, _QueueLog(self.log))
        enter_scope(
            self,
            patch(
                "torch.profiler.record_function",
                lambda label: _ProfilerLabel(self.log, label),
            ),
        )

    def run_loop(self, loop, s):
        try:
            loop(s)
        except _Stop:
            pass
        return _lines(self.log)

    def check(self, name, lines):
        if envs.SGLANG_TEST_REGEN_EVENT_LOOP_ORDER.get():
            GOLDEN[name] = lines
            _write_golden(GOLDEN)
            return
        self.assertEqual(lines, GOLDEN[name], name)


A, B, C = (_B(label) for label in "ABC")


class TestFlatLoopOrder(_LoopOrderCase):
    def flat_scheduler(self, batches, disable=(), paused=(2,), n=8, exit_at=None):
        s = _make_scheduler(self.log)
        _script_ingest(s, n, paused=paused, exit_at=exit_at)
        s.get_next_batch_to_run.side_effect = _plans(batches)
        s.get_next_disagg_prefill_batch_to_run.side_effect = _plans(batches)
        s.get_next_disagg_decode_batch_to_run.side_effect = _plans(batches)
        s.is_disable_overlap_for_batch.side_effect = _values(disable, False)
        s.disagg_prefill_bootstrap_queue.pop_bootstrapped.return_value = []
        return s

    def test_flat_loops(self):
        schedule = [A, B, None, None, C, None]
        for name, loop in (
            ("normal", Scheduler.event_loop_normal),
            ("overlap", Scheduler.event_loop_overlap),
            ("pd_prefill_normal", Scheduler.event_loop_normal_disagg_prefill),
            ("pd_decode_normal", Scheduler.event_loop_normal_disagg_decode),
            ("pd_decode_overlap", Scheduler.event_loop_overlap_disagg_decode),
        ):
            with self.subTest(name):
                s = self.flat_scheduler(schedule, disable=[False, True])
                self.check(name, self.run_loop(loop, s))

    def test_overlap_exit_and_non_generation(self):
        s = self.flat_scheduler([A, B, C], exit_at=2)
        s.is_generation = False
        s.enable_unified_memory = False
        s.is_disable_overlap_for_batch.side_effect = _values([False, True], False)
        self.check("overlap_exit", self.run_loop(Scheduler.event_loop_overlap, s))

    def test_pd_overlap_loops_sample_for_non_generation(self):
        for name, loop in (
            (
                "pd_prefill_overlap_non_generation",
                Scheduler.event_loop_overlap_disagg_prefill,
            ),
            (
                "pd_decode_overlap_non_generation",
                Scheduler.event_loop_overlap_disagg_decode,
            ),
        ):
            with self.subTest(name):
                s = self.flat_scheduler([A, B], paused=(), n=3)
                s.is_generation = False
                s._is_continuous_input_polling_enabled.return_value = False
                self.check(name, self.run_loop(loop, s))

    def test_pd_prefill_without_staging(self):
        for name, loop in (
            (
                "pd_prefill_normal_no_staging",
                Scheduler.event_loop_normal_disagg_prefill,
            ),
            (
                "pd_prefill_overlap_no_staging",
                Scheduler.event_loop_overlap_disagg_prefill,
            ),
        ):
            with self.subTest(name):
                s = self.flat_scheduler([A, B], paused=(), n=3)
                s.enable_staging = False
                s._is_continuous_input_polling_enabled.return_value = False
                self.check(name, self.run_loop(loop, s))

    def test_pd_prefill_overlap_blocking(self):
        s = self.flat_scheduler([A, B, None, None, C, None])
        s._is_continuous_input_polling_enabled.return_value = False
        self.check(
            "pd_prefill_overlap",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )

    def test_pd_prefill_overlap_polling(self):
        # A and B launch; A's copy is pending for two passes, during which new
        # input clears the skip memo and the queue is full; B is ready at
        # once; then a failed plan sets the skip memo, a deferred request and
        # a bootstrap arrival gate the next plans.
        s = self.flat_scheduler([A, B, None, None, C, None], paused=(), n=11)
        s._is_continuous_input_polling_enabled.return_value = True
        s.is_disagg_prefill_batch_result_ready.side_effect = _values(
            [False, False, True, True, False, True], True
        )
        _script_ingest(s, 11, inputs={2: ["req"]})
        bootstrapped = {6: ["b"]}
        calls = iter(range(100))
        s.disagg_prefill_bootstrap_queue.pop_bootstrapped.side_effect = lambda: (
            bootstrapped.get(next(calls), [])
        )
        deferred = iter([[], [], [], [], [], ["d"]])
        orig_ingest = s.ingest_requests.side_effect

        def ingest(*args, **kwargs):
            out = orig_ingest(*args, **kwargs)
            s._deferred_input_requests = next(deferred, [])
            return out

        s.ingest_requests.side_effect = ingest
        self.check(
            "pd_prefill_polling",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )

    def test_pd_prefill_polling_shutdown_while_waiting(self):
        s = self.flat_scheduler([A, B, C], paused=(), n=6, exit_at=2)
        s._is_continuous_input_polling_enabled.return_value = True
        s.is_disagg_prefill_batch_result_ready.side_effect = _values([False], True)
        self.check(
            "pd_prefill_polling_exit",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )

    def test_pd_prefill_polling_pause_while_waiting(self):
        s = self.flat_scheduler([A, B, C], paused=(2,), n=6)
        s._is_continuous_input_polling_enabled.return_value = True
        s.is_disagg_prefill_batch_result_ready.side_effect = _values([False], True)
        self.check(
            "pd_prefill_polling_pause",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )


def _pp_init(s, loop_size, spec_relay=False, metadata=None):
    def init():
        s.pp_loop_size = loop_size
        s.inflight_steps = InflightQueue(depth=loop_size - 1, ring=True)
        s.decoding_batches = [_B(f"db{i}") for i in range(loop_size)]
        s.mb_metadata = list(metadata or [None] * loop_size)
        s.pp_outputs = None
        s.last_rank_comm_queue = _B("lrcq")
        s._pp_spec_relay = spec_relay
        s.send_req_work = []

    s.init_pp_loop_state.side_effect = init


class TestRingLoopOrder(_LoopOrderCase):
    def ring_scheduler(self, batches, loop_size, last_rank, rounds=3, **init):
        s = _make_scheduler(self.log)
        _pp_init(s, loop_size, **init)
        _script_ingest(s, rounds * loop_size)
        s.pp_group = SimpleNamespace(is_last_rank=last_rank)
        s.waiting_queue = []
        s.disagg_prefill_inflight_queue = []
        s.disagg_decode_transfer_queue.queue = []
        s.disagg_decode_prealloc_queue.queue = []
        s.decode_offload_manager.ongoing_offload = []
        s.get_next_batch_to_run.side_effect = _plans(batches)
        s.get_next_disagg_decode_batch_to_run.side_effect = _plans(batches)
        s.get_new_batch_prefill.side_effect = _plans(batches)
        for name in (
            "_pp_slot_last_batch",
            "_pp_pop_due_batch",
            "_pp_debug_check_slot",
        ):
            getattr(s, name).side_effect = MethodType(getattr(Scheduler, name), s)
        s._pp_recv_proxy_tensors.return_value = _B("proxy")
        s._pp_launch_batch.side_effect = lambda mb_id, batch, *rest: (
            _B("res" + batch.label),
            _B("ev" + batch.label),
        )
        s._pp_commit_send_output_work_and_preprocess_output_tensors.side_effect = (
            lambda first, nxt: (_B(f"out{nxt}"), _B(f"nres{nxt}"), s.d2h_event)
        )
        s._pp_send_pyobj_to_next_stage.side_effect = lambda data, async_send: _B(
            "work:" + _render(data)
        )
        s._pp_recv_pyobj_from_prev_stage.return_value = _B("recv")
        s._pp_pd_send_consensus_release_ids.side_effect = lambda tmbs, first, rel, ids: (
            _B("relwork"),
            _B("rel:" + _render(ids)),
        )
        s._pp_pd_send_consensus_bootstrapped_ids.side_effect = (
            lambda mbs, first, cons, ids: (_B("conswork"), _B("cons:" + _render(ids)))
        )
        s.process_retract_queue.side_effect = lambda ids: _B("pr:" + _render(ids))
        s.process_prealloc_queue.side_effect = lambda ids: _B("pp:" + _render(ids))
        s.process_decode_transfer_queue.side_effect = lambda ids: _B(
            "pt:" + _render(ids)
        )
        return s

    def run_ring(self, name, loop, s, pp_size, depth, offload=False):
        with get_parallel().override(pp_size=pp_size, pp_async_batch_depth=depth):
            with get_disagg().override(
                disaggregation_decode_enable_offload_kvcache=offload
            ):
                self.check(name, self.run_loop(loop, s))

    def test_pp(self):
        d = ForwardMode.DECODE
        for name, kwargs, last_rank, depth in (
            ("pp_l2", {}, False, 0),
            ("pp_l3_last", {}, True, 1),
            (
                "pp_l2_spec_relay",
                dict(
                    spec_relay=True,
                    metadata=[None, SimpleNamespace(fwd_batch=_B("F1"))],
                ),
                False,
                0,
            ),
        ):
            with self.subTest(name):
                batches = [A, _B("D", d), None, _B("E", d), None, None]
                s = self.ring_scheduler(batches, 2 + depth, last_rank, **kwargs)
                self.run_ring(name, Scheduler.event_loop_pp, s, 2, depth)

    def test_pp_disagg_prefill(self):
        for name, last_rank, depth in (("pdp_l2", False, 0), ("pdp_l3_last", True, 1)):
            with self.subTest(name):
                s = self.ring_scheduler(
                    [A, B, None, C, None, None], 2 + depth, last_rank
                )
                transferred = iter(
                    [None, ["t1"], None, ["t3"], None, None, None, None, None]
                )
                s._pp_pd_get_prefill_transferred_ids.side_effect = lambda: next(
                    transferred
                )
                if name == "pdp_l2":
                    s.disagg_prefill_inflight_queue = [object()]
                self.run_ring(name, Scheduler.event_loop_pp_disagg_prefill, s, 2, depth)

    def test_pp_disagg_decode(self):
        d = ForwardMode.DECODE
        for name, last_rank, depth, offload in (
            ("pdd_l2", False, 0, True),
            ("pdd_l3_last", True, 1, False),
        ):
            with self.subTest(name):
                batches = [_B("A", d), _B("B", d), None, _B("C", d), None, None]
                s = self.ring_scheduler(batches, 2 + depth, last_rank)
                ids = {
                    "_pp_pd_get_retract_ids": [["r0"], None, None, ["r3"]],
                    "_pp_pd_get_prealloc_ids": [None, ["p1"], None],
                    "_pp_pd_get_decode_transferred_ids": [None, None, ["t2"]],
                }
                for method, values in ids.items():
                    getattr(s, method).side_effect = _values(values, None)
                self.run_ring(
                    name, Scheduler.event_loop_pp_disagg_decode, s, 2, depth, offload
                )

    def test_pp_disagg_decode_idle_with_queued_requests(self):
        # A queued or offloading request keeps the idle rounds from on_idle.
        for name, offload in (("pdd_queued", False), ("pdd_offloading", True)):
            with self.subTest(name):
                s = self.ring_scheduler([_B("A", ForwardMode.DECODE)], 2, False)
                for method in (
                    "_pp_pd_get_retract_ids",
                    "_pp_pd_get_prealloc_ids",
                    "_pp_pd_get_decode_transferred_ids",
                ):
                    getattr(s, method).return_value = None
                if offload:
                    s.decode_offload_manager.ongoing_offload = [object()]
                else:
                    s.disagg_decode_transfer_queue.queue = [object()]
                self.run_ring(
                    name, Scheduler.event_loop_pp_disagg_decode, s, 2, 0, offload
                )


if __name__ == "__main__":
    unittest.main()
