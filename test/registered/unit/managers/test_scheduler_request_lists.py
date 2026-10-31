"""The scheduler's request lists, and the idle check that reads them."""

import itertools
import unittest
from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.hisparse_coordinator import HiSparseAct, HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.running_reqs import RunningReqs
from sglang.srt.runtime_context import get_memory

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _is_fully_idle_before_request_lists(
    self, for_health_check=False, ignore_waiting=False
) -> bool:
    """`Scheduler.is_fully_idle` as it was before it read the request lists,
    with its comments dropped."""
    idle = (
        self.running_batch.is_empty()
        and self.chunked_req is None
        and not self.dllm_manager.any_staging_reqs()
        and (self.last_batch is None or self.last_batch.is_empty())
        and (not self.enable_overlap or len(self.result_queue) == 0)
        and self._pp_microbatches_drained()
    )

    if not ignore_waiting:
        idle &= len(self.waiting_queue) == 0
    else:
        idle &= not any(r.kv.holds_kv for r in self.waiting_queue)

    if (
        for_health_check
        and not self._engine_paused
        and self.disaggregation_mode == DisaggregationMode.DECODE
        and self.disagg_decode_prealloc_queue is not None
    ):
        idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0

    if not for_health_check:
        idle &= len(self.grammar_manager.grammar_queue) == 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            idle &= len(self.disagg_prefill_inflight_queue) == 0
            idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
            idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
            idle &= len(self.disagg_decode_transfer_queue.queue) == 0
            idle &= (
                not self.disagg_decode_transfer_queue.has_pending_deferred_releases()
            )
            if self.decode_offload_manager is not None:
                idle &= len(self.decode_offload_manager.ongoing_offload) == 0

        if self.enable_hisparse:
            idle &= not self.hisparse_coordinator.has_ongoing_staging()

        if self.enable_hierarchical_cache:
            tc = self.tree_cache
            idle &= len(tc.ongoing_write_through) == 0
            idle &= len(tc.ongoing_load_back) == 0
            if tc.enable_storage:
                idle &= len(tc.ongoing_prefetch) == 0
                idle &= len(tc.ongoing_backup) == 0
                if get_memory().hicache_host_memory_mode == "buffer_only":
                    idle &= tc.buffer_pipeline.is_idle()
        elif self.enable_lmcache:
            idle &= not self.tree_cache.has_pending_cache_operations()

    return idle


class _Req:
    def __init__(self, rid, *, holds_kv=False, finished=False):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=holds_kv)
        self._finished = finished

    def finished(self):
        return self._finished


_CORE_HOLDERS = (
    "waiting",
    "waiting_holding_kv",
    "grammar",
    "running",
    # A decode step finished its requests: they are still in the running
    # batch, which is the step's batch, until the next schedule filters it.
    "finished_in_last_batch",
    "chunked",
    # The step holds a request no list does: one whose last prefill chunk
    # launched under PD prefill or HiSparse.
    "last_batch",
    "overlap",
    "result_queue",
    "pp_microbatches",
    "deferred",
)
_MODE_HOLDERS = {
    "null": (),
    "prefill": ("prefill_bootstrap", "prefill_transfer"),
    "decode": (
        "decode_prealloc",
        "decode_retracted",
        "decode_held",
        "decode_transfer",
        "decode_releasing",
        "decode_offload",
    ),
    "dllm": ("dllm_waiting", "dllm_staging"),
    "hisparse": ("hisparse_staging",),
}
# The step records and the copies in flight, which hold no listed request.
_NOT_LISTED = {
    "finished_in_last_batch",
    "last_batch",
    "overlap",
    "result_queue",
    "pp_microbatches",
    "decode_offload",
}
_DISAGGREGATION_MODES = {
    "prefill": DisaggregationMode.PREFILL,
    "decode": DisaggregationMode.DECODE,
}


def _scheduler(mode, holders, *, engine_paused=False):
    """A scheduler whose given holders each hold one request, all else empty."""
    req = {name: _Req(name) for name in holders}
    s = Scheduler.__new__(Scheduler)
    s.disaggregation_mode = _DISAGGREGATION_MODES.get(mode, DisaggregationMode.NULL)
    s._engine_paused = engine_paused
    s.enable_hierarchical_cache = False
    s.enable_lmcache = False

    s.waiting_queue = [req[name] for name in ("waiting",) if name in holders]
    if "waiting_holding_kv" in holders:
        s.waiting_queue.append(_Req("waiting_holding_kv", holds_kv=True))
    s.grammar_manager = SimpleNamespace(
        grammar_queue=[req[name] for name in ("grammar",) if name in holders]
    )
    s.running_reqs = RunningReqs()
    s.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    s.last_batch = None
    if "running" in holders:
        s.running_reqs.join([req["running"]])
        s.running_batch.reqs.append(req["running"])
    if "finished_in_last_batch" in holders:
        s.running_batch.reqs.append(_Req("finished", finished=True))
        s.last_batch = s.running_batch
    if "last_batch" in holders:
        if s.last_batch is None:
            s.last_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        s.last_batch.reqs.append(req["last_batch"])
    s.chunked_req = req.get("chunked")
    s.enable_overlap = "overlap" in holders
    s.result_queue = deque(
        [(ScheduleBatch(reqs=[req["result_queue"]]), None)]
        if "result_queue" in holders
        else []
    )
    s._pp_microbatches_drained = lambda: "pp_microbatches" not in holders
    s._deferred_responses = (
        {"deferred": (req["deferred"], None)} if "deferred" in holders else None
    )

    s.disagg_prefill_bootstrap_queue = None
    s.disagg_prefill_inflight_queue = None
    if mode == "prefill":
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=[req[name] for name in ("prefill_bootstrap",) if name in holders]
        )
        s.disagg_prefill_inflight_queue = [
            req[name] for name in ("prefill_transfer",) if name in holders
        ]

    s.disagg_decode_prealloc_queue = None
    s.disagg_decode_transfer_queue = None
    s.decode_offload_manager = None
    if mode == "decode":
        s.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[
                SimpleNamespace(req=req[name])
                for name in ("decode_prealloc",)
                if name in holders
            ],
            retracted_queue=[
                req[name] for name in ("decode_retracted",) if name in holders
            ],
            held_rebootstrap_reqs=[
                req[name] for name in ("decode_held",) if name in holders
            ],
        )
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = [
            SimpleNamespace(req=req[name])
            for name in ("decode_transfer",)
            if name in holders
        ]
        transfer_queue._deferred_releases = [
            (SimpleNamespace(req=req[name]), 0.0, 0.0, 0, 0)
            for name in ("decode_releasing",)
            if name in holders
        ]
        s.disagg_decode_transfer_queue = transfer_queue
        s.decode_offload_manager = SimpleNamespace(
            ongoing_offload={1: None} if "decode_offload" in holders else {}
        )

    s.dllm_config = None
    s.dllm_manager = DllmManager()
    if mode == "dllm":
        s.dllm_config = SimpleNamespace(max_running_requests=8)
        s.dllm_manager = DllmManager(dllm_config=s.dllm_config)
        for name in ("dllm_waiting", "dllm_staging"):
            if name in holders:
                s.dllm_manager.waiting_queue.append(req[name])
        if "dllm_staging" in holders:
            s.dllm_manager.staging_queue.append(req["dllm_staging"])

    s.enable_hisparse = mode == "hisparse"
    s.hisparse_coordinator = None
    if mode == "hisparse":
        s.hisparse_coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        s.hisparse_coordinator.ack_staging_queue = [
            HiSparseAct(None, None, req[name])
            for name in ("hisparse_staging",)
            if name in holders
        ]
    return s


class TestIsFullyIdleOnRequestLists(CustomTestCase):
    def test_matches_the_holder_walk_it_replaced(self):
        """Moving the idle check onto the request lists' flags keeps its value:
        a wrong flag would let a flush or a weight update run under live
        requests, or answer a health check while requests run."""
        mismatches = []
        for mode, mode_holders in _MODE_HOLDERS.items():
            holders = _CORE_HOLDERS + mode_holders
            states = [()] + [
                combo
                for size in (1, 2)
                for combo in itertools.combinations(holders, size)
            ]
            for (
                state,
                engine_paused,
                for_health_check,
                ignore_waiting,
            ) in itertools.product(states, *[(False, True)] * 3):
                s = _scheduler(mode, state, engine_paused=engine_paused)
                kwargs = dict(
                    for_health_check=for_health_check, ignore_waiting=ignore_waiting
                )
                expected = _is_fully_idle_before_request_lists(s, **kwargs)
                if s.is_fully_idle(**kwargs) != expected:
                    mismatches.append((mode, state, engine_paused, kwargs, expected))
        self.assertEqual(mismatches, [])


class TestRequestLists(CustomTestCase):
    def test_each_holder_is_in_exactly_one_list(self):
        """A holder left out of the lists hides its requests from every reader
        (a weight version change, the idle check); one in two lists counts
        them twice."""
        for mode, mode_holders in _MODE_HOLDERS.items():
            with self.subTest(mode=mode):
                holders = [
                    name
                    for name in _CORE_HOLDERS + mode_holders
                    if name not in _NOT_LISTED
                ]
                lists = _scheduler(mode, holders).request_lists()

                names = [entry.name for entry in lists]
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(
                    sorted(req.rid for entry in lists for req in entry.reqs),
                    sorted(holders),
                )


if __name__ == "__main__":
    unittest.main()
