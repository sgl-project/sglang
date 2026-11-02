"""Readers that ask whether any live request is somewhere read every list of
Scheduler.request_lists()."""

import unittest
from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.scheduler_components.request_list import RequestList
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Req(SimpleNamespace):
    """A request stand-in compared and hashed by identity, like Req."""

    __eq__ = object.__eq__
    __hash__ = object.__hash__


def _req(rid="r", *, sampling_mask=False, finished=False, holds_kv=False):
    return _Req(
        rid=rid,
        return_sampling_mask=sampling_mask,
        finished=lambda: finished,
        kv=SimpleNamespace(holds_kv=holds_kv),
    )


def _idle_scheduler():
    s = Scheduler.__new__(Scheduler)
    s.disaggregation_mode = DisaggregationMode.NULL
    # No batch, step, or non-request work.
    s.running_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
    s.last_batch = None
    s.enable_overlap = False
    s.result_queue = deque()
    s._pp_microbatches_drained = lambda: True
    s._engine_paused = False
    s.request_receiver = object.__new__(SchedulerRequestReceiver)
    s.decode_offload_manager = None
    s.enable_hierarchical_cache = False
    s.enable_lmcache = False
    # Every request list empty.
    s.waiting_queue = []
    s.grammar_manager = SimpleNamespace(grammar_queue=[])
    s.chunked_req = None
    s.prefill_launched_reqs = RequestList()
    s.running_reqs = []
    s.dllm_config = None
    s.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False, waiting_queue=[])
    s.disagg_prefill_bootstrap_queue = None
    s.disagg_prefill_transfer_queue = None
    s.disagg_decode_prealloc_queue = None
    s.disagg_decode_transfer_queue = None
    s.hisparse_coordinator = None
    s.uocr_runtime = None
    s.deferred_responses = DeferredResponses(
        disaggregation_mode=DisaggregationMode.NULL,
        output_streamer=None,
        send_abort=None,
    )
    return s


def _component(name, reqs, **attrs):
    return SimpleNamespace(request_lists=lambda: {name: reqs}, **attrs)


def _place(s, name, req):
    """Put req in the request list called name, as the scheduler would."""
    if name == "waiting":
        s.waiting_queue = [req]
    elif name == "grammar":
        s.grammar_manager.grammar_queue = [req]
    elif name == "chunked":
        s.chunked_req = req
    elif name == "prefill_launched":
        s.prefill_launched_reqs.join([req])
    elif name == "running":
        s.running_reqs = [req]
    elif name.startswith("prefill_"):
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=[req] if name == "prefill_bootstrap" else []
        )
        s.disagg_prefill_transfer_queue = [req] if name == "prefill_transfer" else []
    elif name == "dllm":
        # Queued for a later round, not staged for this one.
        s.dllm_config = object()
        s.dllm_manager.waiting_queue = [req]
    elif name.startswith("decode_"):
        s.disaggregation_mode = DisaggregationMode.DECODE
        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc.queue = [SimpleNamespace(req=req)] if name == "decode_prealloc" else []
        prealloc.pending_reqs = []
        prealloc.retracted_queue = [req] if name == "decode_retracted" else []
        prealloc.held_rebootstrap_reqs = [req] if name == "decode_held" else []
        transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer.queue = [SimpleNamespace(req=req)] if name == "decode_transfer" else []
        transfer._deferred_releases = (
            [(SimpleNamespace(req=req), 0.0, 0.0, 0, 0)]
            if name == "decode_releasing"
            else []
        )
        s.disagg_decode_prealloc_queue = prealloc
        s.disagg_decode_transfer_queue = transfer
    elif name == "hisparse_staging":
        s.hisparse_coordinator = _component(name, [req])
    elif name == "uocr_parked":
        # Parked windows block a flush like waiting requests do.
        s.uocr_runtime = _component(
            name, [req], blocks_flush=lambda ignore_waiting: not ignore_waiting
        )
    elif name == "deferred":
        s.deferred_responses = _component(name, [req])
    else:
        raise ValueError(name)
    assert s.request_lists()[name] == [req]


# Whether the server is idle with one request in the list: (default, health
# check, health check while paused, ignore_waiting).
_IDLE_WITH_ONE_REQUEST = {
    "waiting": (False, False, False, True),
    "grammar": (False, True, True, False),
    "chunked": (False, False, False, False),
    "prefill_bootstrap": (False, True, True, False),
    "prefill_transfer": (False, True, True, False),
    "dllm": (True, True, True, True),
    "decode_prealloc": (False, True, True, False),
    "decode_retracted": (False, False, True, False),
    "decode_held": (True, True, True, True),
    "decode_transfer": (False, True, True, False),
    "decode_releasing": (False, True, True, False),
    "hisparse_staging": (False, True, True, False),
    "uocr_parked": (False, True, True, True),
    "deferred": (True, True, True, True),
}


class TestIsFullyIdle(CustomTestCase):
    def _variants(self, name, req):
        s = _idle_scheduler()
        _place(s, name, req)
        default = s.is_fully_idle()
        health = s.is_fully_idle(for_health_check=True)
        s._engine_paused = True
        paused_health = s.is_fully_idle(for_health_check=True)
        s._engine_paused = False
        return default, health, paused_health, s.is_fully_idle(ignore_waiting=True)

    def test_empty_lists_are_idle(self):
        s = _idle_scheduler()
        self.assertTrue(s.is_fully_idle())
        self.assertTrue(s.is_fully_idle(for_health_check=True))
        self.assertTrue(s.is_fully_idle(ignore_waiting=True))

    def test_each_list_counts_as_its_variants_define(self):
        for name, expected in _IDLE_WITH_ONE_REQUEST.items():
            with self.subTest(name=name):
                self.assertEqual(self._variants(name, _req()), expected)

    def test_ignore_waiting_still_counts_a_waiting_request_holding_kv(self):
        self.assertEqual(
            self._variants("waiting", _req(holds_kv=True)),
            (False, False, False, False),
        )

    def test_a_new_component_list_counts_except_for_health_check(self):
        s = _idle_scheduler()
        s.hisparse_coordinator = _component("new_list", [_req()])
        self.assertFalse(s.is_fully_idle())
        self.assertFalse(s.is_fully_idle(ignore_waiting=True))
        self.assertTrue(s.is_fully_idle(for_health_check=True))


class TestActiveSamplingMaskRequest(CustomTestCase):
    def _active(self, s):
        controller = SchedulerInternalStateController(s, record_step_time=False)
        return controller._has_active_sampling_mask_request()

    def test_every_list_is_read(self):
        names = [*_IDLE_WITH_ONE_REQUEST, "prefill_launched", "running"]
        for name in names:
            with self.subTest(name=name):
                s = _idle_scheduler()
                _place(s, name, _req(sampling_mask=True))
                self.assertTrue(self._active(s))

    def test_finished_or_unmasked_requests_are_not_active(self):
        for req in (_req(sampling_mask=True, finished=True), _req()):
            s = _idle_scheduler()
            _place(s, "waiting", req)
            self.assertFalse(self._active(s))


if __name__ == "__main__":
    unittest.main()
