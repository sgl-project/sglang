import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(*, sampling_mask=False, finished=False):
    return SimpleNamespace(
        rid="r",
        return_sampling_mask=sampling_mask,
        finished=lambda: finished,
        kv=SimpleNamespace(holds_kv=False),
    )


def _scheduler(lists):
    s = Scheduler.__new__(Scheduler)
    s.request_lists = lambda: lists
    s.waiting_queue = lists.get("waiting", [])
    s.running_batch = SimpleNamespace(is_empty=lambda: True)
    s.last_batch = None
    s.enable_overlap = False
    s._pp_microbatches_drained = lambda: True
    s.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
    s._engine_paused = False
    s.request_receiver = object.__new__(SchedulerRequestReceiver)
    s.uocr_runtime = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_hierarchical_cache = False
    s.enable_lmcache = False
    return s


# Idle with one request in the list: (default, health check, health check while
# paused, ignore_waiting).
_IDLE_WITH_ONE_REQUEST = {
    "waiting": (False, False, False, True),
    "chunked": (False, False, False, False),
    "decode_retracted": (False, False, True, False),
    "prefill_transfer": (False, True, True, False),
    "a_new_list": (False, True, True, False),
    "running": (True, True, True, True),
    "prefill_launched": (True, True, True, True),
    "dllm": (True, True, True, True),
    "decode_held": (True, True, True, True),
    "uocr_parked": (True, True, True, True),
    "deferred": (True, True, True, True),
}


class TestIsFullyIdle(CustomTestCase):
    def test_each_list_counts_as_its_variant_defines(self):
        for name, expected in _IDLE_WITH_ONE_REQUEST.items():
            with self.subTest(name=name):
                s = _scheduler({name: [_req()]})
                default = s.is_fully_idle()
                health = s.is_fully_idle(for_health_check=True)
                s._engine_paused = True
                paused_health = s.is_fully_idle(for_health_check=True)
                s._engine_paused = False
                ignore_waiting = s.is_fully_idle(ignore_waiting=True)
                self.assertEqual(
                    (default, health, paused_health, ignore_waiting), expected
                )


class TestActiveSamplingMaskRequest(CustomTestCase):
    def test_an_unfinished_masked_request_in_any_list_is_active(self):
        for req, active in (
            (_req(sampling_mask=True), True),
            (_req(sampling_mask=True, finished=True), False),
            (_req(), False),
        ):
            s = _scheduler({"prefill_transfer": [req]})
            controller = SchedulerInternalStateController(s, record_step_time=False)
            self.assertEqual(controller._has_active_sampling_mask_request(), active)


if __name__ == "__main__":
    unittest.main()
