import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
    UpdateRejected,
)
from sglang.test.disagg_owner_shells import prefill_owner_shell

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_scheduler(transfer) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_dflash=lambda: True)
    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    scheduler.waiting_queue = []
    scheduler.grammar_manager = SimpleNamespace(grammar_queue=[], abort_requests=None)
    scheduler.chunked_req = None
    scheduler.decoding = []
    scheduler.disagg_prefill = prefill_owner_shell(transfer=list(transfer))
    scheduler.disagg_decode = None
    scheduler.dllm = None
    scheduler.hisparse_coordinator = None
    return scheduler


def _controller(scheduler: Scheduler) -> SchedulerInternalStateController:
    return SchedulerInternalStateController(
        tp_worker=None,
        draft_worker=None,
        token_to_kv_pool_allocator=None,
        spec_algorithm=scheduler.spec_algorithm,
        max_running_requests=0,
        max_total_num_tokens=0,
        swa_tokens_per_layer=0,
        startup_available_gpu_memory_gb=0.0,
        metrics_reporter=None,
        rust_server=None,
        get_disaggregation_mode=lambda: scheduler.disaggregation_mode,
        get_request_lists=scheduler.request_lists,
        get_startup_time=lambda: None,
        record_step_time=False,
    )


class TestAcceptThresholdSamplingMaskGate(CustomTestCase):
    def test_pd_prefill_transfer_request_blocks_a_relaxed_threshold(self):
        """A sampling-mask request still sending its KV describes the exact
        target policy; relaxing acceptance under it must be rejected."""
        sending = SimpleNamespace(return_sampling_mask=True, finished=lambda: False)
        controller = _controller(_prefill_scheduler([sending]))
        with self.assertRaises(UpdateRejected):
            controller._validate_accept_threshold("accept_threshold", 0.5)

        controller = _controller(_prefill_scheduler([]))
        controller._validate_accept_threshold("accept_threshold", 0.5)


if __name__ == "__main__":
    unittest.main()
