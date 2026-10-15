import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
    UpdateRejected,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.request_registry import (
    RequestRegistry,
)
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.scheduler_request_lists import make_request_registry

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_registry(transfer) -> RequestRegistry:
    return make_request_registry(
        mode_owners=ModeOwners(
            disagg_prefill=prefill_owner_shell(transfer=list(transfer))
        )
    )


def _controller(registry: RequestRegistry) -> SchedulerInternalStateController:
    return SchedulerInternalStateController(
        workers=SimpleNamespace(tp_worker=None, draft_worker=None),
        token_to_kv_pool_allocator=None,
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        max_running_requests=0,
        max_total_num_tokens=0,
        swa_tokens_per_layer=0,
        startup_available_gpu_memory_gb=0.0,
        metrics_sink=None,
        rust_server=None,
        request_registry=registry,
        startup_time=None,
        record_step_time=False,
    )


class TestAcceptThresholdSamplingMaskGate(CustomTestCase):
    def test_pd_prefill_transfer_request_blocks_a_relaxed_threshold(self):
        """A sampling-mask request still sending its KV describes the exact
        target policy; relaxing acceptance under it must be rejected."""
        sending = SimpleNamespace(return_sampling_mask=True, finished=lambda: False)
        controller = _controller(_prefill_registry([sending]))
        with self.assertRaises(UpdateRejected):
            controller._validate_accept_threshold("accept_threshold", 0.5)

        controller = _controller(_prefill_registry([]))
        controller._validate_accept_threshold("accept_threshold", 0.5)


if __name__ == "__main__":
    unittest.main()
