import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.dllm.config import DllmConfig  # noqa: E402
from sglang.srt.dllm.mixin.scheduler import (  # noqa: E402
    DllmManager,
    SchedulerDllmMixin,
)


def _make_scheduler(
    *, max_running_requests: int, pool_size: int, num_incoming: int
) -> SimpleNamespace:
    config = DllmConfig(
        algorithm="test",
        algorithm_config={},
        block_size=4,
        mask_id=0,
        max_running_requests=max_running_requests,
    )
    return SimpleNamespace(
        dllm_config=config,
        dllm_manager=DllmManager(dllm_config=config),
        req_to_token_pool=SimpleNamespace(size=pool_size),
        waiting_queue=[SimpleNamespace(rid=f"req-{i}") for i in range(num_incoming)],
    )


class TestFetchWaitingReqsClamp(CustomTestCase):
    def test_in_flight_dllm_requests_are_clamped_to_the_row_pool_size(self):
        """A configured max above the effective row pool would let row holders starve an unadmittable prefill candidate forever (FDFO livelock)."""
        scheduler = _make_scheduler(
            max_running_requests=100, pool_size=4, num_incoming=10
        )

        SchedulerDllmMixin._fetch_waiting_reqs(scheduler)

        self.assertEqual(len(scheduler.dllm_manager.waiting_queue), 4)
        self.assertEqual(len(scheduler.waiting_queue), 6)

    def test_a_max_below_the_pool_size_still_applies(self):
        """The clamp must not widen a deliberately small max_running_requests."""
        scheduler = _make_scheduler(
            max_running_requests=2, pool_size=4, num_incoming=10
        )

        SchedulerDllmMixin._fetch_waiting_reqs(scheduler)

        self.assertEqual(len(scheduler.dllm_manager.waiting_queue), 2)
        self.assertEqual(len(scheduler.waiting_queue), 8)


if __name__ == "__main__":
    unittest.main()
