from __future__ import annotations

import unittest
from unittest.mock import patch

from kv_canary_runner_unit_utils import (
    CanaryRunnerTestCase,
    make_config,
    make_forward_batch,
    make_runner,
)

from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import make_radix_cache, make_req_to_token_pool

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitRunnerSweep(CanaryRunnerTestCase):
    def test_sweep_every_n_cadence(self) -> None:
        """Verify sweep execution follows the configured step cadence."""
        config = make_config(sweep_interval=4)
        runner = make_runner(device=self.device, config=config)
        forward_batch = make_forward_batch(self.device)

        sweep_calls: list[int] = []
        real_maybe = runner._sweep_orchestrator.maybe_run_sweep

        def _spy() -> None:
            before = runner._sweep_orchestrator._last_sweep_step
            real_maybe()
            if runner._sweep_orchestrator._last_sweep_step != before:
                sweep_calls.append(runner._step_counter)

        with patch.object(runner._sweep_orchestrator, "maybe_run_sweep", _spy):
            for _ in range(12):
                with runner.with_forward_pass(forward_batch):
                    pass
        self.assertEqual(sweep_calls, [0, 4, 8])

    def test_sweep_path_launches_sweep_kernels(self) -> None:
        """Verify sweep paths launch sweep verify kernels."""
        config = make_config(sweep_interval=1)
        runner = make_runner(device=self.device, config=config)
        forward_batch = make_forward_batch(self.device)
        with runner.with_forward_pass(forward_batch):
            runner.launch_head_kernels(forward_batch)

        cache = make_radix_cache([[], [10, 11, 12]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        runner.attach_radix_cache(cache)

        sweep_kernel_kinds: list[str] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: sweep_kernel_kinds.append(
                kwargs["context"].kernel_kind.name
            ),
        ):
            runner._sweep_orchestrator.maybe_run_sweep()
        self.assertTrue(any("SWEEP" in kind for kind in sweep_kernel_kinds))

    def test_sweep_allocates_verify_plan_from_walker_output(self) -> None:
        """Verify sweep planning sizes the verify plan from walker output."""
        runner = make_runner(device=self.device)
        cache = make_radix_cache([[], [10, 11], [12, 13, 14]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        runner.attach_radix_cache(cache)

        valid_counts: list[int] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: valid_counts.append(
                int(kwargs["plan"].verify_num_valid.item())
            ),
        ):
            runner._sweep_orchestrator.maybe_run_sweep()
        self.assertTrue(all(count == 5 for count in valid_counts))


if __name__ == "__main__":
    unittest.main()
