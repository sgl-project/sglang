from __future__ import annotations

import unittest
from unittest.mock import patch

from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    make_forward_batch,
    make_radix_cache,
    make_req_to_token_pool,
)
from sglang.test.kv_canary.runner_test_base import (
    CanaryManagerTestCase,
    make_config,
    make_manager,
)

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=45, suite="extra-a-test-1-gpu-small-amd")


def _run_one_cycle(manager, forward_batch) -> None:
    with manager.with_ops_outside_graph(
        single_forward_indices=[0],
        maybe_inaccurate_forward_batch=forward_batch,
    ):
        with manager.with_active_single_forward_manager(0):
            pre_ops_output = manager.pre_ops_maybe_inside_graph(forward_batch)
            manager.post_ops_maybe_inside_graph(forward_batch, pre_ops_output)


class TestSelfUnitManagerSweep(CanaryManagerTestCase):
    def test_sweep_every_n_cadence(self) -> None:
        """Verify sweep execution follows the configured step cadence."""
        config = make_config(sweep_interval=4)
        manager = make_manager(device=self.device, config=config)
        forward_batch = make_forward_batch(self.device)

        sweep_calls: list[int] = []
        real_maybe = manager._sweep_orchestrator.maybe_run_sweep

        def _spy() -> None:
            before = manager._sweep_orchestrator._last_sweep_step
            real_maybe()
            if manager._sweep_orchestrator._last_sweep_step != before:
                sweep_calls.append(manager._outer_step_counter)

        with patch.object(manager._sweep_orchestrator, "maybe_run_sweep", _spy):
            for _ in range(12):
                _run_one_cycle(manager, forward_batch)
        self.assertEqual(sweep_calls, [0, 4, 8])

    def test_sweep_path_launches_sweep_kernels(self) -> None:
        """Verify sweep paths launch sweep verify kernels."""
        config = make_config(sweep_interval=1)
        manager = make_manager(device=self.device, config=config)
        forward_batch = make_forward_batch(self.device)
        manager._single_forward_managers[0].pre_ops_outside_graph(
            maybe_inaccurate_forward_batch=forward_batch
        )
        with manager.with_active_single_forward_manager(0):
            manager.pre_ops_maybe_inside_graph(forward_batch)

        cache = make_radix_cache([[], [10, 11, 12]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        manager.attach_radix_cache(cache)

        sweep_kernel_kinds: list[str] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: sweep_kernel_kinds.append(
                kwargs["context"].kernel_kind.name
            ),
        ):
            manager._sweep_orchestrator.maybe_run_sweep()
        self.assertTrue(any("SWEEP" in kind for kind in sweep_kernel_kinds))

    def test_sweep_allocates_verify_plan_from_walker_output(self) -> None:
        """Verify sweep planning sizes the verify plan from walker output."""
        manager = make_manager(device=self.device)
        cache = make_radix_cache([[], [10, 11], [12, 13, 14]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        manager.attach_radix_cache(cache)

        valid_counts: list[int] = []
        with patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: valid_counts.append(
                int(kwargs["plan"].verify_num_valid.item())
            ),
        ):
            manager._sweep_orchestrator.maybe_run_sweep()
        self.assertTrue(all(count == 5 for count in valid_counts))


if __name__ == "__main__":
    unittest.main()
