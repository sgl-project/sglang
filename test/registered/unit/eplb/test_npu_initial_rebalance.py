"""First-period learning must not change steady scheduling or other backends."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.eplb import eplb_manager
from sglang.srt.hardware_backend.npu import eplb as npu_eplb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuInitialRebalance(unittest.TestCase):
    def manager(
        self,
        device="npu",
        nnodes=1,
        dispatch="static",
        layout="trivial",
        elastic=None,
        interval=640,
        window=320,
    ):
        moe = SimpleNamespace(
            eplb_rebalance_layers_per_chunk=None,
            eplb_rebalance_num_iterations=interval,
            expert_distribution_recorder_buffer_size=window,
            ep_dispatch_algorithm=dispatch,
            init_expert_location=layout,
            elastic_ep_backend=elastic,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    eplb_manager, "get_exec", return_value=SimpleNamespace(moe=moe)
                )
            )
            for name, value in (
                ("get_exec", SimpleNamespace(moe=moe)),
                ("get_device", SimpleNamespace(device=device)),
                ("get_parallel", SimpleNamespace(nnodes=nnodes)),
                ("get_global_expert_distribution_recorder", Mock(recording=True)),
            ):
                stack.enter_context(
                    patch.object(
                        eplb_manager
                        if name == "get_global_expert_distribution_recorder"
                        else npu_eplb,
                        name,
                        return_value=value,
                    )
                )
            return eplb_manager.EPLBManager(
                model_config=SimpleNamespace(),
                get_model=lambda: None,
                get_expert_location_updater=lambda: None,
                get_expert_backup_client=lambda: None,
                get_weight_updater=lambda: None,
            )

    def test_only_supported_initial_topology_uses_short_period(self):
        self.assertEqual(self.manager()._initial_rebalance_iterations, 64)
        for changes in (
            {"device": "cuda"},
            {"device": "cpu"},
            {"nnodes": 2},
            {"dispatch": "dynamic"},
            {"layout": "manual.json"},
            {"elastic": "existing"},
            {"interval": 64, "window": 32},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(self.manager(**changes)._initial_rebalance_iterations)

    def test_first_period_is_consumed_once_and_steady_interval_is_preserved(self):
        manager = self.manager()
        attempts = []

        def rebalance():
            attempts.append(True)
            yield from ()

        manager.rebalance = rebalance
        for _ in range(64):
            manager.on_forward_pass_end()
        self.assertEqual(len(attempts), 0)
        manager.on_forward_pass_end()
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(manager._initial_rebalance_iterations)
        for _ in range(639):
            manager.on_forward_pass_end()
        self.assertEqual(len(attempts), 1)
        manager.on_forward_pass_end()
        self.assertEqual(len(attempts), 2)
        manager.disable_rebalance("test")
        manager.enable_rebalance()
        self.assertIsNone(manager._initial_rebalance_iterations)

    def test_steady_window_invariant_is_not_relaxed(self):
        with self.assertRaises(AssertionError):
            self.manager(interval=64, window=320)


if __name__ == "__main__":
    unittest.main()
