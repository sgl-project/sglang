"""EPLBManager must not rebalance expert weights that the weight cache shares."""

import unittest
from unittest.mock import Mock, call

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.runtime_context import get_context, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override, enter_scope

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.eplb.eplb_manager"
_WINDOW = 2


class TestEPLBManagerWeightCache(CustomTestCase):
    def _manager(self, weight_cache_mode: str) -> EPLBManager:
        enter_override(
            self,
            get_context().override_server_args(
                weight_cache_mode=weight_cache_mode,
                eplb_rebalance_num_iterations=_WINDOW,
                # Chunked, so the rebalance does not synchronize the device for timing.
                eplb_rebalance_layers_per_chunk=1,
                expert_distribution_recorder_buffer_size=_WINDOW,
                eplb_min_rebalancing_utilization_threshold=0.5,
            ),
        )
        self.recorder = Mock(recording=False)
        # A busy GPU, so a rebalance that does start stops right after this read.
        self.recorder.dump_record.return_value = {
            "logical_count": None,
            "average_utilization_rate_over_window": 1.0,
        }
        enter_scope(
            self, get_resources().override(expert_distribution_recorder=self.recorder)
        )
        self.get_expert_location_updater = Mock()
        return EPLBManager(
            model_config=None,
            # ModelRunner builds the manager before it loads the model.
            get_model=Mock(side_effect=AssertionError("the model is not loaded")),
            get_expert_location_updater=self.get_expert_location_updater,
            get_expert_backup_client=Mock(),
            get_weight_updater=Mock(),
        )

    def test_weight_cache_keeps_the_initial_layout(self):
        """Shared weights must never be rebalanced, whatever re-enables rebalancing;
        each window is still dumped, since nothing else resets the recorder."""
        windows = 3
        for mode in ("client", "daemon"):
            with self.subTest(mode=mode):
                with self.assertLogs(_LOGGER, "INFO") as logs:
                    manager = self._manager(mode)
                self.assertEqual([r.levelname for r in logs.records], ["WARNING"])
                self.assertIn("--weight-cache-mode off", logs.output[0])
                self.recorder.start_record.assert_called_once_with()

                # Elastic EP re-enables rebalancing after a scale; still none.
                manager.enable_rebalance()
                with self.assertNoLogs(_LOGGER, "INFO"):
                    for _ in range(windows * _WINDOW + 1):
                        manager.on_forward_pass_end()
                self.assertEqual(
                    self.recorder.dump_record.call_args_list,
                    [call(output_mode="object")] * windows,
                )
                self.get_expert_location_updater.assert_not_called()

    def test_rebalance_runs_without_the_weight_cache(self):
        """The guard must not turn EPLB off for engines that own their weights."""
        with self.assertLogs(_LOGGER, "INFO") as logs:
            manager = self._manager("off")
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        with self.assertLogs(_LOGGER, "INFO") as logs:
            self.assertEqual(list(manager.rebalance()), [])
        self.assertIn("rebalance start", logs.output[0])
        self.recorder.dump_record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
