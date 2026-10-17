"""CPU control-path tests for remote weight export groups."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components import weight_exporter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestWeightExporter(unittest.TestCase):
    def setUp(self):
        self.weights = torch.tensor([1.0, 2.0])
        self.model = Mock()
        self.model.named_parameters.return_value = [("weight", self.weights)]
        self.exporter = weight_exporter.WeightExporter(
            tp_rank=1,
            tp_size=2,
            gpu_id=1,
            get_model_path=lambda: "unused",
            get_model=Mock(return_value=self.model),
        )
        self.initialized = patch.object(
            weight_exporter.dist, "is_initialized", return_value=True
        )
        self.initialized.start()
        self.addCleanup(self.initialized.stop)
        platform = patch.object(
            weight_exporter, "current_platform", SimpleNamespace(empty_cache=Mock())
        )
        self.platform = platform.start()
        self.addCleanup(platform.stop)
        broadcast = patch.object(weight_exporter.dist, "broadcast")
        self.broadcast = broadcast.start()
        self.addCleanup(broadcast.stop)
        destroy = patch.object(
            weight_exporter.dist.distributed_c10d, "destroy_process_group"
        )
        self.destroy = destroy.start()
        self.addCleanup(destroy.stop)

    def send(self):
        return self.exporter.send_weights_to_remote_instance(
            "127.0.0.1", "12340,12341", "remote"
        )

    def assert_rejected(self):
        success, message = self.send()
        self.assertFalse(success)
        self.assertIn("remote_12341_1", message)
        self.assertIn("init_weights_send_group_for_remote_instance", message)
        self.exporter.get_model.assert_not_called()
        self.broadcast.assert_not_called()
        self.destroy.assert_not_called()
        self.platform.empty_cache.assert_not_called()

    def test_unknown_group_returns_error_without_collectives(self):
        other_group = object()
        self.exporter._weights_send_group["remote_12340_0"] = other_group
        self.assert_rejected()
        self.assertEqual(
            self.exporter._weights_send_group, {"remote_12340_0": other_group}
        )

    def test_none_group_preserves_existing_failure_response(self):
        self.exporter._weights_send_group["remote_12341_1"] = None
        self.assert_rejected()

    def test_registered_group_is_sent_and_consumed(self):
        group = object()
        self.exporter._weights_send_group["remote_12341_1"] = group
        success, message = self.send()
        self.assertTrue(success, message)
        self.broadcast.assert_called_once_with(self.weights, src=0, group=group)
        self.destroy.assert_called_once_with(group)
        self.assertNotIn("remote_12341_1", self.exporter._weights_send_group)

        self.broadcast.reset_mock()
        self.destroy.reset_mock()
        self.platform.empty_cache.reset_mock()
        self.exporter.get_model.reset_mock()
        self.assert_rejected()

    def test_broadcast_failure_still_releases_registered_group(self):
        group = object()
        self.exporter._weights_send_group["remote_12341_1"] = group
        self.broadcast.side_effect = RuntimeError("controlled broadcast failure")
        success, message = self.send()
        self.assertFalse(success)
        self.assertIn("controlled broadcast failure", message)
        self.destroy.assert_called_once_with(group)
        self.assertNotIn("remote_12341_1", self.exporter._weights_send_group)


if __name__ == "__main__":
    unittest.main()
