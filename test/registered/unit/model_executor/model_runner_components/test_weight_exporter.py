"""Unit tests for missing and consumed remote weight-export groups."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.model_runner_components.weight_exporter"


class TestWeightExporter(CustomTestCase):
    def setUp(self):
        self.get_model = Mock(return_value=torch.nn.Linear(2, 2, bias=False))
        self.exporter = WeightExporter(
            tp_rank=1,
            tp_size=2,
            gpu_id=1,
            get_model_path=lambda: "unused",
            get_model=self.get_model,
        )
        self.group_name = "export_29312_1"
        self.initialized = self.enterContext(
            patch(f"{_MODULE}.dist.is_initialized", return_value=True)
        )
        self.empty_cache = self.enterContext(
            patch(f"{_MODULE}.current_platform.empty_cache")
        )
        self.broadcast = self.enterContext(patch(f"{_MODULE}.dist.broadcast"))
        self.destroy_group = self.enterContext(
            patch(f"{_MODULE}.dist.distributed_c10d.destroy_process_group")
        )

    def send(self):
        return self.exporter.send_weights_to_remote_instance(
            "127.0.0.1", "29311,29312", "export"
        )

    def assert_missing_group(self):
        success, message = self.send()
        self.assertFalse(success)
        self.assertIn(self.group_name, message)
        self.assertIn("init_weights_send_group_for_remote_instance", message)
        self.get_model.assert_not_called()
        self.empty_cache.assert_not_called()
        self.broadcast.assert_not_called()
        self.destroy_group.assert_not_called()

    def test_unknown_group_returns_error(self):
        self.assert_missing_group()
        self.assertEqual(self.exporter._weights_send_group, {})

    def test_none_group_returns_error(self):
        self.exporter._weights_send_group[self.group_name] = None
        self.assert_missing_group()
        self.assertIsNone(self.exporter._weights_send_group[self.group_name])

    def test_send_consumes_group_and_repeat_returns_error(self):
        group = object()
        self.exporter._weights_send_group[self.group_name] = group
        success, _ = self.send()
        self.assertTrue(success)
        self.broadcast.assert_called_once_with(
            self.get_model.return_value.weight, src=0, group=group
        )
        self.destroy_group.assert_called_once_with(group)
        self.assertNotIn(self.group_name, self.exporter._weights_send_group)
        for mock in (
            self.get_model,
            self.empty_cache,
            self.broadcast,
            self.destroy_group,
        ):
            mock.reset_mock()
        self.assert_missing_group()


if __name__ == "__main__":
    unittest.main()
